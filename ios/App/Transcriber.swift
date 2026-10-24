import AVFoundation
import Foundation

/// Streams 16 kHz mono PCM s16le to the relay (wss://RelayHost/ws) and collects the transcript.
/// The relay sends {"type":"text","commit","pending","final_ms"}: commit is appended for good, pending replaces the
/// last pending, final_ms is how much of this connection's audio Soniox has finalized; and {"type":"ping"} every second.
///
/// Network drops are invisible. The app keeps every bit of audio Soniox hasn't finalized and treats a connection as
/// disposable. The audio is cut into segments, each transcribed by its own connection; normally there is one, the live
/// segment. When a connection dies (an error, a close, or 4 s without even a ping) the app opens a new one at once and
/// replays the segment's audio from its last finalized point. Soniox works through replayed audio at only about 1.2×
/// real time, so a live segment that comes back with more than 1.5 s to catch up is split at the quietest moment of the
/// last second (so no word is cut in two): that connection finishes the backlog and a fresh one carries on live. Segments show in audio order, so nothing is lost, repeated or out
/// of place, and only a gap over 10 s shows a note. stop() also survives a drop.
@MainActor
final class Transcriber: ObservableObject {
    enum State: Equatable { case idle, connecting, listening, unstable, failed(String) }

    /// One segment's text: commit is final, pending may still change.
    struct Part: Identifiable { let id: ObjectIdentifier; let commit: String; let pending: String }

    @Published var committed = ""
    @Published var parts: [Part] = []
    @Published var state: State = .idle

    var isRunning: Bool { state == .connecting || state == .listening || state == .unstable }

    /// The audio [from, to) (to = nil: the live segment) and the connection transcribing it.
    private final class Segment {
        let from: Int
        var to: Int?
        var finalAt: Int
        var streamAt: Int
        var commit = ""
        var pending = ""
        var socket: URLSessionWebSocketTask?
        var live = false
        var lastHeard = Date()
        var tries = 0
        var done = false
        init(from: Int, to: Int?) { self.from = from; self.to = to; finalAt = from; streamAt = from }
    }

    private static let bytesPerMs = 32   // 16 kHz × 2 bytes
    private static let splitAt = 1500 * bytesPerMs
    private var segments: [Segment] = []
    /// Audio not yet finalized by every segment, as (position in the session's audio, bytes).
    private var chunks: [(at: Int, data: Data)] = []
    private var recorded = 0
    private var downSince: Date?
    private var stopping: Date?
    private var active = false
    private var watchdog: Timer?
    private var engine: AVAudioEngine?
    private var filePlayback: Task<Void, Never>?
    private let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!
    /// UI tests stream the bundled clip instead of the microphone (the simulator has no real mic).
    private let testAudio = ProcessInfo.processInfo.environment["CANTONESE_TEST_AUDIO"] == "1"
    /// UI tests cut the first connection 3 s in and hold the network down for CANTONESE_TEST_DROP seconds.
    private var testDrop = Double(ProcessInfo.processInfo.environment["CANTONESE_TEST_DROP"] ?? "")
    private var networkDownUntil = Date.distantPast

    private var liveSegment: Segment? { segments.first { $0.to == nil } }

    func start() {
        guard !isRunning else { return }
        if active { finish() }   // the previous session was still sending its last words
        let host = Bundle.main.infoDictionary?["RelayHost"] as? String ?? ""
        guard !host.isEmpty, URL(string: "wss://\(host)/ws") != nil else { state = .failed("未設定伺服器"); return }
        state = .connecting
        chunks = []; recorded = 0; downSince = nil; stopping = nil; segments = []
        active = true
        watchdog = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.watch() }
        }
        addSegment(from: 0, to: nil, after: nil)
        do {
            if testAudio { streamTestFile() } else { try startMicrophone() }
        } catch {
            fail("用唔到咪：\(error.localizedDescription)")
        }
    }

    func stop() {
        guard active, stopping == nil else { return }
        stopAudio()
        stopping = Date()
        state = .idle
        guard let g = liveSegment else { return }
        g.to = recorded
        if g.live, let task = g.socket { g.live = false; task.send(.string("stop")) { _ in } }   // else "ready" sends it
    }

    func clear() {
        committed = ""
        for g in segments { g.commit = ""; g.pending = "" }
        publish()
    }

    private func publish() {
        parts = segments.map { Part(id: ObjectIdentifier($0), commit: $0.commit, pending: $0.pending) }
    }

    @discardableResult
    private func addSegment(from: Int, to: Int?, after: Segment?) -> Segment {
        let g = Segment(from: from, to: to)
        if let after, let i = segments.firstIndex(where: { $0 === after }) { segments.insert(g, at: i + 1) } else { segments.append(g) }
        connect(g)
        return g
    }

    private func connect(_ g: Segment) {
        guard active, !g.done else { return }
        let info = Bundle.main.infoDictionary ?? [:]
        var request = URLRequest(url: URL(string: "wss://\(info["RelayHost"] as? String ?? "")/ws")!)
        request.setValue(info["AccessClientId"] as? String, forHTTPHeaderField: "CF-Access-Client-Id")
        request.setValue(info["AccessClientSecret"] as? String, forHTTPHeaderField: "CF-Access-Client-Secret")
        if Date() < networkDownUntil {   // UI test: the network is still "down"
            request.url = URL(string: "wss://127.0.0.1:9/ws")
        }
        let task = URLSession.shared.webSocketTask(with: request)
        g.socket = task
        g.live = false
        g.lastHeard = Date()
        task.resume()
        receive(on: task, for: g)
        if let down = testDrop {
            testDrop = nil
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { [weak self] in
                MainActor.assumeIsolated { self?.networkDownUntil = Date().addingTimeInterval(down) }
                task.cancel(with: .abnormalClosure, reason: nil)
            }
        }
    }

    /// A segment's connection is gone: forget it and open a new one (at once, then 0.5, 1, 2… up to 5 s apart).
    private func drop(_ g: Segment, _ task: URLSessionWebSocketTask) {
        guard active, !g.done, g.socket === task else { return }
        g.socket = nil
        g.live = false
        task.cancel(with: .goingAway, reason: nil)
        if g.to == nil, downSince == nil { downSince = Date() }
        let wait = g.tries == 0 ? 0 : min(0.25 * pow(2, Double(g.tries)), 5)
        g.tries += 1
        DispatchQueue.main.asyncAfter(deadline: .now() + wait) { [weak self] in MainActor.assumeIsolated { self?.connect(g) } }
    }

    private func watch() {
        guard active else { return }
        for g in segments { if let task = g.socket, Date().timeIntervalSince(g.lastHeard) > 4 { drop(g, task) } }
        if let down = downSince, Date().timeIntervalSince(down) > 10, stopping == nil { state = .unstable }
        if let s = stopping, Date().timeIntervalSince(s) > 60 { finish() }
    }

    /// The middle of the quietest 20 ms in the audio [lo, hi).
    private func quietest(from lo: Int, to hi: Int) -> Int {
        let window = 20 * Self.bytesPerMs / 2   // samples
        var best = hi, bestEnergy = Double.infinity, energy = 0.0, n = 0, at = lo
        for c in chunks where c.at + c.data.count > lo && c.at < hi {
            c.data.withUnsafeBytes { raw in
                let pcm = raw.bindMemory(to: Int16.self)
                var i = max(0, (lo - c.at) / 2)
                while i < pcm.count, c.at + 2 * i < hi {
                    if n == 0 { at = c.at + 2 * i }
                    energy += Double(pcm[i]) * Double(pcm[i])
                    n += 1
                    if n == window {
                        if energy < bestEnergy { bestEnergy = energy; best = at + window }
                        energy = 0; n = 0
                    }
                    i += 1
                }
            }
        }
        return best
    }

    /// Audio every segment has finalized is no longer needed.
    private func trim() {
        let need = segments.map(\.finalAt).min() ?? recorded
        while let first = chunks.first, first.at + first.data.count <= need { chunks.removeFirst() }
    }

    /// Finished segments at the front become plain text; when a stopped session has none left, it is over.
    private func settle() {
        while let g = segments.first, g.done { committed += g.commit; segments.removeFirst() }
        trim()
        publish()
        if stopping != nil, segments.isEmpty { finish() }
    }

    /// The session is over: whatever is still pending stays as text.
    private func finish() {
        guard active else { return }
        active = false
        watchdog?.invalidate(); watchdog = nil
        for g in segments { g.socket?.cancel(with: .normalClosure, reason: nil); g.socket = nil; committed += g.commit + g.pending }
        segments = []
        chunks = []
        publish()
    }

    private func fail(_ message: String) {
        stopAudio()
        finish()
        state = .failed(message)
    }

    /// New audio from the mic: kept until every segment has it finalized, sent now if the live segment is live.
    private func record(_ data: Data) {
        guard active, stopping == nil else { return }
        chunks.append((recorded, data))
        recorded += data.count
        if let g = liveSegment, g.live, let task = g.socket { task.send(.data(data)) { _ in } }
    }

    private func stopAudio() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        filePlayback?.cancel()
        filePlayback = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func receive(on task: URLSessionWebSocketTask, for g: Segment) {
        task.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success(let message):
                    if case .string(let s) = message { self.handle(s, from: task, for: g) }
                    self.receive(on: task, for: g)
                case .failure:
                    self.drop(g, task)
                }
            }
        }
    }

    private func handle(_ json: String, from task: URLSessionWebSocketTask, for g: Segment) {
        guard g.socket === task,
              let data = json.data(using: .utf8),
              let m = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = m["type"] as? String else { return }
        g.lastHeard = Date()
        switch type {
        case "ready":
            if g.to == nil, stopping == nil, recorded - g.finalAt > Self.splitAt {
                let cut = quietest(from: max(g.finalAt, recorded - 1000 * Self.bytesPerMs), to: recorded)
                g.to = cut                                          // this connection finishes the backlog…
                addSegment(from: cut, to: nil, after: g)            // …and a fresh one carries on live
            }
            // The stream starts at the segment's last finalized point: replay from there, then go live or finish.
            g.streamAt = g.finalAt
            let upTo = g.to ?? recorded
            for c in chunks where c.at + c.data.count > g.streamAt && c.at < upTo {
                let lo = max(0, g.streamAt - c.at), hi = min(c.data.count, upTo - c.at)
                task.send(.data(c.data.subdata(in: c.data.startIndex + lo ..< c.data.startIndex + hi))) { _ in }
            }
            if g.to == nil { g.live = true } else { task.send(.string("stop")) { _ in } }
            g.tries = 0
            if g.to == nil || liveSegment == nil {
                downSince = nil
                if stopping == nil { state = .listening }
            }
        case "text":
            g.commit += (m["commit"] as? String) ?? ""
            g.pending = (m["pending"] as? String) ?? ""
            let finalMs = (m["final_ms"] as? NSNumber)?.intValue ?? 0
            g.finalAt = max(g.finalAt, g.streamAt + finalMs * Self.bytesPerMs)
            trim()
            publish()
        case "done":
            if g.to == nil { drop(g, task); return }   // Soniox ended the live stream on its own: carry on
            g.done = true
            g.socket = nil
            g.pending = ""
            settle()
        default: break   // "ping"; "error" is followed by a close, which reconnects
        }
    }

    private func startMicrophone() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setActive(true)
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: format, to: target) else { throw CocoaError(.featureUnsupported) }
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self, target] buffer, _ in
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16000 / format.sampleRate) + 16
            guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var fed = false
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true; status.pointee = .haveData; return buffer
            }
            guard error == nil, out.frameLength > 0, let ch = out.int16ChannelData else { return }
            let data = Data(bytes: ch[0], count: Int(out.frameLength) * 2)
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.record(data) } }
        }
        engine.prepare()
        try engine.start()
        self.engine = engine
    }

    private func streamTestFile() {
        guard let url = Bundle.main.url(forResource: "cantonese-16k", withExtension: "wav"),
              let wav = try? Data(contentsOf: url) else { fail("搵唔到測試錄音"); return }
        let pcm = wav.dropFirst(44)
        filePlayback = Task {
            var offset = pcm.startIndex
            while offset < pcm.endIndex, !Task.isCancelled {
                let end = min(offset + 3200, pcm.endIndex)   // 100 ms
                record(Data(pcm[offset..<end]))
                offset = end
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }
}
