import AVFoundation
import Foundation

/// Streams 16 kHz mono PCM s16le to the relay (wss://RelayHost/ws) and collects the transcript.
/// The relay sends {"type":"text","commit","pending"}: commit is appended for good, pending replaces the last pending.
/// A dropped connection reconnects by itself (1, 2, 4… up to 10 s apart) while the mic keeps recording; audio from the
/// gap is held (up to 60 s) and sent once the new stream is ready. Only stop() ends a session.
@MainActor
final class Transcriber: ObservableObject {
    enum State: Equatable { case idle, connecting, listening, reconnecting, failed(String) }

    @Published var committed = ""
    @Published var pending = ""
    @Published var state: State = .idle

    var isRunning: Bool { state == .connecting || state == .listening || state == .reconnecting }

    private var socket: URLSessionWebSocketTask?
    private var ready = false
    private var held: [Data] = []
    private var tries = 0
    private var retry: DispatchWorkItem?
    private var engine: AVAudioEngine?
    private var filePlayback: Task<Void, Never>?
    private let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!
    /// UI tests stream the bundled clip instead of the microphone (the simulator has no real mic).
    private let testAudio = ProcessInfo.processInfo.environment["CANTONESE_TEST_AUDIO"] == "1"
    /// UI tests cut the first connection 3 s in, to walk the reconnect.
    private var testDrop = ProcessInfo.processInfo.environment["CANTONESE_TEST_DROP"] == "1"

    func start() {
        guard !isRunning else { return }
        let info = Bundle.main.infoDictionary ?? [:]
        guard let host = info["RelayHost"] as? String, !host.isEmpty, URL(string: "wss://\(host)/ws") != nil else {
            state = .failed("未設定伺服器"); return
        }
        state = .connecting
        held = []
        tries = 0
        connect()
        do {
            if testAudio { streamTestFile() } else { try startMicrophone() }
        } catch {
            fail("用唔到咪：\(error.localizedDescription)")
        }
    }

    private func connect() {
        let info = Bundle.main.infoDictionary ?? [:]
        var request = URLRequest(url: URL(string: "wss://\(info["RelayHost"] as? String ?? "")/ws")!)
        request.setValue(info["AccessClientId"] as? String, forHTTPHeaderField: "CF-Access-Client-Id")
        request.setValue(info["AccessClientSecret"] as? String, forHTTPHeaderField: "CF-Access-Client-Secret")
        let task = URLSession.shared.webSocketTask(with: request)
        socket = task
        ready = false
        task.resume()
        receive(on: task)
        if testDrop {
            testDrop = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { task.cancel(with: .abnormalClosure, reason: nil) }
        }
    }

    /// The connection broke while running: keep what Soniox hadn't finished, start a new line, try again.
    private func reconnect(_ task: URLSessionWebSocketTask) {
        guard socket === task else { return }
        task.cancel(with: .goingAway, reason: nil)
        socket = nil
        committed += pending
        pending = ""
        if !committed.isEmpty && !committed.hasSuffix("\n") { committed += "\n" }
        state = .reconnecting
        let wait = min(pow(2, Double(tries)), 10)
        tries += 1
        let item = DispatchWorkItem { [weak self] in
            guard let self, self.state == .reconnecting else { return }
            self.connect()
        }
        retry = item
        DispatchQueue.main.asyncAfter(deadline: .now() + wait, execute: item)
    }

    /// Audio goes to the current stream once it is ready; until then (and between connections) it is held.
    private func send(_ data: Data) {
        if let task = socket, ready { task.send(.data(data)) { _ in }; return }
        held.append(data)
        var total = held.reduce(0) { $0 + $1.count }
        while total > 60 * 32000, !held.isEmpty { total -= held.removeFirst().count }   // 60 s at 32 kB/s
    }

    func stop() {
        retry?.cancel()
        retry = nil
        held = []
        stopAudio()
        guard let task = socket else { state = .idle; return }
        socket = nil
        task.send(.string("stop")) { _ in }
        // The relay closes after the last results; give up after 15 s.
        DispatchQueue.main.asyncAfter(deadline: .now() + 15) { task.cancel(with: .normalClosure, reason: nil) }
        state = .idle
    }

    func clear() { committed = ""; pending = "" }

    private func fail(_ message: String) {
        retry?.cancel()
        retry = nil
        stopAudio()
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        state = .failed(message)
    }

    private func stopAudio() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        filePlayback?.cancel()
        filePlayback = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func receive(on task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success(let message):
                    if case .string(let s) = message { self.handle(s, from: task) }
                    self.receive(on: task)
                case .failure:
                    self.reconnect(task)
                }
            }
        }
    }

    private func handle(_ json: String, from task: URLSessionWebSocketTask) {
        guard let data = json.data(using: .utf8),
              let m = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = m["type"] as? String else { return }
        switch type {
        case "ready":
            guard socket === task else { return }
            ready = true
            tries = 0
            state = .listening
            for data in held { task.send(.data(data)) { _ in } }
            held = []
        case "text":
            committed += (m["commit"] as? String) ?? ""
            pending = (m["pending"] as? String) ?? ""
        case "error": break   // the relay closes after an error, and the close reconnects
        case "done": pending = ""
        default: break
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
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.send(data) } }
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
                send(Data(pcm[offset..<end]))
                offset = end
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }
}
