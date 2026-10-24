import AVFoundation
import Foundation

/// Streams 16 kHz mono PCM s16le to the relay (wss://RelayHost/ws?style=raw|written) and collects the transcript.
/// The relay sends {"type":"text","commit","pending"}: commit is appended for good ("\n" ends a line), pending replaces
/// the last pending. With style=written it also sends {"type":"rewrite","line","text"}: line n of this stream in 書面語.
@MainActor
final class Transcriber: ObservableObject {
    enum Style: String, CaseIterable, Identifiable {
        case raw, written
        var id: String { rawValue }
        var label: String { self == .raw ? "原文" : "書面語" }
    }
    enum State: Equatable { case idle, connecting, listening, failed(String) }

    @Published private(set) var lines = [""]
    var committed: String { lines.joined(separator: "\n") }
    @Published var pending = ""
    @Published var state: State = .idle
    @Published var style: Style {
        didSet {
            UserDefaults.standard.set(style.rawValue, forKey: "style")
            if oldValue != style, isRunning { stop(); start() }   // switching mid-stream restarts in the new style
        }
    }
    /// lines[base + n] is line n of the current stream.
    private var base = 0

    var isRunning: Bool { state == .connecting || state == .listening }

    private var socket: URLSessionWebSocketTask?
    private var engine: AVAudioEngine?
    private var filePlayback: Task<Void, Never>?
    private let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!
    /// UI tests stream the bundled clip instead of the microphone (the simulator has no real mic).
    private let testAudio = ProcessInfo.processInfo.environment["CANTONESE_TEST_AUDIO"] == "1"

    init() {
        style = Style(rawValue: UserDefaults.standard.string(forKey: "style") ?? "") ?? .raw
    }

    func start() {
        guard !isRunning else { return }
        state = .connecting
        let info = Bundle.main.infoDictionary ?? [:]
        let host = (info["RelayHost"] as? String) ?? ""
        if lines.last?.isEmpty == false { lines.append("") }
        base = lines.count - 1
        guard !host.isEmpty, let url = URL(string: "wss://\(host)/ws?style=\(style.rawValue)") else {
            state = .failed("未設定伺服器"); return
        }
        var request = URLRequest(url: url)
        request.setValue(info["AccessClientId"] as? String, forHTTPHeaderField: "CF-Access-Client-Id")
        request.setValue(info["AccessClientSecret"] as? String, forHTTPHeaderField: "CF-Access-Client-Secret")
        let task = URLSession.shared.webSocketTask(with: request)
        socket = task
        task.resume()
        receive(on: task)
        do {
            if testAudio { streamTestFile(to: task) } else { try startMicrophone(to: task) }
        } catch {
            fail("用唔到咪：\(error.localizedDescription)")
        }
    }

    func stop() {
        guard let task = socket else { state = .idle; return }
        socket = nil
        stopAudio()
        task.send(.string("stop")) { _ in }
        // The relay closes after the last results (and 書面語 rewrites); give up after 15 s.
        DispatchQueue.main.asyncAfter(deadline: .now() + 15) { task.cancel(with: .normalClosure, reason: nil) }
        state = .idle
    }

    func clear() { lines = [""]; base = 0; pending = "" }

    private func fail(_ message: String) {
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
                case .failure(let error):
                    if self.socket === task { self.fail("連線斷咗：\(error.localizedDescription)") }
                }
            }
        }
    }

    private func handle(_ json: String, from task: URLSessionWebSocketTask) {
        guard let data = json.data(using: .utf8),
              let m = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = m["type"] as? String else { return }
        switch type {
        case "ready": if socket === task { state = .listening }
        case "text":
            let parts = ((m["commit"] as? String) ?? "").components(separatedBy: "\n")
            lines[lines.count - 1] += parts[0]
            lines += parts.dropFirst()
            pending = (m["pending"] as? String) ?? ""
        case "rewrite":
            if let n = m["line"] as? Int, let text = m["text"] as? String, lines.indices.contains(base + n) {
                lines[base + n] = text
            }
        case "error": if socket === task { fail((m["message"] as? String) ?? "出錯") }
        case "done": pending = ""
        default: break
        }
    }

    private func startMicrophone(to task: URLSessionWebSocketTask) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setActive(true)
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: format, to: target) else { throw CocoaError(.featureUnsupported) }
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [target] buffer, _ in
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16000 / format.sampleRate) + 16
            guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var fed = false
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true; status.pointee = .haveData; return buffer
            }
            guard error == nil, out.frameLength > 0, let ch = out.int16ChannelData else { return }
            task.send(.data(Data(bytes: ch[0], count: Int(out.frameLength) * 2))) { _ in }
        }
        engine.prepare()
        try engine.start()
        self.engine = engine
    }

    private func streamTestFile(to task: URLSessionWebSocketTask) {
        guard let url = Bundle.main.url(forResource: "cantonese-16k", withExtension: "wav"),
              let wav = try? Data(contentsOf: url) else { fail("搵唔到測試錄音"); return }
        let pcm = wav.dropFirst(44)
        filePlayback = Task {
            var offset = pcm.startIndex
            while offset < pcm.endIndex, !Task.isCancelled {
                let end = min(offset + 3200, pcm.endIndex)   // 100 ms
                task.send(.data(Data(pcm[offset..<end]))) { _ in }
                offset = end
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }
}
