import AVFoundation
import Foundation
import OpenCC
import os
import Qwen3ASR
import UIKit

/// Transcribes on the phone: Qwen3-ASR 1.7B (5-bit, aufklarer/Qwen3-ASR-1.7B-MLX-5bit) on the GPU through MLX, via
/// speech-swift. Nothing leaves the phone. iOS allows no GPU work in the background, so it works with the app on
/// screen: the screen stays awake while listening, and leaving the app stops it.
///
/// The model runs all the time, with no speech detection in front of it, streaming the way offline models are usually
/// streamed (LocalAgreement-2, as whisper_streaming does it): whenever it is free and 0.5 s more of the mic's 16 kHz
/// audio has come in, the audio since the line began is transcribed again, continuing from the text the last two passes
/// agreed on (only the rest is written, and shown token by token). What this pass and the last agree on is settled
/// (black); the rest is the grey tail. When passes stop finding new words (a pause) the line ends; 30 s without one, the
/// line is cut at its quietest 100 ms.
@MainActor
final class Transcriber: ObservableObject {
    enum State: Equatable { case idle, loading(String), listening, failed(String) }

    /// What has been said since the last clear, one line per segment, oldest first.
    @Published var lines: [Worker.Line] = []
    @Published var state: State = .idle

    var isRunning: Bool {
        switch state { case .loading, .listening: return true; default: return false }
    }

    private var engine: AVAudioEngine?
    private var sessionObservers: [NSObjectProtocol] = []
    private var filePlayback: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var worker: Worker?
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    /// UI tests stream a bundled clip instead of the microphone (the simulator has no real mic).
    private let testClip = TestClip.current
    private var testAudio: Bool { testClip != nil }

    init() {
        // The GPU is off limits in the background: leaving the app (or locking the phone) stops listening.
        NotificationCenter.default.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil,
                                               queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.stop() }
        }
    }

    func start() {
        guard !isRunning else { return }
        state = .loading("載入模型…")
        loadTask = Task {
            do {
                let worker = try await Worker.shared()
                try Task.checkCancellation()
                worker.onUpdate = { [weak self] lines in
                    Task { @MainActor in self?.lines = lines }
                }
                worker.begin()
                self.worker = worker
            } catch is CancellationError {
                return
            } catch {
                fail("用唔到模型：\(error.localizedDescription)")
                return
            }
            if !testAudio, !(await AVAudioApplication.requestRecordPermission()) {
                fail("未有權限用咪：去「設定」>「廣東話字幕」開返")
                return
            }
            do {
                if testAudio { streamTestFile() } else { try startMicrophone() }
                UIApplication.shared.isIdleTimerDisabled = true   // the screen stays on while listening
                state = .listening
            } catch {
                fail("用唔到咪：\(error.localizedDescription)")
            }
        }
    }

    func stop() {
        loadTask?.cancel()
        loadTask = nil
        stopAudio()
        worker?.end()
        state = .idle
    }

    func clear() {
        lines.removeAll { $0.final }
        worker?.clear()
    }

    private func fail(_ message: String) {
        stopAudio()
        worker?.end()
        state = .failed(message)
    }

    private func stopAudio() {
        UIApplication.shared.isIdleTimerDisabled = false
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        filePlayback?.cancel()
        filePlayback = nil
        for o in sessionObservers { NotificationCenter.default.removeObserver(o) }
        sessionObservers = []
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func startMicrophone() throws {
        // Always the iPhone's own mic, mixed with whatever else plays: Bluetooth headphones stay on A2DP (output only,
        // so they keep playing music in full quality and never become the mic); with no headphones other apps' audio
        // goes to the speaker, not the earpiece. A call or another app taking the audio session stops transcription.
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement,
                                options: [.mixWithOthers, .allowBluetoothA2DP, .defaultToSpeaker])
        try session.setActive(true)
        pinBuiltInMic()
        let center = NotificationCenter.default
        sessionObservers = [
            center.addObserver(forName: AVAudioSession.routeChangeNotification, object: session, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.pinBuiltInMic() }
            },
            center.addObserver(forName: AVAudioSession.interruptionNotification, object: session, queue: .main) { [weak self] note in
                let type = (note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt).flatMap(AVAudioSession.InterruptionType.init)
                guard type == .began else { return }
                MainActor.assumeIsolated { self?.stop() }
            },
            center.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: session, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.stop() }
            },
        ]
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: format, to: target) else { throw CocoaError(.featureUnsupported) }
        let worker = self.worker
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [target] buffer, _ in
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16000 / format.sampleRate) + 16
            guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var fed = false
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true; status.pointee = .haveData; return buffer
            }
            guard error == nil, out.frameLength > 0, let ch = out.floatChannelData else { return }
            worker?.feed(Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength))))
        }
        engine.prepare()
        try engine.start()
        self.engine = engine
    }

    private func pinBuiltInMic() {
        let session = AVAudioSession.sharedInstance()
        guard session.currentRoute.inputs.first?.portType != .builtInMic,
              let mic = session.availableInputs?.first(where: { $0.portType == .builtInMic }) else { return }
        try? session.setPreferredInput(mic)
    }

    private func streamTestFile() {
        guard let samples = testClip?.samples else { fail("搵唔到測試錄音"); return }
        let worker = self.worker
        Worker.log.notice("test audio starts")
        filePlayback = Task {
            var offset = 0
            while offset < samples.count, !Task.isCancelled {
                let end = min(offset + 1600, samples.count)   // 100 ms
                worker?.feed(Array(samples[offset..<end]))
                offset = end
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            // A second of silence so the last words end their line like a pause would.
            for _ in 0..<10 where !Task.isCancelled {
                worker?.feed([Float](repeating: 0, count: 1600))
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }
}

/// A bundled 16 kHz mono clip the UI tests stream instead of the microphone, with what the real model says for each of
/// its lines (testdata/*.json, taken from the model-check workflow) for the simulator's stand-in.
struct TestClip {
    struct Line: Decodable { let start: Double; let end: Double; let text: String }
    let samples: [Float]
    let lines: [Line]

    /// CANTONESE_TEST_AUDIO: "1" streams the short test clip, "demo" the App Store demo conversation.
    static let current: TestClip? = ProcessInfo.processInfo.environment["CANTONESE_TEST_AUDIO"]
        .flatMap { load($0 == "demo" ? "demo-16k" : "cantonese-16k") }

    private static func load(_ name: String) -> TestClip? {
        guard let wav = Bundle.main.url(forResource: name, withExtension: "wav").flatMap({ try? Data(contentsOf: $0) }),
              let json = Bundle.main.url(forResource: name, withExtension: "json").flatMap({ try? Data(contentsOf: $0) }),
              let lines = try? JSONDecoder().decode([Line].self, from: json) else { return nil }
        // 16-bit PCM from the "data" chunk: WAV headers are not always 44 bytes.
        var i = 12
        while i + 8 <= wav.count {
            let size = (0..<4).reduce(0) { $0 | Int(wav[i + 4 + $1]) << (8 * $1) }
            if wav[i..<(i + 4)] == Data("data".utf8) {
                let pcm = wav[(i + 8)..<min(wav.count, i + 8 + size)]
                let samples = pcm.withUnsafeBytes { raw in
                    (0..<(raw.count / 2)).map { Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 2, as: Int16.self))) / 32768 }
                }
                return TestClip(samples: samples, lines: lines)
            }
            i += 8 + size + (size & 1)
        }
        return nil
    }
}

/// The model and everything that touches it. Audio state lives on audioQueue, the model runs on modelQueue.
final class Worker: @unchecked Sendable {
    static let log = Logger(subsystem: "dev.de0ch.cantonese", category: "model")

    /// One line on screen: `settled` never changes again, `tail` is still being revised (shown grey). A final line has
    /// no tail.
    struct Line: Identifiable { let id: Int; var settled: String; var tail: String; var final: Bool }
    /// A model job: transcribe `audio` continuing from `start`; a final one closes its line.
    private struct Job { let id: Int; let audio: [Float]; let start: String; let final: Bool }
    typealias ASR = (_ audio: [Float], _ start: String, _ onText: @escaping @Sendable (String) -> Void) throws -> String

    /// Every line on screen, after each change.
    var onUpdate: (([Line]) -> Void)?

    private let asr: ASR
    private let convert: ChineseConverter
    private let audioQueue = DispatchQueue(label: "cantonese.audio")
    private let modelQueue = DispatchQueue(label: "cantonese.model", qos: .userInitiated)

    // audioQueue state. The current line is `buffer` (its audio since the line began) and its model text so far:
    // `black` is what the last two passes agreed on (LocalAgreement-2), `previous` the last pass's whole text.
    private var lines: [Line] = []
    private var buffer: [Float] = []
    private var currentId = 0
    private var black = ""
    private var previous = ""
    private var shownTail = ""            // model text of the grey tail on screen
    private var unchanged = 0             // passes in a row that heard no new words
    private var lastPass = 0              // buffer.count at the last pass
    private var ended: [Job] = []         // closed lines waiting for their last pass
    private var keepFrom = 0              // lines below this id were cleared
    private var busy = false
    private var running = false

    private static let passEvery = 8_000             // a pass once 0.5 s more audio is in and the model is free
    private static let closeAfter = 2                // two more passes with no new words: a pause, the line closes
    private static let maxSamples = 30 * 16000       // 30 s without a pause: the line is cut...
    private static let cutWindow = 3 * 16000         // ...at the quietest 100 ms of its last 3 s
    private static let quietFrame = 1600
    private static let cutRollback = 10              // characters the cut line's last pass may rewrite

    private static var cached: Worker?

    /// Loads the model shipped inside the app (ios/fetch-model.sh puts it in Model/) once and keeps it.
    static func shared() async throws -> Worker {
        if let w = cached { return w }
        guard let dir = Bundle.main.url(forResource: "Model", withExtension: nil) else {
            throw CocoaError(.fileNoSuchFile)
        }
        #if targetEnvironment(simulator)
        // The simulator has no Metal for MLX, so the UI tests run everything but the model: this stand-in finds where the
        // audio sits in the bundled clip and answers with what the real model says for the clip's lines whose middle falls
        // inside it (TestClip; the model-check workflow hears the real model on a Mac).
        _ = dir
        let asr: ASR = { audio, start, onText in
            guard let clip = TestClip.current else { return start }
            // Match 10 ms from the first sound in the audio (the silence after the clip matches nothing: no words).
            guard let k = audio.firstIndex(where: { $0 != 0 }), k + 160 <= audio.count else { return start }
            let probe = audio[k..<(k + 160)]
            guard let at = (0...max(0, clip.samples.count - 160)).first(where: { o in
                zip(clip.samples[o..<(o + 160)], probe).allSatisfy { $0 == $1 }
            }) else { return start }
            let from = Double(at - k) / 16000, to = Double(at - k + audio.count) / 16000
            let said = clip.lines.filter { ($0.start + $0.end) / 2 >= from && ($0.start + $0.end) / 2 < to }.map(\.text).joined()
            let text = said.hasPrefix(start) ? said : start + said   // like the model, it continues from `start`
            onText(text)
            return text
        }
        #else
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-5bit", cacheDir: dir.appendingPathComponent("asr"), offlineMode: true)
        let asr: ASR = { audio, start, onText in
            var options = Qwen3DecodingOptions()
            options.maxTokens = 200
            options.prefix = start
            options.onText = onText
            return try model.transcribeThrowing(audio: audio, sampleRate: 16000, options: options)
        }
        #endif
        let convert = try ChineseConverter(options: [.traditionalize, .hkStandard])
        log.notice("model loaded")
        let w = Worker(asr: asr, convert: convert)
        cached = w
        return w
    }

    private init(asr: @escaping ASR, convert: ChineseConverter) {
        self.asr = asr
        self.convert = convert
    }

    func begin() {
        audioQueue.async { [self] in
            running = true
            resetLine()
        }
    }

    /// Stop listening: the line being spoken gets its last pass and is finished.
    func end() {
        audioQueue.async { [self] in
            running = false
            close(at: buffer.count, start: black)
        }
    }

    /// Clear the screen: everything already finished goes, the line being spoken carries on.
    func clear() {
        audioQueue.async { [self] in
            ended = []
            keepFrom = currentId
            lines.removeAll { $0.id != currentId }
            onUpdate?(lines)
        }
    }

    func feed(_ samples: [Float]) {
        audioQueue.async { [self] in
            guard running else { return }
            buffer += samples
            schedule()
        }
    }

    private func resetLine() {
        buffer = []; black = ""; previous = ""; shownTail = ""; unchanged = 0; lastPass = 0
    }

    /// The current line's audio before `at` gets one last pass from `start` and is finished; the rest starts a new line.
    private func close(at: Int, start: String) {
        if at > 0 {
            Self.log.notice("line \(self.currentId) closes: \(at / 16) ms")
            ended.append(Job(id: currentId, audio: Array(buffer[..<at]), start: start, final: true))
        }
        let rest = Array(buffer[at...])
        currentId += 1
        resetLine()
        buffer = rest
        schedule()
    }

    /// Where in the last 3 s of the buffer the quietest 100 ms is (Qwen's own long-audio split works the same way).
    private func quietest() -> Int {
        var best = buffer.count, bestEnergy = Float.infinity
        var i = max(0, buffer.count - Self.cutWindow)
        while i + Self.quietFrame <= buffer.count {
            var e: Float = 0
            for j in i..<(i + Self.quietFrame) { e += buffer[j] * buffer[j] }
            if e < bestEnergy { bestEnergy = e; best = i + Self.quietFrame / 2 }
            i += Self.quietFrame / 4
        }
        return best
    }

    /// Start the next model job if the model is free: a closed line's last pass first, else a pass over the current line.
    private func schedule() {
        guard !busy else { return }
        let job: Job
        if !ended.isEmpty {
            job = ended.removeFirst()
        } else if running, buffer.count >= 4000, buffer.count - lastPass >= Self.passEvery {
            lastPass = buffer.count
            job = Job(id: currentId, audio: buffer, start: black, final: false)
        } else { return }
        busy = true
        modelQueue.async { [self] in
            // The tail fills in token by token while the model writes it.
            let text = transcribe(job) { [weak self] sofar in
                self?.audioQueue.async { self?.showPartial(job, sofar) }
            }
            audioQueue.async { [self] in
                busy = false
                if job.final { finish(job, text) } else { passed(job, text) }
                schedule()
            }
        }
    }

    /// A pass over the current line ended: black is now what this pass and the last agree on (LocalAgreement-2).
    private func passed(_ job: Job, _ text: String) {
        guard job.id == currentId else { return }    // the line was closed meanwhile; its last pass decides
        let agreed = Self.agreed(text, previous, atLeast: black)
        unchanged = text == previous && !text.isEmpty ? unchanged + 1 : 0
        black = agreed
        previous = text
        shownTail = String(text.dropFirst(black.count))
        show(currentId, settled: black, tail: shownTail, final: false)
        if unchanged >= Self.closeAfter {
            // No new words for a while: everything is settled, the line ends here and a fresh one starts.
            Self.log.notice("line \(self.currentId) settles after a pause")
            show(currentId, settled: text, tail: "", final: true)
            currentId += 1
            resetLine()
        } else if text.isEmpty, buffer.count > 3 * 16000 {
            resetLine()                               // only silence so far: drop it
        } else if buffer.count >= Self.maxSamples {
            close(at: quietest(), start: String(black.dropLast(Self.cutRollback)))
        }
    }

    /// A closed line's last pass ended.
    private func finish(_ job: Job, _ text: String) {
        show(job.id, settled: text, tail: "", final: true)
    }

    /// Show the text a pass is writing, once it is at least as long as the grey tail it replaces (no flicker).
    private func showPartial(_ job: Job, _ sofar: String) {
        guard !job.final, job.id == currentId, sofar.hasPrefix(job.start) else { return }
        let tail = String(sofar.dropFirst(job.start.count))
        guard job.start.count + tail.count >= black.count + shownTail.count else { return }
        show(job.id, settled: job.start, tail: tail, final: false)
    }

    /// Put a line on screen in HK Traditional. A cleared line doesn't come back; a finished line with no words leaves.
    private func show(_ id: Int, settled: String, tail: String, final: Bool) {
        let line = Line(id: id, settled: convert.convert(settled), tail: convert.convert(tail), final: final)
        if let i = lines.firstIndex(where: { $0.id == id }) {
            if final && settled.isEmpty { lines.remove(at: i) } else { lines[i] = line }
        } else if id >= keepFrom, !(settled + tail).isEmpty {
            lines.append(line)
            lines.sort { $0.id < $1.id }
        } else { return }
        onUpdate?(lines)
    }

    /// The longest common start of two passes' texts, never inside an English word and never shorter than `atLeast`
    /// (the start both passes were given).
    static func agreed(_ a: String, _ b: String, atLeast: String) -> String {
        var n = zip(a, b).prefix { $0 == $1 }.count
        let ca = Array(a)
        let isWord = { (c: Character) in c.isASCII && (c.isLetter || c.isNumber) }
        if n < ca.count, n > 0, isWord(ca[n - 1]), isWord(ca[n]) {
            while n > 0, isWord(ca[n - 1]) { n -= 1 }
        }
        return n >= atLeast.count ? String(ca[..<n]) : atLeast
    }

    private func transcribe(_ job: Job, onText: @escaping @Sendable (String) -> Void) -> String {
        guard job.audio.count >= 4000 else { return job.start }
        let t0 = Date()
        do {
            let raw = try asr(job.audio, job.start, onText).trimmingCharacters(in: .whitespacesAndNewlines)
            Self.log.notice("pass over \(job.audio.count / 16) ms from \(job.start.count) chars in \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(raw, privacy: .private)")
            return raw
        } catch {
            Self.log.error("transcription failed after \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(error.localizedDescription, privacy: .public)")
            return job.start
        }
    }
}
