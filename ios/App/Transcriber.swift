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
/// The model runs all the time, with no speech detection in front of it: whenever it is free and 0.5 s more of the mic's
/// 16 kHz audio has come in, everything since the last finished line is transcribed again as the pending line. Once that
/// audio reaches 6 s it is cut at its quietest 20 ms in the last 2 s; the part before the cut is transcribed once more
/// and becomes a finished line. The model works on one job at a time on its own queue; a finished line always goes
/// before a pending refresh.
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
    /// UI tests stream the bundled clip instead of the microphone (the simulator has no real mic).
    private let testAudio = ProcessInfo.processInfo.environment["CANTONESE_TEST_AUDIO"] == "1"

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
        guard let url = Bundle.main.url(forResource: "cantonese-16k", withExtension: "wav"),
              let wav = try? Data(contentsOf: url) else { fail("搵唔到測試錄音"); return }
        let samples = wav.dropFirst(44).withUnsafeBytes { raw in
            raw.bindMemory(to: Int16.self).map { Float($0) / 32768 }
        }
        let worker = self.worker
        filePlayback = Task {
            var offset = 0
            while offset < samples.count, !Task.isCancelled {
                let end = min(offset + 1600, samples.count)   // 100 ms
                worker?.feed(Array(samples[offset..<end]))
                offset = end
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            // A second of silence so the last words end their segment like a pause would.
            for _ in 0..<10 where !Task.isCancelled {
                worker?.feed([Float](repeating: 0, count: 1600))
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }
}

/// The model and everything that touches it. Audio state lives on audioQueue, the model runs on modelQueue.
final class Worker: @unchecked Sendable {
    static let log = Logger(subsystem: "dev.de0ch.cantonese", category: "model")

    /// One line's text: final once its audio has been transcribed for the last time, pending (and may change) until then.
    struct Line: Identifiable { let id: Int; var text: String; var final: Bool }
    private struct Segment { let id: Int; let audio: [Float] }

    /// Every line on screen, after each change.
    var onUpdate: (([Line]) -> Void)?

    private let asr: ([Float]) throws -> String
    private let convert: ChineseConverter
    private let audioQueue = DispatchQueue(label: "cantonese.audio")
    private let modelQueue = DispatchQueue(label: "cantonese.model", qos: .userInitiated)

    // audioQueue state
    private var lines: [Line] = []
    private var buffer: [Float] = []      // audio since the last finished line
    private var currentId = 0             // the line buffer becomes
    private var ended: [Segment] = []     // cut off, waiting for their final transcription
    private var lastPartial = 0           // buffer.count at the last pending transcription
    private var keepFrom = 0              // lines below this id were cleared
    private var busy = false
    private var running = false

    private static let partialEvery = 8_000          // refresh pending once 0.5 s more audio is in and the model is free
    private static let maxSamples = 6 * 16000        // a line is cut once its audio reaches 6 s
    private static let cutWindow = 2 * 16000         // ...at the quietest 20 ms of its last 2 s
    private static let quietFrame = 320

    private static var cached: Worker?

    /// Loads the model shipped inside the app (ios/fetch-model.sh puts it in Model/) once and keeps it.
    static func shared() async throws -> Worker {
        if let w = cached { return w }
        guard let dir = Bundle.main.url(forResource: "Model", withExtension: nil) else {
            throw CocoaError(.fileNoSuchFile)
        }
        #if targetEnvironment(simulator)
        // The simulator has no Metal for MLX, so the UI tests run everything but the model: this stand-in finds where the
        // audio sits in the bundled clip and answers with the clauses of what the model says for it (in its Simplified
        // output, so OpenCC still runs) whose middle falls inside it. The real model is heard on a Mac by model-check.
        _ = dir
        let clip = Bundle.main.url(forResource: "cantonese-16k", withExtension: "wav").flatMap { try? Data(contentsOf: $0) }
            .map { $0.dropFirst(44).withUnsafeBytes { raw in raw.bindMemory(to: Int16.self).map { Float($0) / 32768 } } } ?? []
        let clauses: [(text: String, middle: Double)] = [("我哋今日去咗饮茶，", 0.28), ("啲虾饺好好食。", 0.78)]
        let asr: ([Float]) throws -> String = { audio in
            // Match 10 ms from the first sound in the audio (the silence after the clip matches nothing: no words).
            guard let k = audio.firstIndex(where: { $0 != 0 }), k + 160 <= audio.count else { return "" }
            let probe = audio[k..<(k + 160)]
            guard let at = (0...max(0, clip.count - 160)).first(where: { o in
                zip(clip[o..<(o + 160)], probe).allSatisfy { $0 == $1 }
            }) else { return "" }
            let from = Double(at - k) / Double(clip.count), to = Double(at - k + audio.count) / Double(clip.count)
            return clauses.filter { $0.middle >= from && $0.middle < to }.map(\.text).joined()
        }
        #else
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-5bit", cacheDir: dir.appendingPathComponent("asr"), offlineMode: true)
        let asr: ([Float]) throws -> String = { audio in
            var options = Qwen3DecodingOptions()
            options.maxTokens = 200
            return try model.transcribeThrowing(audio: audio, sampleRate: 16000, options: options)
        }
        #endif
        let convert = try ChineseConverter(options: [.traditionalize, .hkStandard])
        log.notice("model loaded")
        let w = Worker(asr: asr, convert: convert)
        cached = w
        return w
    }

    private init(asr: @escaping ([Float]) throws -> String, convert: ChineseConverter) {
        self.asr = asr
        self.convert = convert
    }

    func begin() {
        audioQueue.async { [self] in
            running = true
            buffer = []
            lastPartial = 0
        }
    }

    /// Stop listening: what has been said since the last finished line is still transcribed and becomes one.
    func end() {
        audioQueue.async { [self] in
            running = false
            cut(at: buffer.count)
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
            if buffer.count >= Self.maxSamples { cut(at: quietest()) }
            schedule()
        }
    }

    /// Where in the last 2 s of buffer the 20 ms with the least energy starts.
    private func quietest() -> Int {
        var best = buffer.count, bestEnergy = Float.infinity
        var i = buffer.count - Self.cutWindow
        while i + Self.quietFrame <= buffer.count {
            var e: Float = 0
            for j in i..<(i + Self.quietFrame) { e += buffer[j] * buffer[j] }
            if e < bestEnergy { bestEnergy = e; best = i + Self.quietFrame / 2 }
            i += Self.quietFrame / 2
        }
        return best
    }

    /// The audio before `at` becomes the current line's last transcription; the rest starts the next line.
    private func cut(at: Int) {
        if at > 0 {
            Self.log.notice("line \(self.currentId) ends: \(at / 16) ms")
            ended.append(Segment(id: currentId, audio: Array(buffer[..<at])))
            buffer.removeFirst(at)
            currentId += 1
        }
        lastPartial = 0
        schedule()
    }

    /// Start the next transcription if the model is free: the oldest finished line first, else a pending refresh.
    private func schedule() {
        guard !busy else { return }
        let job: Segment, final: Bool
        if !ended.isEmpty {
            job = ended.removeFirst(); final = true
        } else if running, buffer.count - lastPartial >= Self.partialEvery {
            job = Segment(id: currentId, audio: buffer); final = false; lastPartial = buffer.count
        } else { return }
        busy = true
        modelQueue.async { [self] in
            let text = job.audio.count < 4000 ? "" : transcribe(job.audio)
            audioQueue.async { [self] in
                busy = false
                // One job runs at a time and ids only grow, so a line's final lands after its pending refreshes and lines
                // stay in order. A cleared line doesn't come back; a finished line with no words leaves.
                if let i = lines.firstIndex(where: { $0.id == job.id }) {
                    if final && text.isEmpty { lines.remove(at: i) } else { lines[i] = Line(id: job.id, text: text, final: final) }
                    onUpdate?(lines)
                } else if job.id >= keepFrom, !text.isEmpty {
                    lines.append(Line(id: job.id, text: text, final: final))
                    onUpdate?(lines)
                }
                schedule()
            }
        }
    }

    private func transcribe(_ audio: [Float]) -> String {
        let t0 = Date()
        do {
            let raw = try asr(audio)
            Self.log.notice("transcribed \(audio.count / 16) ms of audio in \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(raw, privacy: .private)")
            return convert.convert(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        } catch {
            Self.log.error("transcription failed after \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(error.localizedDescription, privacy: .public)")
            return ""
        }
    }
}
