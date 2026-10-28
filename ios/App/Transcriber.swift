import AVFoundation
import Foundation
import OpenCC
import os
import Qwen3ASR
import SpeechVAD
import UIKit

/// Transcribes on the phone: Qwen3-ASR 1.7B (5-bit, aufklarer/Qwen3-ASR-1.7B-MLX-5bit) on the GPU through MLX, via
/// speech-swift. Nothing leaves the phone. iOS allows no GPU work in the background, so it works with the app on
/// screen: the screen stays awake while listening, and leaving the app stops it.
///
/// The mic's 16 kHz audio goes through Silero VAD (CoreML). Speech is cut into segments at pauses (or at 10 s); while a
/// segment is still being spoken it is transcribed again every so often as pending text, and when it ends it is
/// transcribed once more and becomes a finished line. The model works on one segment at a time on its own queue; a
/// finished segment always goes before a pending refresh.
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
                worker.begin(partials: !testAudio)
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

/// The models and everything that touches them. Audio state lives on audioQueue, the model runs on modelQueue.
final class Worker: @unchecked Sendable {
    static let log = Logger(subsystem: "dev.de0ch.cantonese", category: "model")

    /// One segment's text: final once the whole segment has been transcribed, pending (and may change) until then.
    struct Line: Identifiable { let id: Int; var text: String; var final: Bool }
    private struct Segment { let id: Int; var audio: [Float] }

    /// Every line on screen, after each change.
    var onUpdate: (([Line]) -> Void)?

    private let asr: ([Float]) throws -> String
    private let vad: SileroVADModel
    private let convert: ChineseConverter
    private let audioQueue = DispatchQueue(label: "cantonese.audio")
    private let modelQueue = DispatchQueue(label: "cantonese.model", qos: .userInitiated)

    // audioQueue state
    private var lines: [Line] = []
    private var current: Segment?        // being spoken
    private var ended: [Segment] = []     // spoken, waiting for the model
    private var lastPartial = 0           // samples of current at its last pending transcription
    private var vadBuffer: [Float] = []
    private var preRoll: [Float] = []
    private var silentChunks = 0
    private var nextId = 0
    private var busy = false
    private var running = false
    private var partials = true

    private static let chunk = SileroVADModel.chunkSize            // 512 samples = 32 ms
    private static let startProb: Float = 0.5
    private static let stopProb: Float = 0.35
    private static let endSilenceChunks = 19                         // ~0.6 s pause ends a segment
    private static let maxSamples = 10 * 16000                       // force a cut at 10 s
    private static let preRollSamples = 5 * 512                      // keep 160 ms before speech starts
    private static let partialEvery = 24_000                         // refresh pending after 1.5 s more audio

    private static var cached: Worker?

    /// Loads the models shipped inside the app (ios/fetch-model.sh puts them in Model/) once and keeps them.
    static func shared() async throws -> Worker {
        if let w = cached { return w }
        guard let dir = Bundle.main.url(forResource: "Model", withExtension: nil) else {
            throw CocoaError(.fileNoSuchFile)
        }
        #if targetEnvironment(simulator)
        // The simulator has no Metal for MLX, so the UI tests run everything but the model: this stand-in answers each
        // segment of the bundled clip with what the model says for it (in its Simplified output, so OpenCC still runs).
        // The real model is heard on a Mac by the model-check workflow.
        let lines = ["我哋今日去咗饮茶，啲虾饺好好食。", "佢话听日会再嚟。"]
        var said = 0
        let asr: ([Float]) throws -> String = { _ in defer { said += 1 }; return lines[min(said, lines.count - 1)] }
        #else
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-5bit", cacheDir: dir.appendingPathComponent("asr"), offlineMode: true)
        let asr: ([Float]) throws -> String = { audio in
            var options = Qwen3DecodingOptions()
            options.maxTokens = 200
            return try model.transcribeThrowing(audio: audio, sampleRate: 16000, options: options)
        }
        #endif
        let vad = try await SileroVADModel.fromPretrained(engine: .coreml, cacheDir: dir.appendingPathComponent("vad"),
                                                          offlineMode: true)
        let convert = try ChineseConverter(options: [.traditionalize, .hkStandard])
        log.notice("models loaded")
        let w = Worker(asr: asr, vad: vad, convert: convert)
        cached = w
        return w
    }

    private init(asr: @escaping ([Float]) throws -> String, vad: SileroVADModel, convert: ChineseConverter) {
        self.asr = asr
        self.vad = vad
        self.convert = convert
    }

    func begin(partials: Bool) {
        audioQueue.async { [self] in
            running = true
            self.partials = partials
            vad.resetState()
            vadBuffer = []; preRoll = []; silentChunks = 0; current = nil
        }
    }

    /// Stop listening: the segment being spoken ends now and is still transcribed.
    func end() {
        audioQueue.async { [self] in
            running = false
            endCurrent()
        }
    }

    /// Clear the screen: everything already spoken goes, the segment being spoken carries on.
    func clear() {
        audioQueue.async { [self] in
            ended = []
            lines.removeAll { $0.id != current?.id }
            onUpdate?(lines)
        }
    }

    func feed(_ samples: [Float]) {
        audioQueue.async { [self] in
            guard running else { return }
            vadBuffer += samples
            while vadBuffer.count >= Self.chunk {
                let c = Array(vadBuffer.prefix(Self.chunk))
                vadBuffer.removeFirst(Self.chunk)
                step(c, prob: vad.processChunk(c))
            }
            schedule()
        }
    }

    private func step(_ c: [Float], prob: Float) {
        if current != nil {
            current!.audio += c
            silentChunks = prob < Self.stopProb ? silentChunks + 1 : 0
            if silentChunks >= Self.endSilenceChunks || current!.audio.count >= Self.maxSamples { endCurrent() }
        } else {
            preRoll += c
            if preRoll.count > Self.preRollSamples { preRoll.removeFirst(preRoll.count - Self.preRollSamples) }
            if prob >= Self.startProb {
                Self.log.notice("segment \(self.nextId) starts (vad \(prob))")
                current = Segment(id: nextId, audio: preRoll)
                lines.append(Line(id: nextId, text: "", final: false))
                nextId += 1
                preRoll = []
                silentChunks = 0
                lastPartial = 0
            }
        }
    }

    private func endCurrent() {
        if let s = current {
            Self.log.notice("segment \(s.id) ends: \(s.audio.count / 16) ms")
            ended.append(s)
        }
        current = nil
        silentChunks = 0
        schedule()
    }

    /// Start the next transcription if the model is free: the oldest ended segment first, else a pending refresh.
    private func schedule() {
        guard !busy else { return }
        let job: Segment, final: Bool
        if !ended.isEmpty {
            job = ended.removeFirst(); final = true
        } else if partials, let s = current, s.audio.count - lastPartial >= Self.partialEvery {
            job = s; final = false; lastPartial = s.audio.count
        } else { return }
        busy = true
        modelQueue.async { [self] in
            let text = job.audio.count < 4000 ? "" : transcribe(job.audio)
            audioQueue.async { [self] in
                busy = false
                // The line is gone if the screen was cleared meanwhile; a finished segment with no words leaves. One job
                // runs at a time, so a segment's final always lands after its pending refreshes.
                if let i = lines.firstIndex(where: { $0.id == job.id }) {
                    if final && text.isEmpty { lines.remove(at: i) } else { lines[i] = Line(id: job.id, text: text, final: final) }
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
