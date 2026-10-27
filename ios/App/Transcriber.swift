import AVFoundation
import CoreML
import Foundation
import OpenCC
import os
import Qwen3ASR
import SpeechVAD

/// Transcribes on the phone: Qwen3-ASR 1.7B converted to CoreML (DE0CH/Qwen3-ASR-1.7B-CoreML), run on the Neural
/// Engine and CPU only, never the GPU, so it keeps going with the screen locked. Nothing leaves the phone.
///
/// The mic's 16 kHz audio goes through Silero VAD (CoreML). Speech is cut into segments at pauses (or at 10 s); while a
/// segment is still being spoken it is transcribed again every so often as pending text, and when it ends it is
/// transcribed once more and committed as its own line. The model works on one segment at a time on its own queue;
/// a finished segment always goes before a pending refresh.
@MainActor
final class Transcriber: ObservableObject {
    enum State: Equatable { case idle, loading(String), listening, failed(String) }

    /// One segment's text: commit is final, pending may still change.
    struct Part: Identifiable { let id: Int; let commit: String; let pending: String }

    @Published var committed = ""
    @Published var parts: [Part] = []
    @Published var state: State = .idle

    var isRunning: Bool {
        switch state { case .loading, .listening: return true; default: return false }
    }

    static let modelRepo = "DE0CH/Qwen3-ASR-1.7B-CoreML"

    private var engine: AVAudioEngine?
    private var sessionObservers: [NSObjectProtocol] = []
    private var filePlayback: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var worker: Worker?
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    /// UI tests stream the bundled clip instead of the microphone (the simulator has no real mic).
    private let testAudio = ProcessInfo.processInfo.environment["CANTONESE_TEST_AUDIO"] == "1"

    func start() {
        guard !isRunning else { return }
        state = .loading("載入模型…")
        loadTask = Task {
            do {
                let worker = try await Worker.shared(progress: { [weak self] p in
                    Task { @MainActor in
                        guard let self, case .loading = self.state else { return }
                        self.state = .loading(p >= 1 ? "載入模型…" : "下載模型 \(Int(p * 100))%（第一次要下載，約 2.5 GB）")
                    }
                })
                try Task.checkCancellation()
                worker.onUpdate = { [weak self] parts, finished in
                    Task { @MainActor in self?.show(parts, finished: finished) }
                }
                worker.begin(partials: !testAudio)
                self.worker = worker
            } catch is CancellationError {
                return
            } catch {
                fail("用唔到模型：\(error.localizedDescription)")
                return
            }
            do {
                if testAudio { streamTestFile() } else { try startMicrophone() }
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
        committed = ""
        parts = []
        worker?.clear()
    }

    /// Finished segments become plain text; the rest show as parts (commit + pending).
    private func show(_ segments: [Worker.Shown], finished: String) {
        committed += finished
        parts = segments.map { Part(id: $0.id, commit: $0.commit, pending: $0.pending) }
    }

    private func fail(_ message: String) {
        stopAudio()
        worker?.end()
        state = .failed(message)
    }

    private func stopAudio() {
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

/// The models and everything that touches them, on one serial queue.
final class Worker: @unchecked Sendable {
    static let log = Logger(subsystem: "dev.de0ch.cantonese", category: "model")
    struct Shown { let id: Int; let commit: String; let pending: String }

    /// Segments shown on screen and the text that just finished.
    var onUpdate: (([Shown], String) -> Void)?

    private let asr: CoreMLASRModel
    private let vad: SileroVADModel
    private let convert: ChineseConverter
    private let audioQueue = DispatchQueue(label: "cantonese.audio")
    private let modelQueue = DispatchQueue(label: "cantonese.model", qos: .userInitiated)

    // audioQueue state
    private final class Segment {
        let id: Int
        var samples: [Float]
        var ended = false
        var transcribedCount = 0     // samples in the last pending transcription
        var pending = ""
        var final: String?
        var discard = false          // cleared while still being transcribed
        init(id: Int, samples: [Float]) { self.id = id; self.samples = samples }
    }
    private var segments: [Segment] = []
    private var current: Segment?
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

    /// Loads the models once (downloading them the first time) and keeps them for later sessions.
    static func shared(progress: @escaping (Double) -> Void) async throws -> Worker {
        if let w = cached { return w }
        let dir = try modelDirectory()
        let asrDir = dir.appendingPathComponent("qwen3-asr-1.7b")
        try await ModelFiles.ensure(repo: Transcriber.modelRepo, in: asrDir, progress: progress)
        progress(1)
        let asr = try await CoreMLASRModel.fromPretrained(
            encoderModelId: Transcriber.modelRepo, decoderModelId: Transcriber.modelRepo,
            tokenizerModelId: Transcriber.modelRepo,
            encoderComputeUnits: .cpuAndNeuralEngine, decoderComputeUnits: .cpuAndNeuralEngine,
            cacheDir: asrDir, offlineMode: true)
        let vad = try await SileroVADModel.fromPretrained(engine: .coreml, cacheDir: dir.appendingPathComponent("silero-vad"))
        let convert = try ChineseConverter(options: [.traditionalize, .hkStandard])
        log.info("models loaded")
        let w = Worker(asr: asr, vad: vad, convert: convert)
        cached = w
        return w
    }

    /// Application Support/models, kept out of iCloud backups (the models download again).
    private static func modelDirectory() throws -> URL {
        var dir = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                              appropriateFor: nil, create: true).appendingPathComponent("models")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try dir.setResourceValues(values)
        return dir
    }

    private init(asr: CoreMLASRModel, vad: SileroVADModel, convert: ChineseConverter) {
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

    /// Clear the screen: text of segments already spoken is dropped, the one being spoken carries on.
    func clear() {
        audioQueue.async { [self] in
            for s in segments where s !== current { s.discard = true }
            current?.pending = ""
            settle()
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
        if let s = current {
            s.samples += c
            silentChunks = prob < Self.stopProb ? silentChunks + 1 : 0
            if silentChunks >= Self.endSilenceChunks || s.samples.count >= Self.maxSamples { endCurrent() }
        } else {
            preRoll += c
            if preRoll.count > Self.preRollSamples { preRoll.removeFirst(preRoll.count - Self.preRollSamples) }
            if prob >= Self.startProb {
                Self.log.info("segment \(self.nextId) starts (vad \(prob))")
                let s = Segment(id: nextId, samples: preRoll)
                nextId += 1
                segments.append(s)
                current = s
                preRoll = []
                silentChunks = 0
            }
        }
    }

    private func endCurrent() {
        if let s = current { Self.log.info("segment \(s.id) ends: \(s.samples.count / 16) ms") }
        current?.ended = true
        current = nil
        silentChunks = 0
        schedule()
    }

    /// Start the next transcription if the model is free: the oldest ended segment first, else a pending refresh.
    private func schedule() {
        guard !busy else { return }
        let job: (Segment, [Float], Bool)
        if let s = segments.first(where: { $0.ended && $0.final == nil }) {
            job = (s, s.samples, true)
        } else if partials, let s = current, s.samples.count - s.transcribedCount >= Self.partialEvery {
            job = (s, s.samples, false)
        } else { return }
        busy = true
        modelQueue.async { [self] in
            let text = job.1.count < 4000 ? "" : transcribe(job.1)
            audioQueue.async { [self] in
                busy = false
                let (s, audio, isFinal) = job
                if isFinal { s.final = text } else { s.pending = text; s.transcribedCount = audio.count }
                settle()
                schedule()
            }
        }
    }

    private func transcribe(_ audio: [Float]) -> String {
        let t0 = Date()
        do {
            let raw = try asr.transcribeWithoutMLX(audio: audio, sampleRate: 16000, language: nil, maxTokens: 200)
            Self.log.info("transcribed \(audio.count / 16) ms of audio in \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(raw, privacy: .private)")
            return convert.convert(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        } catch {
            Self.log.error("transcription failed after \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(error.localizedDescription, privacy: .public)")
            return ""
        }
    }

    /// Finished segments at the front leave the list as committed lines.
    private func settle() {
        var finished = ""
        while let s = segments.first, let text = s.final {
            if !text.isEmpty, !s.discard { finished += text + "\n" }
            segments.removeFirst()
        }
        publish(finished: finished)
    }

    private func publish(finished: String) {
        let shown = segments.filter { !$0.discard }.map {
            Shown(id: $0.id, commit: $0.final.map { $0.isEmpty ? "" : $0 + "\n" } ?? "", pending: $0.final == nil ? $0.pending : "")
        }
        onUpdate?(shown, finished)
    }
}
