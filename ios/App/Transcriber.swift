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
/// The model runs all the time, with no speech detection in front of it, over a sliding window of the mic's 16 kHz
/// audio (Worker): the window's words are timed by Qwen3-ForcedAligner 0.6B, and those well inside it turn black.
@MainActor
final class Transcriber: ObservableObject {
    enum State: Equatable { case idle, loading(String), listening, failed(String) }

    /// What has been said since the last clear, one line per segment, oldest first.
    @Published var lines: [Worker.Line] = []
    @Published var state: State = .idle
    /// UI tests: the loop is stopped at a run for a screenshot (Worker.holdAfter, from CANTONESE_HOLD).
    @Published var held = false
    /// UI tests: the whole test clip has gone in and the loop has caught up.
    @Published var played = false

    var isRunning: Bool {
        switch state { case .loading, .listening: return true; default: return false }
    }

    private var engine: AVAudioEngine?
    private var sessionObservers: [NSObjectProtocol] = []
    private var filePlayback: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var worker: Worker?
    /// The audio session and engine are switched on and off here, not on the main thread: those calls block (switching
    /// the session off can take a few hundred ms), which froze the button at the tap.
    private nonisolated static let session = DispatchQueue(label: "cantonese.session", qos: .userInitiated)
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
        // The button changes at the tap: listening at once once the model is in memory, the audio comes on behind it.
        state = Worker.loaded ? .listening : .loading("載入模型…")
        loadTask = Task {
            do {
                let worker = try await Worker.shared()
                try Task.checkCancellation()
                let latest = Latest<[Worker.Line]> { [weak self] lines in self?.lines = lines }
                worker.onUpdate = { latest.send($0) }
                worker.onHold = { [weak self] held in
                    Task { @MainActor in self?.held = held }
                }
                worker.onFail = { [weak self] message in
                    Task { @MainActor in self?.fail(message) }
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
                if testAudio { streamTestFile() } else { try await startMicrophone() }
                UIApplication.shared.isIdleTimerDisabled = true   // the screen stays on while listening
                state = .listening
            } catch is CancellationError {
                return
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
        let engine = self.engine
        self.engine = nil
        filePlayback?.cancel()
        filePlayback = nil
        for o in sessionObservers { NotificationCenter.default.removeObserver(o) }
        sessionObservers = []
        Self.session.async { Self.tearDown(engine) }
    }

    private nonisolated static func tearDown(_ engine: AVAudioEngine?) {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func startMicrophone() async throws {
        let session = AVAudioSession.sharedInstance()
        let center = NotificationCenter.default
        sessionObservers = [
            center.addObserver(forName: AVAudioSession.routeChangeNotification, object: session, queue: .main) { _ in
                Self.session.async { Self.pinBuiltInMic() }
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
        let (worker, target) = (self.worker, self.target)
        let engine = try await withCheckedThrowingContinuation { (done: CheckedContinuation<AVAudioEngine, Error>) in
            Self.session.async {
                do { done.resume(returning: try Self.startEngine(feeding: worker, target: target)) } catch { done.resume(throwing: error) }
            }
        }
        if Task.isCancelled {   // stopped while the audio came on
            Self.session.async { Self.tearDown(engine) }
            throw CancellationError()
        }
        self.engine = engine
    }

    /// The mic's audio, converted to 16 kHz mono, into the worker. Runs on the session queue.
    private nonisolated static func startEngine(feeding worker: Worker?, target: AVAudioFormat) throws -> AVAudioEngine {
        // Always the iPhone's own mic, mixed with whatever else plays: Bluetooth headphones stay on A2DP (output only,
        // so they keep playing music in full quality and never become the mic); with no headphones other apps' audio
        // goes to the speaker, not the earpiece. A call or another app taking the audio session stops transcription.
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement,
                                options: [.mixWithOthers, .allowBluetoothA2DP, .defaultToSpeaker])
        try session.setActive(true)
        pinBuiltInMic()
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: format, to: target) else { throw CocoaError(.featureUnsupported) }
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
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
        return engine
    }

    private nonisolated static func pinBuiltInMic() {
        let session = AVAudioSession.sharedInstance()
        guard session.currentRoute.inputs.first?.portType != .builtInMic,
              let mic = session.availableInputs?.first(where: { $0.portType == .builtInMic }) else { return }
        try? session.setPreferredInput(mic)
    }

    private func streamTestFile() {
        guard let clip = testClip, let worker else { fail("搵唔到測試錄音"); return }
        worker.holdAfter = clip.holdAfter
        worker.holdFor = 8
        played = false
        Worker.log.notice("test audio starts")
        filePlayback = Task { [weak self] in
            await worker.play(clip.samples, pace: true)
            if !Task.isCancelled { self?.played = true }
        }
    }
}

/// A bundled 16 kHz mono clip the UI tests stream instead of the microphone, with every model run and alignment the app's
/// loop made over it with the real models on a Mac (testdata/…passes.json, from tools/demo-passes in the model-check
/// workflow): the simulator's stand-in answers with them.
struct TestClip: Sendable {
    struct Run: Codable, Sendable { let audio: Int; let sum: Float; let context: String; let start: String; let text: String }
    struct Alignment: Codable, Sendable { let audio: Int; let sum: Float; let text: String; let words: [Worker.Word] }
    struct Passes: Codable, Sendable { var runs: [Run] = []; var alignments: [Alignment] = [] }
    let samples: [Float]
    let passes: Passes
    /// CANTONESE_HOLD: the runs (comma-separated, counted from 1) to stop at for a screenshot.
    var holdAfter: Set<Int> {
        Set((ProcessInfo.processInfo.environment["CANTONESE_HOLD"] ?? "").split(separator: ",").compactMap { Int($0) })
    }

    /// CANTONESE_TEST_AUDIO: "1" streams the short test clip, "demo" the App Store demo conversation.
    static let current: TestClip? = ProcessInfo.processInfo.environment["CANTONESE_TEST_AUDIO"]
        .flatMap { load($0 == "demo" ? "demo-16k" : "cantonese-16k") }

    private static func load(_ name: String) -> TestClip? {
        let file = { (n: String, e: String) in Bundle.main.url(forResource: n, withExtension: e).flatMap { try? Data(contentsOf: $0) } }
        guard let samples = file(name, "wav").flatMap(Worker.samples(wav:)),
              let passes = file(name + ".passes", "json").flatMap({ try? JSONDecoder().decode(Passes.self, from: $0) })
        else { return nil }
        return TestClip(samples: samples, passes: passes)
    }
}

extension Worker {
    private static var cached: Worker?
    /// The model is in memory: starting again is immediate.
    static var loaded: Bool { cached != nil }

    /// Loads the models shipped inside the app (ios/fetch-model.sh puts them in Model/) once and keeps them.
    static func shared() async throws -> Worker {
        if let w = cached { return w }
        guard let dir = Bundle.main.url(forResource: "Model", withExtension: nil) else {
            throw CocoaError(.fileNoSuchFile)
        }
        #if targetEnvironment(simulator)
        // The simulator has no Metal for MLX, so the UI tests run everything but the models, with this stand-in for
        // them: the real models' answers to this very run (same audio, context and start) and alignment.
        _ = dir
        let asr: ASR = { audio, context, start, _, onText in
            let sum = audio.reduce(0, +)
            guard let r = TestClip.current?.passes.runs.first(where: {
                $0.audio == audio.count && $0.sum == sum && $0.context == context && $0.start == start
            }) else {
                log.error("run not recorded: \(audio.count) samples from \(start.count) chars")
                return start
            }
            onText(r.text)
            return r.text
        }
        let align: Aligner = { audio, text in
            let sum = audio.reduce(0, +)
            guard let a = TestClip.current?.passes.alignments.first(where: {
                $0.audio == audio.count && $0.sum == sum && $0.text == text
            }) else {
                log.error("alignment not recorded: \(audio.count) samples, \(text.count) chars")
                return []
            }
            return a.words
        }
        #else
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: modelId, cacheDir: dir.appendingPathComponent("asr"), offlineMode: true)
        let aligner = try await Qwen3ForcedAligner.fromPretrained(
            modelId: alignerId, cacheDir: dir.appendingPathComponent("aligner"), offlineMode: true)
        let asr = Worker.asr(model)
        let align = Worker.aligner(aligner)
        #endif
        let converter = try ChineseConverter(options: [.traditionalize, .hkStandard])
        log.notice("model loaded")
        let w = Worker(asr: asr, align: align, convert: { converter.convert($0) })
        cached = w
        return w
    }
}

/// Hands the newest value to the main thread at most every 0.1 s. The model writes a token at a time, and laying out the
/// whole transcript that often kept the main thread too busy to show a tap.
final class Latest<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value?
    private let deliver: @MainActor (Value) -> Void

    init(_ deliver: @escaping @MainActor (Value) -> Void) { self.deliver = deliver }

    func send(_ new: Value) {
        lock.lock()
        let first = value == nil
        value = new
        lock.unlock()
        guard first else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [self] in
            lock.lock()
            let v = value
            value = nil
            lock.unlock()
            if let v { MainActor.assumeIsolated { deliver(v) } }
        }
    }
}
