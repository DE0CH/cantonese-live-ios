import AVFoundation
import Foundation
import OpenCC
import os
import UIKit

/// Transcribes on the phone (Worker): Silero cuts the mic's 16 kHz audio at pauses, SenseVoice writes the grey text
/// while someone speaks and each finished stretch in black. Nothing leaves the phone. It works with the app on screen:
/// the screen stays awake while listening, and leaving the app stops it.
@MainActor
final class Transcriber: ObservableObject {
    enum State: Equatable { case idle, loading(String), listening, failed(String) }

    /// What has been said since the last clear, one line per segment, oldest first.
    @Published var lines: [Worker.Line] = []
    @Published var state: State = .idle
    /// UI tests: the clip is stopped for a screenshot (CANTONESE_HOLD).
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
        // The app records only on screen: leaving it (or locking the phone) stops listening.
        NotificationCenter.default.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil,
                                               queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.stop() }
        }
    }

    func start() {
        guard !isRunning else { return }
        // The button changes at the tap: listening at once once the model is in memory, the audio comes on behind it.
        state = Worker.loaded ? .listening : .loading(String(localized: "載入模型…"))
        loadTask = Task {
            do {
                let worker = try await Worker.shared()
                try Task.checkCancellation()
                let latest = Latest<[Worker.Line]> { [weak self] lines in self?.lines = lines }
                worker.configure(onUpdate: { latest.send($0) })
                worker.begin()
                self.worker = worker
            } catch is CancellationError {
                return
            } catch {
                fail(String(localized: "用唔到模型：\(error.localizedDescription)"))
                return
            }
            if !testAudio, !(await AVAudioApplication.requestRecordPermission()) {
                fail(String(localized: "未有權限用咪：去「設定」>「廣東話字幕」開返"))
                return
            }
            do {
                if testAudio { streamTestFile() } else { try await startMicrophone() }
                UIApplication.shared.isIdleTimerDisabled = true   // the screen stays on while listening
                state = .listening
            } catch is CancellationError {
                return
            } catch {
                fail(String(localized: "用唔到咪：\(error.localizedDescription)"))
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
        // A change of input route or sample rate stops the engine (and the tap's converter is for the old format): it
        // comes on again with the new one, else listening stops with an error.
        sessionObservers.append(center.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine,
                                                   queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.engine === engine else { return }
                Self.session.async {
                    do { try Self.restart(engine, feeding: worker, target: target) } catch {
                        Task { @MainActor [weak self] in
                            guard let self, self.engine === engine else { return }
                            self.fail(String(localized: "用唔到咪：\(error.localizedDescription)"))
                        }
                    }
                }
            }
        })
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
        try installTap(engine, feeding: worker, target: target)
        engine.prepare()
        try engine.start()
        return engine
    }

    /// The engine again after a configuration change: the tap for the input's new format. Runs on the session queue.
    private nonisolated static func restart(_ engine: AVAudioEngine, feeding worker: Worker?, target: AVAudioFormat) throws {
        engine.inputNode.removeTap(onBus: 0)
        let format = engine.inputNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { return }   // no input yet: the next change brings it
        try installTap(engine, feeding: worker, target: target)
        engine.prepare()
        try engine.start()
    }

    /// The tap that converts the input to 16 kHz mono for the worker.
    private nonisolated static func installTap(_ engine: AVAudioEngine, feeding worker: Worker?, target: AVAudioFormat) throws {
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
    }

    private nonisolated static func pinBuiltInMic() {
        let session = AVAudioSession.sharedInstance()
        guard session.currentRoute.inputs.first?.portType != .builtInMic,
              let mic = session.availableInputs?.first(where: { $0.portType == .builtInMic }) else { return }
        try? session.setPreferredInput(mic)
    }

    private func streamTestFile() {
        guard let clip = testClip, let worker else { fail("搵唔到測試錄音"); return }
        played = false
        Worker.log.notice("test audio starts")
        filePlayback = Task { [weak self] in
            await worker.play(clip.samples, pace: true, holds: clip.holds, holdFor: 8) { held in
                Task { @MainActor in self?.held = held }
            }
            if !Task.isCancelled { self?.played = true }
        }
    }
}

/// A bundled 16 kHz mono clip the UI tests stream instead of the microphone (the simulator has no real mic); the models
/// run for real on it.
struct TestClip: Sendable {
    let samples: [Float]
    /// CANTONESE_HOLD: the seconds of the clip (comma-separated) to stop at for a screenshot.
    var holds: [Double] {
        (ProcessInfo.processInfo.environment["CANTONESE_HOLD"] ?? "").split(separator: ",").compactMap { Double($0) }
    }

    /// CANTONESE_TEST_AUDIO: "1" streams the short test clip, "demo" the App Store demo conversation.
    static let current: TestClip? = ProcessInfo.processInfo.environment["CANTONESE_TEST_AUDIO"]
        .flatMap { Bundle.main.url(forResource: $0 == "demo" ? "demo-16k" : "cantonese-16k", withExtension: "wav") }
        .flatMap { try? Data(contentsOf: $0) }
        .flatMap(Worker.samples(wav:))
        .map { TestClip(samples: $0) }
}

extension Worker {
    @MainActor private static var cached: Worker?
    @MainActor private static var loading: Task<Worker, Error>?
    /// The models are in memory: starting again is immediate.
    @MainActor static var loaded: Bool { cached != nil }

    /// Loads the models shipped inside the app (ios/fetch-model.sh puts them in Model/) once and keeps them. One load
    /// at a time: a start after a cancelled one waits for the load under way (which a cancel doesn't stop).
    @MainActor static func shared() async throws -> Worker {
        if let w = cached { return w }
        let task = loading ?? Task.detached { try await Worker.load() }
        loading = task
        do {
            let w = try await task.value
            cached = w
            loading = nil
            return w
        } catch {
            loading = nil
            throw error
        }
    }

    private static func load() async throws -> Worker {
        guard let dir = Bundle.main.url(forResource: "Model", withExtension: nil) else {
            throw CocoaError(.fileNoSuchFile)
        }
        let models = Models.load(dir)
        let converter = try ChineseConverter(options: [.traditionalize, .hkStandard])   // used on one queue only
        log.notice("models loaded")
        return Worker(models: models, convert: converter.convert)
    }
}

/// Hands the newest value to the main thread at most every 0.1 s, so laying out the transcript never keeps the main
/// thread too busy to show a tap.
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
