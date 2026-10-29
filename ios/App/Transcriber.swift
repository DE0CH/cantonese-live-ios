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
    /// UI tests: the loop is stopped at a pass for a screenshot (Worker.holdAfter, from CANTONESE_HOLD).
    @Published var held = false

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
        Worker.log.notice("test audio starts")
        filePlayback = Task { await worker.play(clip.samples, pace: true) }
    }
}

/// A bundled 16 kHz mono clip the UI tests stream instead of the microphone, and for the simulator's stand-in model
/// what the real model said: for the test clip, per line (testdata/cantonese-16k.json, from the model-check workflow);
/// for the App Store demo, every pass the app's loop made over it with the real model on a Mac
/// (testdata/demo/demo-16k.passes.json, from tools/demo-passes in the model-check workflow).
struct TestClip {
    struct Line: Decodable { let start: Double; let end: Double; let text: String }
    struct Pass: Decodable { let audio: Int; let sum: Float; let start: String; let text: String }
    let samples: [Float]
    var lines: [Line] = []
    var passes: [Pass]? = nil
    /// CANTONESE_HOLD: the passes (comma-separated, counted from 1) to stop at for a screenshot.
    var holdAfter: Set<Int> {
        Set((ProcessInfo.processInfo.environment["CANTONESE_HOLD"] ?? "").split(separator: ",").compactMap { Int($0) })
    }

    /// CANTONESE_TEST_AUDIO: "1" streams the short test clip, "demo" the App Store demo conversation.
    static let current: TestClip? = ProcessInfo.processInfo.environment["CANTONESE_TEST_AUDIO"]
        .flatMap { $0 == "demo" ? load("demo-16k", passes: true) : load("cantonese-16k", passes: false) }

    private static func load(_ name: String, passes: Bool) -> TestClip? {
        let json = { (n: String) in Bundle.main.url(forResource: n, withExtension: "json").flatMap { try? Data(contentsOf: $0) } }
        guard let wav = Bundle.main.url(forResource: name, withExtension: "wav").flatMap({ try? Data(contentsOf: $0) }),
              let samples = Worker.samples(wav: wav) else { return nil }
        var clip = TestClip(samples: samples)
        if passes {
            guard let p = json(name + ".passes").flatMap({ try? JSONDecoder().decode([Pass].self, from: $0) }) else { return nil }
            clip.passes = p
        } else {
            guard let l = json(name).flatMap({ try? JSONDecoder().decode([Line].self, from: $0) }) else { return nil }
            clip.lines = l
        }
        return clip
    }
}

extension Worker {
    private static var cached: Worker?
    /// The model is in memory: starting again is immediate.
    static var loaded: Bool { cached != nil }

    /// Loads the model shipped inside the app (ios/fetch-model.sh puts it in Model/) once and keeps it.
    static func shared() async throws -> Worker {
        if let w = cached { return w }
        guard let dir = Bundle.main.url(forResource: "Model", withExtension: nil) else {
            throw CocoaError(.fileNoSuchFile)
        }
        #if targetEnvironment(simulator)
        // The simulator has no Metal for MLX, so the UI tests run everything but the model, with this stand-in for it.
        _ = dir
        let asr: ASR = { audio, start, onText in
            guard let clip = TestClip.current else { return start }
            if let passes = clip.passes {
                // The demo: the real model's answer to this very pass (same audio, same start).
                let sum = audio.reduce(0, +)
                guard let p = passes.first(where: { $0.audio == audio.count && $0.sum == sum && $0.start == start }) else {
                    log.error("pass not recorded: \(audio.count) samples from \(start.count) chars")
                    return start
                }
                onText(p.text)
                return p.text
            }
            // The test clip: find where the audio sits in it (10 ms from the first sound; the silence after the clip
            // matches nothing: no words) and answer with the lines whose middle falls inside it.
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
            modelId: modelId, cacheDir: dir.appendingPathComponent("asr"), offlineMode: true)
        let asr = Worker.asr(model)
        #endif
        let converter = try ChineseConverter(options: [.traditionalize, .hkStandard])
        log.notice("model loaded")
        let w = Worker(asr: asr, convert: { converter.convert($0) })
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
