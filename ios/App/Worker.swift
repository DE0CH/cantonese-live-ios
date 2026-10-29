import Foundation
import os
import Qwen3ASR

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
    /// After each model job, with how many have finished (the jobs of a clip played with `play` are always the same).
    var onPass: ((Int) -> Void)?
    /// For the App Store screenshots: after these jobs (counted from 1) the loop stops for `holdFor` seconds, with
    /// `onHold(true)` while it waits, so the screen can be captured exactly as that pass left it.
    var holdAfter: Set<Int> = []
    var holdFor: TimeInterval = 5
    var onHold: ((Bool) -> Void)?

    private let asr: ASR
    private let convert: (String) -> String
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
    private var passes = 0                // model jobs finished
    private var holding = false
    private var waiting: [CheckedContinuation<Void, Never>] = []   // `idle()` callers

    private static let passEvery = 8_000             // a pass once 0.5 s more audio is in and the model is free
    private static let closeAfter = 2                // two more passes with no new words: a pause, the line closes
    private static let maxSamples = 30 * 16000       // 30 s without a pause: the line is cut...
    private static let cutWindow = 3 * 16000         // ...at the quietest 100 ms of its last 3 s
    private static let quietFrame = 1600
    private static let cutRollback = 10              // characters the cut line's last pass may rewrite

    static let modelId = "aufklarer/Qwen3-ASR-1.7B-MLX-5bit"

    init(asr: @escaping ASR, convert: @escaping (String) -> String) {
        self.asr = asr
        self.convert = convert
    }

    /// The model as the loop calls it: up to 200 tokens, continuing from `start`, the text so far after every token.
    static func asr(_ model: Qwen3ASRModel) -> ASR {
        { audio, start, onText in
            var options = Qwen3DecodingOptions()
            options.maxTokens = 200
            options.prefix = start
            options.onText = onText
            return try model.transcribeThrowing(audio: audio, sampleRate: 16000, options: options)
        }
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

    /// Returns once the model has nothing to do (no pass due, none running, not held).
    func idle() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            audioQueue.async { [self] in
                waiting.append(c)
                schedule()
            }
        }
    }

    /// Streams a recorded 16 kHz clip as the mic would, in 100 ms pieces, then a second of silence (so the last words
    /// end their line like a pause). Each piece goes in once the model has caught up with the last, so a clip always
    /// gets the same passes on any machine; `pace` also waits 100 ms a piece, like the mic.
    func play(_ samples: [Float], pace: Bool) async {
        let all = samples + [Float](repeating: 0, count: 16000)
        var offset = 0
        while offset < all.count, !Task.isCancelled {
            let end = min(offset + 1600, all.count)
            feed(Array(all[offset..<end]))
            offset = end
            await idle()
            if pace { try? await Task.sleep(nanoseconds: 100_000_000) }
        }
    }

    /// 16-bit PCM from a WAV file's "data" chunk (WAV headers are not always 44 bytes).
    static func samples(wav: Data) -> [Float]? {
        var i = 12
        while i + 8 <= wav.count {
            let size = (0..<4).reduce(0) { $0 | Int(wav[i + 4 + $1]) << (8 * $1) }
            if wav[i..<(i + 4)] == Data("data".utf8) {
                let pcm = wav[(i + 8)..<min(wav.count, i + 8 + size)]
                return pcm.withUnsafeBytes { raw in
                    (0..<(raw.count / 2)).map { Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 2, as: Int16.self))) / 32768 }
                }
            }
            i += 8 + size + (size & 1)
        }
        return nil
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
        guard !busy, !holding else { return }
        let job: Job
        if !ended.isEmpty {
            job = ended.removeFirst()
        } else if running, buffer.count >= 4000, buffer.count - lastPass >= Self.passEvery {
            lastPass = buffer.count
            job = Job(id: currentId, audio: buffer, start: black, final: false)
        } else {
            for c in waiting { c.resume() }
            waiting = []
            return
        }
        busy = true
        modelQueue.async { [self] in
            // The tail fills in token by token while the model writes it.
            let text = transcribe(job) { [weak self] sofar in
                self?.audioQueue.async { self?.showPartial(job, sofar) }
            }
            audioQueue.async { [self] in
                busy = false
                if job.final { finish(job, text) } else { passed(job, text) }
                passes += 1
                onPass?(passes)
                if holdAfter.contains(passes) {
                    Self.log.notice("held after pass \(self.passes)")
                    holding = true
                    onHold?(true)
                    audioQueue.asyncAfter(deadline: .now() + holdFor) { [self] in
                        holding = false
                        onHold?(false)
                        schedule()
                    }
                }
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
        let line = Line(id: id, settled: convert(settled), tail: convert(tail), final: final)
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
