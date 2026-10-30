import Foundation
import os
import Qwen3ASR

/// An agreement run (Worker.Model.agreement): the model transcribing one window afresh, a small step at a time so that
/// grey runs can go in between its steps. `text` is "" until it has finished.
protocol AgreementRun: AnyObject {
    /// One step of the run's work, done before it returns; true once the run has finished (and after).
    func step() throws -> Bool
    var text: String { get }
    var finished: Bool { get }
    /// Grey or last runs that reused this run's first 8 s.
    var reuses: Int { get }
}

extension Qwen3AgreementRun: AgreementRun {}

/// The model and everything that touches it. Audio and text state live on audioQueue, the model runs on modelQueue,
/// one piece of work at a time.
///
/// Black text: the mic's audio is cut into 8 s windows, a new one starting every 8/3 s (window k from 8/3·k s to
/// 8/3·k + 8 s, k from -2; what would lie before listening began is left out, so the first two hold 2.67 s and 5.33 s).
/// Once the mic has heard a window to its end, the model transcribes it afresh (an agreement run) with the last 100
/// characters of black text as context; one window at a time, in order. Its text is merged with the window before it
/// (8/3 s earlier, overlapping it by 16/3 s) on their longest common stretch, as Hugging Face's ASR pipeline merges
/// overlapping chunks: the earlier window's text up to the end of that stretch turns black; what follows, which only the
/// later window heard (or the earlier one guessed at its cut-off end), waits for the next merge. If the two share no
/// stretch of at least 2 characters, nothing turns black that time. So the first 2.67 s turn black at about 5.33 s,
/// then more every 2.67 s.
///
/// Grey text: whenever at least 100 ms of audio has come in since the last grey run started, the model transcribes from
/// the start of the latest window whose agreement run has finished (0 before any has) to now. It continues from the last
/// grey run's text over the same stretch (else that window's text), less its end punctuation and its last token, so it
/// rewrites the end. What it writes after the point where its text meets the black text is grey. Once windows are 8 s
/// the stretch is longer than 8 s; the model hears it in 8 s blocks, and the first block is that finished window, so
/// the model's work on it (and on the prompt before it, with that window's context) is used again as it is.
///
/// Sharing the model: an agreement run goes in small steps (the audio, each encoder layer, each decoder layer over the
/// prompt, each token written), a grey run in one go. Grey runs are held back just enough for the agreement run to end
/// before the next window is due: after a grey run that took g ms, the agreement run gets g·f/(1-f) ms of steps before
/// the next grey run, f being the share of the time until the next window is due that its remaining steps need (at most
/// 0.9). With no grey run due, its steps go one after another, and their time counts too: credit can go below zero,
/// letting the next grey runs go at once.
///
/// 10 s or more behind (an agreement run still unfinished 10 s after the end of its window): the phone can't keep up
/// and listening stops with an error. Stopping lets the agreement run under way finish, then runs the model once more,
/// afresh, from the start of the latest finished window to the end: all of it black.
/// Black text is split into lines at the end of each sentence; the last line holds the unfinished black text and the grey.
final class Worker: @unchecked Sendable {
    static let log = Logger(subsystem: "dev.de0ch.cantonese", category: "model")

    /// One line on screen: `settled` (black) never changes again, `tail` (grey) is still being revised. A final line
    /// has no tail.
    struct Line: Identifiable, Equatable { let id: Int; var settled: String; var tail: String; var final: Bool }

    /// The model as the loop calls it (the real one: `Worker.model`; the simulator's stand-in replays a recording).
    struct Model {
        /// A grey run: transcribe `audio` with `context` (older text, in the system prompt) continuing from `prefix`
        /// (written as if the model had already said it; the answer includes it). `base`, if given, is the finished
        /// agreement run over the first 8 s of `audio` with this same `context`: its work is used again. `cache` keeps
        /// the prompt before the audio between calls with the same context. `onText` gets the text so far after each
        /// token.
        let grey: (_ audio: [Float], _ context: String, _ prefix: String, _ base: AgreementRun?, _ cache: Qwen3PromptCache?,
                   _ onText: @escaping @Sendable (String) -> Void) throws -> String
        /// An agreement run over `audio` (at most 8 s) with `context`, not yet started.
        let agreement: (_ audio: [Float], _ context: String) -> AgreementRun
        /// The last run, after Stop: afresh (no prefix), `base` as for a grey run.
        let last: (_ audio: [Float], _ context: String, _ base: AgreementRun?,
                   _ onText: @escaping @Sendable (String) -> Void) throws -> String
        /// A text to continue from: without the punctuation and spaces it ends with, then without its last token.
        let trim: (String) -> String
    }

    /// Every line on screen, after each change.
    var onUpdate: (([Line]) -> Void)?
    /// After each model run (each grey run, each finished agreement run, the last run), with how many have finished
    /// (the runs over a clip played with `play` are always the same).
    var onPass: ((Int) -> Void)?
    /// Listening stopped by itself (the phone fell behind), with the message to show.
    var onFail: ((String) -> Void)?
    /// For the App Store screenshots: after these runs (counted from 1) the loop stops for `holdFor` seconds, with
    /// `onHold(true)` while it waits, so the screen can be captured exactly as that run left it.
    var holdAfter: Set<Int> = []
    var holdFor: TimeInterval = 5
    var onHold: ((Bool) -> Void)?

    static let rate = 16000
    static let window = 8 * rate                     // an agreement run hears 8 s...
    static let step = window / 3                     // ...and the next window starts 8/3 s later
    static let overlap = 2                           // characters two texts must share to be merged
    static let runEvery = rate / 10                  // a grey run at most every 100 ms of audio
    static let shortest = rate / 4                   // the model gets at least 0.25 s
    static let behind = 10 * rate                    // an agreement run unfinished 10 s after its window: too slow
    static let contextChars = 100                    // older black text given as context
    static let maxTokens = 200
    /// An agreement run's steps before it writes: the audio, the 24 encoder layers, the 28 decoder layers over the prompt.
    static let prefillSteps = 1 + 24 + 28
    static let firstStepGuess = 30.0                 // ms per agreement step until one has been timed
    static var tooSlow: String { String(localized: "部手機轉得唔夠快，跟唔上，停咗。") }

    static let modelId = "aufklarer/Qwen3-ASR-1.7B-MLX-5bit"
    /// MLX keeps freed buffers for reuse; one run's arrays need well under this (8 s of audio: the encoder's first
    /// convolution, 8 × 480 × 64 × 50 floats, is the largest at 49 MB), and the default (a quarter of the phone's
    /// memory, 2 GB) on top of the weights got the app killed.
    static let cacheLimit = 256 * 1024 * 1024

    private let asr: Model
    private let convert: (String) -> String
    private let audioQueue = DispatchQueue(label: "cantonese.audio")
    private let modelQueue = DispatchQueue(label: "cantonese.model", qos: .userInitiated)

    /// The line being spoken in one listening session: black text since its last sentence ended and the grey after it.
    private final class Open {
        var id: Int?            // in `lines` once it has text
        var black = ""
        var grey = ""
        var dropped = false     // cleared away: nothing more shows
    }
    /// The agreement run over window k (samples k·step to k·step + window since listening began, from 0).
    private final class Agreement {
        let k: Int
        let audio: [Float]
        let context: String
        var run: AgreementRun?          // made on modelQueue at the first step
        var text = ""                   // once finished
        var steps = 0                   // steps taken, their total and longest time (ms)
        var ms = 0.0
        var longest = 0.0
        init(k: Int, audio: [Float], context: String) { self.k = k; self.audio = audio; self.context = context }
    }
    private enum Kind: String { case grey, last }
    private struct Job {
        let kind: Kind; let audio: [Float]; let context: String
        let prefix: String            // the text to continue from, before `trim`
        let base: Agreement?; let cache: Qwen3PromptCache?
        let line: Open; let from: Int // where the audio starts (samples since listening began)
        var black = ""                // for the last run, which ends the session: the black text so far
        var agreed = ""               // and the latest finished window's text
    }

    // audioQueue state.
    private var lines: [Line] = []
    private var nextId = 0
    private var open = Open()
    private var audio: [Float] = []       // the mic's audio from `kept` on
    private var kept = 0                  // samples since listening began before `audio`
    private var heard = 0                 // samples since listening began
    private var lastGrey = 0              // `heard` when the last grey run started
    private var black = ""                // all black text this session
    private var context = ""              // the end of the black text, given to every run
    private let cache = Qwen3PromptCache()   // the prompt before the audio, kept while the context stays the same
    private var next = -2                 // the window whose agreement run is under way or comes next
    private var agreeing: Agreement?      // the agreement run under way
    private var agreed: Agreement?        // the latest finished one
    private var greyFrom: Int?            // where the last grey run's audio started, and its text
    private var greyText = ""
    private var credit = 0.0              // ms of grey runs not yet paid back with agreement steps (see the top)
    private var lastStepMs: Double?       // the last agreement run's mean step time and how many tokens it took
    private var lastTokens = 1
    private var incoming: [Float] = []    // mic audio of a new session while the last one's last run waits
    private var busy = false
    private var running = false
    private var finishing = false         // stopped; the last run is still to start
    private var passes = 0                // model runs finished
    private var holding = false
    private var waiting: [CheckedContinuation<Void, Never>] = []   // `idle()` callers

    init(model: Model, convert: @escaping (String) -> String) {
        self.asr = model
        self.convert = convert
    }

    /// The real model: up to 200 tokens, every window heard in 8 s blocks, the text so far after every token.
    static func model(_ m: Qwen3ASRModel) -> Model {
        func options(_ context: String, _ base: AgreementRun?) -> Qwen3DecodingOptions {
            var o = Qwen3DecodingOptions()
            o.maxTokens = Worker.maxTokens
            o.context = context
            o.blockwise = true
            // Greedy at any length: a grey run can hear over 15 s when the phone is slow.
            o.longInputThresholdSeconds = .infinity
            if let run = base as? Qwen3AgreementRun, run.finished, run.audioCount == Qwen3ASRModel.blockSamples {
                o.firstBlock = run
            }
            return o
        }
        return Model(
            grey: { audio, context, prefix, base, cache, onText in
                var o = options(context, base)
                o.prefix = prefix
                o.promptCache = cache
                o.onText = onText
                return try m.transcribeThrowing(audio: audio, sampleRate: rate, options: o)
            },
            agreement: { audio, context in
                m.agreementRun(audio: audio, context: context, maxTokens: Worker.maxTokens)
            },
            last: { audio, context, base, onText in
                var o = options(context, base)
                o.onText = onText
                return try m.transcribeThrowing(audio: audio, sampleRate: rate, options: o)
            },
            trim: { text in
                let t = Worker.withoutEndPunctuation(text)
                return t.isEmpty ? "" : m.droppingLastToken(t)
            })
    }

    /// `text` without the punctuation and spaces it ends with.
    static func withoutEndPunctuation(_ text: String) -> String {
        var t = Substring(text)
        while let c = t.last, c.isPunctuation || c.isWhitespace { t = t.dropLast() }
        return String(t)
    }

    func begin() {
        audioQueue.async { [self] in
            running = true
            if !finishing { newSession() }
        }
    }

    /// Stop listening: the agreement run under way finishes, then one last run over what is left, all of it black.
    func end() {
        audioQueue.async { [self] in
            guard running else { return }
            running = false
            // A window due at Stop (not yet started while the model was busy) is the pending one; no later ones start.
            if !finishing, agreeing == nil, heard >= Self.readyAt(next) {
                agreeing = Agreement(k: next, audio: slice(max(0, next * Self.step), Self.readyAt(next)), context: context)
            }
            finishing = true
            schedule()
        }
    }

    /// Clear the screen: everything already finished goes; while listening, the line being spoken carries on.
    func clear() {
        audioQueue.async { [self] in
            if running && !finishing {
                lines.removeAll { $0.id != open.id }
            } else {
                open.dropped = true
                lines.removeAll()
            }
            onUpdate?(lines)
        }
    }

    func feed(_ samples: [Float]) {
        audioQueue.async { [self] in
            guard running else { return }
            if finishing { incoming += samples; return }
            audio += samples
            heard += samples.count
            schedule()
        }
    }

    /// Returns once the model has nothing to do (no grey run due, no agreement run under way, none running, not held).
    func idle() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            audioQueue.async { [self] in
                waiting.append(c)
                schedule()
            }
        }
    }

    /// Streams a recorded 16 kHz clip as the mic would, in 100 ms pieces, then a second of silence. Each piece goes in
    /// once the model has caught up with the last (an agreement run due finishes within that piece), so a clip always
    /// gets the same runs, in the same order, on any machine; `pace` also waits 100 ms a piece, like the mic.
    func play(_ samples: [Float], pace: Bool) async {
        let all = samples + [Float](repeating: 0, count: Self.rate)
        var offset = 0
        while offset < all.count, !Task.isCancelled {
            let end = min(offset + Self.runEvery, all.count)
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

    /// Where window k ends: its agreement run can start once the mic has heard this far.
    private static func readyAt(_ k: Int) -> Int { k * step + window }

    /// A fresh line and windows for a new listening session.
    private func newSession() {
        open = Open()
        audio = []; kept = 0; heard = 0; lastGrey = 0
        black = ""; context = ""
        next = -2; agreeing = nil; agreed = nil
        greyFrom = nil; greyText = ""; credit = 0
    }

    private var listening: Bool { running && !finishing }
    /// Where grey runs start: the start of the latest finished window (0 before any).
    private var greyStart: Int { agreed.map { max(0, $0.k * Self.step) } ?? 0 }

    /// The mic's audio from `from` to `to` (samples since listening began).
    private func slice(_ from: Int, _ to: Int) -> [Float] { Array(audio[(from - kept)..<(to - kept)]) }

    /// Give the model its next piece of work if it is free: a grey run when one is due and the agreement run under way
    /// has had its share (or there is none), else a step of the agreement run, else the last run after a stop.
    private func schedule() {
        guard !busy, !holding else { return }
        // While listening: the next window, once the mic has heard it to its end (after Stop, only the run end() left
        // under way finishes).
        if listening, agreeing == nil, heard >= Self.readyAt(next) {
            agreeing = Agreement(k: next, audio: slice(max(0, next * Self.step), Self.readyAt(next)), context: context)
        }
        if listening, let a = agreeing, heard - Self.readyAt(a.k) >= Self.behind {
            let late = (heard - Self.readyAt(a.k)) / 16
            Self.log.error("fell behind: window \(a.k) still unfinished \(late) ms after its end")
            fallBehind()
            return
        }
        let greyDue = listening && heard - lastGrey >= Self.runEvery && heard - greyStart >= Self.shortest
        if greyDue && (credit <= 0 || agreeing == nil) {
            runGrey()
        } else if let a = agreeing {
            runStep(a)
        } else if finishing {
            runLast()
        } else {
            for c in waiting { c.resume() }
            waiting = []
        }
    }

    /// The latest finished agreement run, if it heard the full 8 s that a longer window from `from` starts with.
    private func reusable(_ from: Int) -> Agreement? {
        guard let a = agreed, a.k >= 0, a.run?.finished == true, heard - from > Self.window else { return nil }
        return a
    }

    private func runGrey() {
        lastGrey = heard
        let from = greyStart
        let base = reusable(from)
        // With the agreement run's first block comes its context; else the black text's end, with the cached prompt.
        work(Job(kind: .grey, audio: slice(from, heard), context: base?.context ?? context,
                 prefix: greyFrom == from ? greyText : (agreed?.text ?? ""),
                 base: base, cache: base == nil ? cache : nil, line: open, from: from))
    }

    /// The last run after a stop (no agreement run under way any more): afresh from the latest finished window's start.
    /// The next session, if listening started again meanwhile, begins with the audio it has heard so far.
    private func runLast() {
        let from = greyStart
        let base = reusable(from)
        let job = Job(kind: .last, audio: slice(from, heard), context: base?.context ?? context, prefix: "", base: base,
                      cache: nil, line: open, from: from, black: black, agreed: agreed?.text ?? "")
        finishing = false
        let queued = incoming
        incoming = []
        newSession()
        if running { audio = queued; heard = queued.count }
        work(job)
    }

    /// A grey or last run on the model.
    private func work(_ job: Job) {
        busy = true
        modelQueue.async { [self] in
            let t0 = Date()
            // The grey fills in token by token while the model writes it.
            let text = transcribe(job) { [weak self] sofar in
                self?.audioQueue.async { self?.showPartial(job, sofar) }
            }
            let took = Date().timeIntervalSince(t0) * 1000
            audioQueue.async { [self] in
                busy = false
                switch job.kind {
                case .grey:
                    if agreeing == nil {
                        credit = 0
                    } else {
                        let f = share()
                        credit += took * f / (1 - f)
                    }
                    greyRan(job, text)
                case .last:
                    finished(job, text)
                }
                passed()
            }
        }
    }

    /// One step of an agreement run on the model; its time is taken off the grey runs' credit (see the top).
    private func runStep(_ a: Agreement) {
        busy = true
        modelQueue.async { [self] in
            let t0 = Date()
            let done: Bool
            do {
                let run = a.run ?? asr.agreement(a.audio, a.context)
                a.run = run
                done = try run.step()
            } catch {
                Self.log.error("agreement run over window \(a.k) failed: \(error.localizedDescription, privacy: .public)")
                done = true
            }
            let took = Date().timeIntervalSince(t0) * 1000
            audioQueue.async { [self] in
                busy = false
                guard a === agreeing else { schedule(); return }   // the session ended meanwhile
                a.steps += 1
                a.ms += took
                a.longest = max(a.longest, took)
                credit -= took
                if done {
                    // Only a finished run counts as a run (onPass, holds); a failed one just makes way for the next.
                    let ok = a.run?.finished == true
                    agreementFinished(a)
                    if ok { passed() } else { schedule() }
                } else {
                    schedule()
                }
            }
        }
    }

    /// The share of the model's time the agreement run under way needs to end before the next window is due, at most 0.9:
    /// its remaining steps (as many tokens as the last run wrote, at least 1) at its mean step time so far (else the
    /// last run's).
    private func share() -> Double {
        guard let a = agreeing else { return 0 }
        let mean = a.steps > 0 ? a.ms / Double(a.steps) : (lastStepMs ?? Self.firstStepGuess)
        let left = Double(max(1, Self.prefillSteps + lastTokens - a.steps)) * mean
        let until = Double(Self.readyAt(a.k + 1) - heard) / 16
        return until <= 0 ? 0.9 : min(0.9, left / until)
    }

    /// A count of finished model runs, and the hold for screenshots.
    private func passed() {
        passes += 1
        onPass?(passes)
        if holdAfter.contains(passes) {
            Self.log.notice("held after run \(self.passes)")
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

    /// A grey run ended: what it wrote after the point where its text meets the black text is grey.
    private func greyRan(_ job: Job, _ text: String) {
        guard job.line === open else { return }
        greyFrom = job.from
        greyText = text
        open.grey = Self.after(black, in: text)
        show(open)
    }

    /// An agreement run ended: merge its text with the last window's on their longest common stretch. The last
    /// window's text up to the end of it turns black (from where the black text so far meets it); grey runs then start
    /// at this window, and the audio before it goes.
    private func agreementFinished(_ a: Agreement) {
        agreeing = nil
        next += 1
        credit = 0
        // A failed run changes nothing: the next window merges with the last good one, grey runs keep their start.
        guard let run = a.run, run.finished else {
            Self.log.notice("agreement run over window \(a.k) failed after \(a.steps) steps; keeping window \(self.agreed?.k ?? -3)")
            return
        }
        a.text = run.text.trimmingCharacters(in: .whitespacesAndNewlines)
        lastStepMs = a.ms / Double(a.steps)
        lastTokens = max(1, a.steps - Self.prefillSteps)
        Self.log.notice("agreement run over window \(a.k) (\(a.audio.count / 16) ms) in \(Int(a.ms)) ms, \(a.steps) steps, longest \(Int(a.longest)) ms: \(a.text, privacy: .private)")
        Self.log.notice("memory: \(Self.memory(), privacy: .public)")
        let more = Self.merged(black: black, old: agreed?.text ?? "", new: a.text)
        black += more
        open.black += more
        context = String(black.suffix(Self.contextChars).drop { $0.isWhitespace })
        agreed = a
        let keep = greyStart
        audio.removeFirst(keep - kept)
        kept = keep
        open.grey = Self.after(black, in: greyText)
        splitSentences(open)
        show(open)
    }

    /// Merging the last window's text `old` with `new` on their longest common stretch: the text of `old` that turns
    /// black, from where the black text so far ends in it (only the stretch itself if it isn't there) to the end of the
    /// stretch. Empty if they share no stretch of at least 2 characters.
    static func merged(black: String, old: String, new: String) -> String {
        let o = Array(old), n = Array(new)
        guard let m = longestCommon(o, n), m.length >= overlap else {
            if !o.isEmpty { log.notice("merge: the two windows share no stretch, nothing turns black") }
            return ""
        }
        let end = m.a + m.length
        let from = black.isEmpty ? 0 : (meets(black, in: o) ?? m.a)
        log.notice("merge on \(m.length) chars: black to \(end) of the last window's \(o.count)")
        return end > from ? String(o[from..<end]) : ""
    }

    /// Where the black text so far ends in `text`: just after the longest end of `black` (at least 2 characters) that
    /// `text` contains, and any punctuation that follows it there, or nil. Punctuation and spaces don't count, as two
    /// runs often differ only in them (「到。」 and 「到，」).
    static func meets(_ black: String, in text: [Character]) -> Int? {
        let skip = { (c: Character) in c.isPunctuation || c.isWhitespace }
        let tail = Array(black.suffix(60).filter { !skip($0) }.suffix(40))
        let at = text.indices.filter { !skip(text[$0]) }       // positions of the characters that count
        let core = at.map { text[$0] }
        guard tail.count >= overlap, !core.isEmpty else { return nil }
        for k in stride(from: tail.count, through: overlap, by: -1) {
            let end = Array(tail[(tail.count - k)...])
            guard k <= core.count else { continue }
            if let i = (0...(core.count - k)).first(where: { Array(core[$0..<($0 + k)]) == end }) {
                var j = at[i + k - 1] + 1
                while j < text.count, skip(text[j]) { j += 1 }
                return j
            }
        }
        return nil
    }

    /// What `text` has after the point where it meets the black text (all of it if they don't meet).
    static func after(_ black: String, in text: String) -> String {
        let chars = Array(text)
        return String(chars[(meets(black, in: chars) ?? 0)...]).trimmingCharacters(in: .whitespaces)
    }

    /// The longest stretch two texts have in common: where it starts in each, and its length (characters, ignoring
    /// case). Nil if they share nothing.
    static func longestCommon(_ a: [Character], _ b: [Character]) -> (a: Int, b: Int, length: Int)? {
        guard !a.isEmpty, !b.isEmpty else { return nil }
        let same = { (x: Character, y: Character) in x == y || x.lowercased() == y.lowercased() }
        var row = [Int](repeating: 0, count: b.count + 1)
        var best: (a: Int, b: Int, length: Int)?
        for i in 1...a.count {
            var prev = 0
            for j in 1...b.count {
                let keep = row[j]
                row[j] = same(a[i - 1], b[j - 1]) ? prev + 1 : 0
                if row[j] > (best?.length ?? 0) { best = (i - row[j], j - row[j], row[j]) }
                prev = keep
            }
        }
        return best
    }

    /// The last run after a stop ended: all of its text is black and the line is finished.
    /// The latest finished window is merged with it first, so the audio that window heard isn't lost.
    private func finished(_ job: Job, _ text: String) {
        let line = job.line
        let more = Self.merged(black: job.black, old: job.agreed, new: text)
        line.black += more + Self.after(job.black + more, in: text)
        line.grey = ""
        splitSentences(line)
        close(line)
    }

    /// The phone fell behind: listening stops, the black text stays.
    private func fallBehind() {
        running = false
        open.grey = ""
        close(open)
        newSession()
        for c in waiting { c.resume() }
        waiting = []
        onFail?(Self.tooSlow)
    }

    /// Sentences finished in the line's black text become lines of their own, before it.
    private func splitSentences(_ line: Open) {
        guard !line.dropped else { return }
        while let end = line.black.firstIndex(where: { "。！？!?".contains($0) }) {
            let sentence = String(line.black[...end])
            line.black = String(line.black[line.black.index(after: end)...].drop { $0.isWhitespace })
            let done = Line(id: newId(), settled: convert(sentence), tail: "", final: true)
            if let id = line.id, let i = lines.firstIndex(where: { $0.id == id }) { lines.insert(done, at: i) }
            else { lines.append(done) }
        }
    }

    /// Put the line on screen in HK Traditional, after its sentences.
    private func show(_ line: Open) {
        guard !line.dropped else { return }
        let shown = Line(id: line.id ?? 0, settled: convert(line.black), tail: convert(line.grey), final: false)
        if let id = line.id, let i = lines.firstIndex(where: { $0.id == id }) {
            lines[i] = shown
        } else if line.id == nil, !(line.black + line.grey).isEmpty {
            line.id = newId()
            lines.append(Line(id: line.id!, settled: shown.settled, tail: shown.tail, final: false))
        }
        onUpdate?(lines)
    }

    /// The line is finished: what's left of it stays as a black line (or goes, if nothing is left).
    private func close(_ line: Open) {
        guard !line.dropped else { onUpdate?(lines); return }
        if let id = line.id, let i = lines.firstIndex(where: { $0.id == id }) {
            if line.black.isEmpty { lines.remove(at: i) }
            else { lines[i] = Line(id: id, settled: convert(line.black), tail: "", final: true) }
        } else if line.id == nil, !line.black.isEmpty {
            line.id = newId()
            lines.append(Line(id: line.id!, settled: convert(line.black), tail: "", final: true))
        }
        line.dropped = true   // nothing more comes for it
        onUpdate?(lines)
    }

    private func newId() -> Int {
        nextId += 1
        return nextId
    }

    /// Show the text a run is writing (what follows the black text), once it is at least as long as the grey it
    /// replaces (no flicker).
    private func showPartial(_ job: Job, _ sofar: String) {
        let line = job.line
        guard line === open || job.kind == .last, !line.dropped else { return }
        let known = job.kind == .last ? job.black : black
        // Until the run has written past the end of the black text, its text is only the black text again.
        guard known.isEmpty || Self.meets(known, in: Array(sofar)) != nil else { return }
        let grey = Self.after(known, in: sofar)
        guard grey.count >= line.grey.count else { return }
        line.grey = grey
        show(line)
    }

    /// A grey or last run on the model, its prefix trimmed first; on an error, the prefix as given (untrimmed, so a
    /// failed run doesn't wear it down).
    private func transcribe(_ job: Job, onText: @escaping @Sendable (String) -> Void) -> String {
        let prefix = job.prefix.isEmpty ? "" : asr.trim(job.prefix)
        guard job.audio.count >= Self.shortest else { return job.prefix }
        let t0 = Date()
        let hits = job.cache?.hits ?? 0
        let reuses = job.base?.run?.reuses ?? 0
        do {
            let text = try (job.kind == .grey
                ? asr.grey(job.audio, job.context, prefix, job.base?.run, job.cache, onText)
                : asr.last(job.audio, job.context, job.base?.run, onText))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let took = Int(Date().timeIntervalSince(t0) * 1000)
            let notes = ((job.cache?.hits ?? 0) > hits ? " (cached prompt)" : "")
                + ((job.base?.run?.reuses ?? 0) > reuses ? " (first 8 s from window \(job.base?.k ?? 0)'s agreement run)" : "")
            Self.log.notice("\(job.kind.rawValue, privacy: .public) run over \(job.audio.count / 16) ms with \(job.context.count) chars of context from \(prefix.count) chars in \(took) ms\(notes, privacy: .public): \(text, privacy: .private)")
            Self.log.notice("memory: \(Self.memory(), privacy: .public)")
            return text.hasPrefix(prefix) ? text : job.prefix
        } catch {
            Self.log.error("\(job.kind.rawValue, privacy: .public) run failed after \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(error.localizedDescription, privacy: .public)")
            return job.prefix
        }
    }

    /// The app's memory footprint (what iOS counts against its limit), the headroom iOS still allows, and MLX's live and
    /// pooled bytes, in MB.
    static func memory() -> String {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let ok = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        } == KERN_SUCCESS
        let mb = { (b: Int) in b / 1_048_576 }
        var out = "footprint \(ok ? mb(Int(info.phys_footprint)) : -1)"
        #if os(iOS)
        out += ", headroom \(mb(Int(os_proc_available_memory())))"
        #endif
        #if !targetEnvironment(simulator)
        out += ", MLX active \(mb(Qwen3ASRMLXMemory.active)) pooled \(mb(Qwen3ASRMLXMemory.cached)) peak \(mb(Qwen3ASRMLXMemory.peak))"
        #endif
        return out
    }
}
