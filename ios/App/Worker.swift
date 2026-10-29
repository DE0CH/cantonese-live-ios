import Foundation
import os
import Qwen3ASR

/// The model and everything that touches it. Audio and text state live on audioQueue, the models run on modelQueue.
///
/// The loop (a sliding window): the window is the mic's audio from `start` to now. Whenever the model is free and at
/// least 100 ms of audio has come in since the last run started:
/// - window of 8 s or more (a slide): the model transcribes its first 8 s, continuing from the black text of its first
///   8/3 s, with the older black text as context; the forced aligner times that text, the words starting before 16/3 s
///   turn black, the rest is grey; the window then starts 8/3 s later.
/// - shorter (a grey refresh): the model transcribes the whole window the same way; what follows the black text is grey.
/// - 10 s or more behind (window over 18 s): the phone can't keep up and listening stops with an error.
/// Before the first slide nothing is black. Stopping runs the model once more over what is left, all of it black.
/// Black text is split into lines at the end of each sentence; the last line holds the unfinished black text and the grey.
final class Worker: @unchecked Sendable {
    static let log = Logger(subsystem: "dev.de0ch.cantonese", category: "model")

    /// One line on screen: `settled` (black) never changes again, `tail` (grey) is still being revised. A final line
    /// has no tail.
    struct Line: Identifiable, Equatable { let id: Int; var settled: String; var tail: String; var final: Bool }
    /// A word the aligner timed: its text as written (with punctuation), start and end in seconds from the audio's start.
    struct Word: Codable, Sendable { let text: String; let start: Float; let end: Float }
    /// The model: transcribe `audio` with `context` (older text, in the system prompt) continuing from `start`
    /// (written as if the model had already said it; the answer includes it). `cache` keeps the prompt before the audio
    /// between calls with the same context. `onText` gets the text so far after each token.
    typealias ASR = (_ audio: [Float], _ context: String, _ start: String, _ cache: Qwen3PromptCache?,
                     _ onText: @escaping @Sendable (String) -> Void) throws -> String
    /// The forced aligner: when each word of `text` starts and ends in `audio`.
    typealias Aligner = (_ audio: [Float], _ text: String) throws -> [Word]

    /// Every line on screen, after each change.
    var onUpdate: (([Line]) -> Void)?
    /// After each model run, with how many have finished (the runs over a clip played with `play` are always the same).
    var onPass: ((Int) -> Void)?
    /// Listening stopped by itself (the phone fell behind), with the message to show.
    var onFail: ((String) -> Void)?
    /// For the App Store screenshots: after these runs (counted from 1) the loop stops for `holdFor` seconds, with
    /// `onHold(true)` while it waits, so the screen can be captured exactly as that run left it.
    var holdAfter: Set<Int> = []
    var holdFor: TimeInterval = 5
    var onHold: ((Bool) -> Void)?

    static let rate = 16000
    static let window = 8 * rate                     // a slide transcribes the window's first 8 s...
    static let blackBefore: Float = 16.0 / 3          // ...its words starting before 16/3 s turn black...
    static func start(after slides: Int) -> Int { slides * window / 3 }   // ...and the window moves on 8/3 s
    static let runEvery = rate / 10                  // a run at most every 100 ms of audio
    static let shortest = rate / 4                   // the model gets at least 0.25 s
    static let behind = 10 * rate                    // window - 8 s >= 10 s: the phone can't keep up
    static let contextChars = 100                    // older black text given as context
    static var tooSlow: String { String(localized: "部手機轉得唔夠快，跟唔上，停咗。") }

    static let modelId = "aufklarer/Qwen3-ASR-1.7B-MLX-5bit"
    static let alignerId = "aufklarer/Qwen3-ForcedAligner-0.6B-5bit"

    private let asr: ASR
    private let align: Aligner
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
    private enum Kind: String { case grey, slide, last }
    private struct Job {
        let kind: Kind; let audio: [Float]; let context: String; let start: String; let cache: Qwen3PromptCache?
        let line: Open; let at: Int   // the window's start (samples since listening began)
    }

    // audioQueue state.
    private var lines: [Line] = []
    private var nextId = 0
    private var open = Open()
    private var audio: [Float] = []       // the window: from Self.start(after: slides) to `heard`
    private var heard = 0                 // samples since listening began
    private var slides = 0
    private var lastRun = 0               // `heard` when the last run started
    private var blackWords: [(text: String, at: Int)] = []   // black words in the window (the start text of a run)
    private var context = ""              // black text before the window
    private var cache: Qwen3PromptCache?  // grey refreshes between two slides share it (same context)
    private var incoming: [Float] = []    // mic audio of a new session while the last one's last run waits
    private var busy = false
    private var running = false
    private var finishing = false         // stopped; the last run is still to start
    private var passes = 0                // model runs finished
    private var holding = false
    private var waiting: [CheckedContinuation<Void, Never>] = []   // `idle()` callers

    init(asr: @escaping ASR, align: @escaping Aligner, convert: @escaping (String) -> String) {
        self.asr = asr
        self.align = align
        self.convert = convert
    }

    /// The model as the loop calls it: up to 200 tokens, the text so far after every token.
    static func asr(_ model: Qwen3ASRModel) -> ASR {
        { audio, context, start, cache, onText in
            var options = Qwen3DecodingOptions()
            options.maxTokens = 200
            options.context = context.isEmpty ? nil : context
            options.prefix = start
            options.promptCache = cache
            options.onText = onText
            return try model.transcribeThrowing(audio: audio, sampleRate: rate, options: options)
        }
    }

    static func aligner(_ aligner: Qwen3ForcedAligner) -> Aligner {
        { audio, text in
            aligner.align(audio: audio, text: text, sampleRate: rate, language: "Chinese")
                .map { Word(text: $0.text, start: $0.startTime, end: $0.endTime) }
        }
    }

    func begin() {
        audioQueue.async { [self] in
            running = true
            if !finishing { newSession() }
        }
    }

    /// Stop listening: one last run over what is left of the window, all of it black.
    func end() {
        audioQueue.async { [self] in
            guard running else { return }
            running = false
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

    /// Returns once the model has nothing to do (no run due, none running, not held).
    func idle() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            audioQueue.async { [self] in
                waiting.append(c)
                schedule()
            }
        }
    }

    /// Streams a recorded 16 kHz clip as the mic would, in 100 ms pieces, then a second of silence. Each piece goes in
    /// once the model has caught up with the last, so a clip always gets the same runs on any machine; `pace` also
    /// waits 100 ms a piece, like the mic.
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

    /// A fresh window and line for a new listening session.
    private func newSession() {
        open = Open()
        audio = []; heard = 0; slides = 0; lastRun = 0
        blackWords = []; context = ""; cache = nil
    }

    /// The text a run starts from: the black words in the window.
    private var forced: String {
        String(blackWords.map(\.text).joined().drop { $0.isWhitespace })
    }

    /// Start the next run if the model is free: the last one after a stop, a slide once the window holds 8 s, else a
    /// grey refresh.
    private func schedule() {
        guard !busy, !holding else { return }
        let job: Job
        if finishing {
            job = Job(kind: .last, audio: audio, context: context, start: forced, cache: nil, line: open, at: Self.start(after: slides))
            finishing = false
            let next = incoming
            incoming = []
            newSession()
            if running { audio = next; heard = next.count }
        } else if running, heard - lastRun >= Self.runEvery, audio.count >= Self.shortest {
            if audio.count - Self.window >= Self.behind {
                Self.log.error("fell behind: window \(self.audio.count / 16) ms")
                fallBehind()
                return
            }
            lastRun = heard
            if audio.count >= Self.window {
                job = Job(kind: .slide, audio: Array(audio[..<Self.window]), context: context, start: forced, cache: nil,
                          line: open, at: Self.start(after: slides))
            } else {
                if cache == nil { cache = Qwen3PromptCache() }
                job = Job(kind: .grey, audio: audio, context: context, start: forced, cache: cache, line: open,
                          at: Self.start(after: slides))
            }
        } else {
            for c in waiting { c.resume() }
            waiting = []
            return
        }
        busy = true
        modelQueue.async { [self] in
            // The grey fills in token by token while the model writes it.
            let text = transcribe(job) { [weak self] sofar in
                self?.audioQueue.async { self?.showPartial(job, sofar) }
            }
            let words = job.kind == .slide ? timed(job, text) : []
            audioQueue.async { [self] in
                busy = false
                switch job.kind {
                case .grey: refreshed(job, text)
                case .slide: slid(job, text, words)
                case .last: finished(job, text)
                }
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
        }
    }

    /// A grey refresh ended: everything after the black text is grey.
    private func refreshed(_ job: Job, _ text: String) {
        guard job.line === open else { return }
        open.grey = String(text.dropFirst(job.start.count))
        show(open)
    }

    /// A slide ended: the words starting in the window's first 16/3 s are black, the rest grey; the window moves on
    /// 8/3 s, and the black words now before it become context.
    private func slid(_ job: Job, _ text: String, _ words: [Word]) {
        guard job.line === open else { return }
        var black = job.start.count     // characters of `text` that are black
        var last = blackWords.last?.at ?? 0
        let chars = Array(text)
        for (end, start) in Self.ends(of: Self.glued(words), in: text) where end > black {
            guard start < Self.blackBefore else { break }
            last = max(last, job.at + Int((start * Float(Self.rate)).rounded()))
            let piece = String(chars[black..<end])
            blackWords.append((piece, last))
            open.black += piece
            black = end
        }
        open.grey = String(chars[black...])
        slides += 1
        let from = Self.start(after: slides)
        audio.removeFirst(from - job.at)
        while let w = blackWords.first, w.at < from {
            context += w.text
            blackWords.removeFirst()
        }
        context = String(context.suffix(Self.contextChars).drop { $0.isWhitespace })
        cache = nil
        splitSentences(open)
        show(open)
    }

    /// The last run after a stop ended: all of its text is black and the line is finished.
    private func finished(_ job: Job, _ text: String) {
        let line = job.line
        line.black += text.dropFirst(job.start.count)
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

    /// The aligner gives the silence before a word to that word: after a pause the next word "starts" right where the
    /// last one ended (a word of 2 s, or of none with its neighbour 2 s later; the demo's 明 of 明天 turned black with the
    /// 「到。」 before the pause, and the next run, continuing from it, never wrote 天). A word not ending in punctuation
    /// is said with the word after it, so it starts at most `glue` seconds before that one.
    static let glue: Float = 0.32
    static func glued(_ words: [Word]) -> [Word] {
        words.indices.map { i in
            let w = words[i]
            guard i + 1 < words.count, let last = w.text.last, last.isLetter || last.isNumber else { return w }
            return Word(text: w.text, start: max(w.start, words[i + 1].start - glue), end: w.end)
        }
    }

    /// When each word of the run's text starts: for every word the aligner timed, the end of its text in `text`
    /// (characters; the aligner leaves out spaces between words) and its start in seconds.
    static func ends(of words: [Word], in text: String) -> [(end: Int, start: Float)] {
        let chars = Array(text)
        var i = 0
        var out: [(end: Int, start: Float)] = []
        for w in words {
            for c in w.text where !c.isWhitespace {
                while i < chars.count, chars[i] != c, chars[i].isWhitespace { i += 1 }
                guard i < chars.count, chars[i] == c else { return out }
                i += 1
            }
            out.append((end: i, start: w.start))
        }
        return out
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

    /// Show the text a run is writing, once its grey is at least as long as the grey it replaces (no flicker).
    private func showPartial(_ job: Job, _ sofar: String) {
        let line = job.line
        guard line === open || job.kind == .last, !line.dropped, sofar.hasPrefix(job.start) else { return }
        let grey = String(sofar.dropFirst(job.start.count))
        guard grey.count >= line.grey.count else { return }
        line.grey = grey
        show(line)
    }

    private func transcribe(_ job: Job, onText: @escaping @Sendable (String) -> Void) -> String {
        guard job.audio.count >= Self.shortest else { return job.start }
        let t0 = Date()
        let hits = job.cache?.hits ?? 0
        do {
            let text = try asr(job.audio, job.context, job.start, job.cache, onText)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let cached = (job.cache?.hits ?? 0) > hits ? " (cached prompt)" : ""
            Self.log.notice("\(job.kind.rawValue, privacy: .public) run over \(job.audio.count / 16) ms with \(job.context.count) chars of context from \(job.start.count) chars in \(Int(Date().timeIntervalSince(t0) * 1000)) ms\(cached, privacy: .public): \(text, privacy: .private)")
            return text.hasPrefix(job.start) ? text : job.start
        } catch {
            Self.log.error("\(job.kind.rawValue, privacy: .public) run failed after \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(error.localizedDescription, privacy: .public)")
            return job.start
        }
    }

    /// The aligner's times for a slide's text. If it has none, the text is spread evenly over the window.
    private func timed(_ job: Job, _ text: String) -> [Word] {
        guard !text.isEmpty else { return [] }
        let t0 = Date()
        var words: [Word] = []
        do { words = try align(job.audio, text) } catch {
            Self.log.error("alignment failed: \(error.localizedDescription, privacy: .public)")
        }
        Self.log.notice("alignment of \(text.count) chars over \(job.audio.count / 16) ms in \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(words.count) words")
        if Self.ends(of: words, in: text).isEmpty {
            let chars = Array(text)
            let seconds = Float(job.audio.count) / Float(Self.rate)
            words = chars.enumerated().map { i, c in
                Word(text: String(c), start: seconds * Float(i) / Float(chars.count), end: seconds * Float(i + 1) / Float(chars.count))
            }
        }
        return words
    }
}
