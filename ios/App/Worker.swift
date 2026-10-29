import Foundation
import os
import Qwen3ASR

/// The model and everything that touches it. Audio and text state live on audioQueue, the model runs on modelQueue.
///
/// The loop (a sliding window): the window is the mic's audio from `start` to now, at most 8 s. Whenever the model is
/// free and at least 100 ms of audio has come in since the last run started, the model transcribes the window afresh,
/// with the black text so far as context; what it writes after the point where its text meets the black text is grey.
/// Text turns black only where two complete 8 s windows agree: whenever the window holds 8 s, its text is merged with the
/// last complete window's (8/3 s earlier, overlapping it by 16/3 s) on their longest common stretch, as Hugging Face's
/// ASR pipeline merges overlapping chunks. The earlier window's text up to the end of that stretch is black; what
/// follows, which only the later window heard (or the earlier one guessed at its cut-off end), waits for the next merge.
/// The window then moves on 8/3 s. If the two share no stretch of at least 2 characters, nothing turns black that time.
/// 10 s or more behind (the window waiting at over 18 s): the phone can't keep up and listening stops with an error.
/// Stopping runs the model once more over what is left, all of it black.
/// Black text is split into lines at the end of each sentence; the last line holds the unfinished black text and the grey.
final class Worker: @unchecked Sendable {
    static let log = Logger(subsystem: "dev.de0ch.cantonese", category: "model")

    /// One line on screen: `settled` (black) never changes again, `tail` (grey) is still being revised. A final line
    /// has no tail.
    struct Line: Identifiable, Equatable { let id: Int; var settled: String; var tail: String; var final: Bool }
    /// The model: transcribe `audio` with `context` (older text, in the system prompt) continuing from `start`
    /// (written as if the model had already said it; the answer includes it). `cache` keeps the prompt before the audio
    /// between calls with the same context. `onText` gets the text so far after each token.
    typealias ASR = (_ audio: [Float], _ context: String, _ start: String, _ cache: Qwen3PromptCache?,
                     _ onText: @escaping @Sendable (String) -> Void) throws -> String

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
    static let window = 8 * rate                     // a run hears at most 8 s...
    static let step = window / 3                     // ...and a full window slides on 8/3 s
    static let overlap = 2                           // characters two texts must share to be merged
    static let runEvery = rate / 10                  // a run at most every 100 ms of audio
    static let shortest = rate / 4                   // the model gets at least 0.25 s
    static let behind = 10 * rate                    // window - 8 s >= 10 s: the phone can't keep up
    static let contextChars = 100                    // older black text given as context
    static var tooSlow: String { String(localized: "部手機轉得唔夠快，跟唔上，停咗。") }

    static let modelId = "aufklarer/Qwen3-ASR-1.7B-MLX-5bit"
    /// MLX keeps freed buffers for reuse; one run's arrays need well under this (8 s of audio: the encoder's first
    /// convolution, 8 × 480 × 64 × 50 floats, is the largest at 49 MB), and the default (a quarter of the phone's
    /// memory, 2 GB) on top of the weights got the app killed.
    static let cacheLimit = 256 * 1024 * 1024

    private let asr: ASR
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
    private enum Kind: String { case run, slide, last }
    private struct Job {
        let kind: Kind; let audio: [Float]; let context: String; let start: String; let cache: Qwen3PromptCache?
        let line: Open; let at: Int   // the window's start (samples since listening began)
        var black = ""                // the black text so far when it started (for the last run, which ends the session)
    }

    // audioQueue state.
    private var lines: [Line] = []
    private var nextId = 0
    private var open = Open()
    private var audio: [Float] = []       // the window: the audio not yet slid past
    private var heard = 0                 // samples since listening began
    private var start = 0                 // where the window starts (samples since listening began)
    private var lastRun = 0               // `heard` when the last run started
    private var black = ""                // all black text this session
    private var lastFull = ""             // the text of the last complete (8 s) window
    private var context = ""              // the end of the black text, given to every run
    private let cache = Qwen3PromptCache()   // the prompt before the audio, kept while the context stays the same
    private var incoming: [Float] = []    // mic audio of a new session while the last one's last run waits
    private var busy = false
    private var running = false
    private var finishing = false         // stopped; the last run is still to start
    private var passes = 0                // model runs finished
    private var holding = false
    private var waiting: [CheckedContinuation<Void, Never>] = []   // `idle()` callers

    init(asr: @escaping ASR, convert: @escaping (String) -> String) {
        self.asr = asr
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
        audio = []; heard = 0; start = 0; lastRun = 0
        black = ""; lastFull = ""; context = ""
    }

    /// Start the next run if the model is free: the last one after a stop, else a run over the window (sliding it first
    /// once it holds 8 s).
    private func schedule() {
        guard !busy, !holding else { return }
        let job: Job
        if finishing {
            job = Job(kind: .last, audio: Array(audio.prefix(Self.window)), context: context, start: "", cache: nil,
                      line: open, at: start, black: black)
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
            // A complete window (8 s) is merged with the last one when its run ends, then slides.
            job = Job(kind: audio.count >= Self.window ? .slide : .run, audio: Array(audio.prefix(Self.window)),
                      context: context, start: "", cache: cache, line: open, at: start)
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
            audioQueue.async { [self] in
                busy = false
                switch job.kind {
                case .run: ran(job, text)
                case .slide: slid(job, text)
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

    /// A run within the window ended: what it wrote after the point where its text meets the black text is grey.
    private func ran(_ job: Job, _ text: String) {
        guard job.line === open, job.at == start else { return }
        open.grey = Self.after(black, in: text)
        show(open)
    }

    /// A complete window's run ended: merge its text with the last complete window's on their longest common stretch.
    /// The last window's text up to the end of it turns black (from where the black text so far meets it); the window
    /// then moves on 8/3 s.
    private func slid(_ job: Job, _ text: String) {
        guard job.line === open, job.at == start else { return }
        let old = Array(lastFull), new = Array(text)
        if let m = Self.longestCommon(old, new), m.length >= Self.overlap {
            let end = m.a + m.length
            // From where the black text so far ends in it; if it isn't there, only the agreed stretch is new.
            let from = black.isEmpty ? 0 : (Self.meets(black, in: old) ?? m.a)
            if end > from {
                let more = String(old[from..<end])
                black += more
                open.black += more
            }
            Self.log.notice("merge on \(m.length) chars: black to \(end) of the last window's \(old.count)")
        } else if !old.isEmpty {
            Self.log.notice("merge: the two windows share no stretch, nothing turns black")
        }
        lastFull = text
        context = String(black.suffix(Self.contextChars).drop { $0.isWhitespace })
        audio.removeFirst(Self.step)
        start += Self.step
        open.grey = Self.after(black, in: text)
        splitSentences(open)
        show(open)
    }

    /// Where the black text so far ends in `text`: just after the longest end of `black` (at least 2 characters) that
    /// `text` contains, or nil.
    static func meets(_ black: String, in text: [Character]) -> Int? {
        let tail = Array(black.suffix(40))
        guard tail.count >= Self.overlap, !text.isEmpty else { return nil }
        for k in stride(from: tail.count, through: Self.overlap, by: -1) {
            let end = Array(tail[(tail.count - k)...])
            if let i = (0...max(0, text.count - k)).first(where: { i in i + k <= text.count && Array(text[i..<(i + k)]) == end }) {
                return i + k
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
    private func finished(_ job: Job, _ text: String) {
        let line = job.line
        line.black += Self.after(job.black, in: text)
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

    private func transcribe(_ job: Job, onText: @escaping @Sendable (String) -> Void) -> String {
        guard job.audio.count >= Self.shortest else { return job.start }
        let t0 = Date()
        let hits = job.cache?.hits ?? 0
        do {
            let text = try asr(job.audio, job.context, job.start, job.cache, onText)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let cached = (job.cache?.hits ?? 0) > hits ? " (cached prompt)" : ""
            Self.log.notice("\(job.kind.rawValue, privacy: .public) run over \(job.audio.count / 16) ms with \(job.context.count) chars of context from \(job.start.count) chars in \(Int(Date().timeIntervalSince(t0) * 1000)) ms\(cached, privacy: .public): \(text, privacy: .private)")
            Self.log.notice("memory: \(Self.memory(), privacy: .public)")
            return text.hasPrefix(job.start) ? text : job.start
        } catch {
            Self.log.error("\(job.kind.rawValue, privacy: .public) run failed after \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(error.localizedDescription, privacy: .public)")
            return job.start
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
