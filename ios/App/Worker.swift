import Foundation
import os

/// Speech detection over the mic's audio (Silero, `Models.swift`).
protocol Segmenter: AnyObject {
    /// Takes the next audio; returns the stretches of speech that ended in it, each with where it starts (in samples
    /// since `reset`).
    func feed(_ audio: [Float]) -> [(start: Int, audio: [Float])]
    /// Ends the stretch under way (Stop) and returns it.
    func flush() -> [(start: Int, audio: [Float])]
    /// A stretch of speech has begun and not yet ended.
    var speaking: Bool { get }
    func reset()
}

/// The loop, in three parts:
/// - Silero (speech detection, CPU) cuts the mic's audio into stretches of speech at pauses of 0.5 s (a stretch is cut
///   at 15 s if nobody pauses).
/// - While someone speaks, SenseVoice (CPU) transcribes what has been said since the last cut, at most twice a second:
///   the grey text. Once a stretch ends, SenseVoice hears it once more whole while it waits for Qwen3-ASR.
/// - Each finished stretch goes once through Qwen3-ASR 1.7B (GPU), with the last 100 characters of black text as
///   context: its text turns black in place of the stretch's grey. Stretches go in order, one at a time.
/// Nothing runs while nobody speaks. Stop ends the stretch under way, which goes through Qwen3-ASR like the others.
/// iOS allows no GPU work in the background (`suspend`): Qwen3-ASR waits until the app is back.
/// Black text is split into lines at the end of each sentence.
final class Worker: @unchecked Sendable {
    static let log = Logger(subsystem: "dev.de0ch.cantonese", category: "model")

    /// One line on screen: `settled` (black) never changes again, `tail` (grey) is still being revised. A final line
    /// has no tail.
    struct Line: Identifiable, Equatable { let id: Int; var settled: String; var tail: String; var final: Bool }

    /// The models as the loop calls them (the real ones: `Models.swift`; the simulator's stand-in for Qwen3-ASR
    /// replays a recording).
    struct Models {
        let segmenter: Segmenter
        /// SenseVoice: grey text for some audio.
        let grey: ([Float]) -> String
        /// Qwen3-ASR: black text for a finished stretch, with older black text as context.
        let black: (_ audio: [Float], _ context: String) throws -> String
    }

    static let rate = 16000
    /// Grey runs start at most this often (in samples of audio heard).
    static let greyEvery = rate / 2
    /// Silence kept before speech for the grey runs (the detector starts a stretch a little before it hears speech).
    static let keptSilence = rate / 2
    static let contextChars = 100

    /// Every line on screen, after each change.
    var onUpdate: (([Line]) -> Void)?
    /// Each black run, with the stretch and its context, and what came back (the tool that records them).
    var onBlack: ((_ audio: [Float], _ context: String, _ text: String) -> Void)?

    private let models: Models
    private let convert: (String) -> String
    private let queue = DispatchQueue(label: "cantonese.loop")   // everything below but `context`
    private let cpu = DispatchQueue(label: "cantonese.grey", qos: .userInitiated)
    private let gpu = DispatchQueue(label: "cantonese.black", qos: .userInitiated)

    private var listening = false
    /// Audio since the last cut (what a grey run hears), and where it starts.
    private var heard: [Float] = []
    private var heardStart = 0
    private var total = 0
    private var lastGrey = 0
    private var greyRunning = false
    /// Counts cuts: a grey run for the stretch before a cut is dropped.
    private var stretch = 0
    /// Grey text of the stretch under way.
    private var live = ""
    /// Black lines, and the stretches waiting for Qwen3-ASR (grey only).
    private var lines: [Line] = []
    private var nextId = 0
    /// Runs queued or under way on the CPU and GPU.
    private var busy = 0
    /// The black text so far, for context; only touched on the GPU queue.
    private var context = ""
    private let gate = NSCondition()
    private var suspended = false

    init(models: Models, convert: @escaping (String) -> String) {
        self.models = models
        self.convert = convert
    }

    func configure(onUpdate: @escaping ([Line]) -> Void) {
        queue.sync { self.onUpdate = onUpdate }
    }

    func begin() {
        queue.async { [self] in
            models.segmenter.reset()
            heard = []; heardStart = 0; total = 0; lastGrey = 0; live = ""; stretch += 1
            listening = true
        }
    }

    /// The stretch under way goes through Qwen3-ASR like the others.
    func end() {
        queue.async { [self] in
            guard listening else { return }
            listening = false
            for s in models.segmenter.flush() { finish(s.audio) }
            live = ""; stretch += 1
            publish()
        }
    }

    func clear() {
        queue.async { [self] in
            lines.removeAll { $0.final }
            publish()
        }
    }

    /// The app went to the background (no GPU work allowed) or came back.
    func suspend(_ on: Bool) {
        gate.lock()
        suspended = on
        gate.broadcast()
        gate.unlock()
    }

    /// The mic's audio, 16 kHz mono.
    func feed(_ audio: [Float]) {
        queue.async { [self] in take(audio) }
    }

    private func take(_ audio: [Float]) {
        guard listening else { return }
        heard += audio
        total += audio.count
        for s in models.segmenter.feed(audio) {
            let end = s.start + s.audio.count
            heard.removeFirst(min(heard.count, max(0, end - heardStart)))
            heardStart = max(heardStart, end)
            finish(s.audio)
            live = ""; stretch += 1
        }
        if !models.segmenter.speaking, heard.count > Self.keptSilence {
            heardStart += heard.count - Self.keptSilence
            heard.removeFirst(heard.count - Self.keptSilence)
        }
        if models.segmenter.speaking, total - lastGrey >= Self.greyEvery { startGrey() }
        publish()
    }

    /// A grey run over what has been said since the last cut.
    private func startGrey() {
        guard !greyRunning, !heard.isEmpty else { return }
        greyRunning = true
        lastGrey = total
        busy += 1
        let (audio, at) = (heard, stretch)
        cpu.async { [self] in
            let text = tidy(models.grey(audio))
            queue.async { [self] in
                greyRunning = false
                busy -= 1
                if at == stretch, listening { live = text; publish() }
            }
        }
    }

    /// A stretch has ended: it waits for Qwen3-ASR as a grey line (SenseVoice hears it once more whole meanwhile).
    private func finish(_ audio: [Float]) {
        let id = nextId
        nextId += 1
        lines.append(Line(id: id, settled: "", tail: live, final: false))
        busy += 2
        cpu.async { [self] in
            let text = tidy(models.grey(audio))
            queue.async { [self] in
                busy -= 1
                if let i = lines.firstIndex(where: { $0.id == id && !$0.final }), !text.isEmpty {
                    lines[i].tail = text
                    publish()
                }
            }
        }
        gpu.async { [self] in
            gate.lock()
            while suspended { gate.wait() }
            gate.unlock()
            let t0 = Date()
            var text: String
            do {
                text = try models.black(audio, context)
            } catch {
                Self.log.error("black run failed: \(error.localizedDescription, privacy: .public)")
                text = models.grey(audio)
            }
            onBlack?(audio, context, text)
            let black = convert(text.trimmingCharacters(in: .whitespacesAndNewlines))
            context = String((context + black).suffix(Self.contextChars))
            Self.log.notice("black run: \(audio.count * 1000 / Self.rate) ms of audio in \(Int(Date().timeIntervalSince(t0) * 1000)) ms: \(black, privacy: .public)")
            queue.async { [self] in
                busy -= 1
                guard let i = lines.firstIndex(where: { $0.id == id }) else { return }
                var new: [Line] = []
                for s in Self.sentences(black) {
                    new.append(Line(id: new.isEmpty ? id : nextId, settled: s, tail: "", final: true))
                    if new.count > 1 { nextId += 1 }
                }
                lines.replaceSubrange(i...i, with: new)
                publish()
            }
        }
    }

    private func publish() {
        var out = lines.filter { !$0.settled.isEmpty || !$0.tail.isEmpty }
        if !live.isEmpty { out.append(Line(id: -1, settled: "", tail: live, final: false)) }
        onUpdate?(out)
    }

    /// SenseVoice's text in HK Traditional, its English in lower case (it writes English in capitals).
    private func tidy(_ text: String) -> String {
        convert(text.trimmingCharacters(in: .whitespaces)).lowercased()
    }

    /// Text split after each 。！？!?, the punctuation kept.
    static func sentences(_ text: String) -> [String] {
        var out: [String] = []
        var cur = ""
        for c in text {
            cur.append(c)
            if "。！？!?".contains(c) { out.append(cur.trimmingCharacters(in: .whitespaces)); cur = "" }
        }
        let rest = cur.trimmingCharacters(in: .whitespaces)
        if !rest.isEmpty || out.isEmpty { out.append(rest) }
        return out
    }

    /// Nothing waiting or running.
    var idle: Bool { queue.sync { busy == 0 } }

    /// Streams a clip as the mic would (UI tests, the tool that records the black runs): 0.1 s at a time, in real time
    /// if `pace`. At each second in `holds` the clip stops until the loop is idle with the grey text for all of it, then
    /// `onHold(true)` and `holdFor` seconds before it goes on. A second of silence follows the clip.
    func play(_ samples: [Float], pace: Bool, holds: [Double] = [], holdFor: Double = 0,
              onHold: @escaping (Bool) -> Void = { _ in }) async {
        let chunk = Self.rate / 10
        var holds = holds.map { Int($0 * Double(Self.rate)) }.sorted()
        let all = samples + [Float](repeating: 0, count: Self.rate)
        var i = 0
        while i < all.count, !Task.isCancelled {
            let n = min(chunk, all.count - i, (holds.first ?? Int.max) - i)
            if n > 0 {
                feed(Array(all[i..<(i + n)]))
                i += n
                if pace { try? await Task.sleep(nanoseconds: UInt64(n) * 1_000_000_000 / UInt64(Self.rate)) }
            }
            if let h = holds.first, i >= h {
                holds.removeFirst()
                await settle()
                queue.async { [self] in startGrey() }
                await settle()
                onHold(true)
                try? await Task.sleep(nanoseconds: UInt64(holdFor * 1e9))
                onHold(false)
            }
        }
        await settle()
    }

    /// Waits until nothing is waiting or running.
    func settle() async {
        while !idle { try? await Task.sleep(nanoseconds: 50_000_000) }
    }

    /// 16-bit PCM WAV data as samples.
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
}
