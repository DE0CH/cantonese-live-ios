// demo-passes MODEL_DIR CLIP.wav OUT.json [CLIP.wav OUT.json ...]
//   MODEL_DIR: the app's model folder (ios/Model from ios/fetch-model.sh: asr/)
//   OUT.json:  {runs: [{kind, audio, sum, context, prefix, text}], trims: {text: trimmed}}: every model call the app's
//              loop made over the clip (grey runs, agreement runs, the last run: the audio's length and sum, what it was
//              given, the answer) and every text it trimmed to continue from, for the simulator's stand-in
//              (testdata/…passes.json). Stops the loop after the clip like the Stop button, so the last run is recorded too.
// Checks (exit 1 on any difference): every grey or last run that reused an agreement run's first 8 s (its encoder output
// and decoder keys and values, or only its encoder output when too little is left to prefill) is run again without it
// (the same blockwise computation, nothing reused), every grey run that reused the cached prompt is run again without
// the cache, and every agreement run, taken in steps, is run again as one plain call; the texts must be the same. A clip
// long enough for them that got no agreement checks, or no reuse checks, fails too.
// The checks run between the loop's calls (when a run is counted, the model is free), so none of the times the loop
// measures (its log, its scheduling) include them. Timings per kind of call and memory are printed, with MLX's buffer
// cache capped as in the app.
import Foundation
import Qwen3ASR

let args = CommandLine.arguments
guard args.count >= 4, args.count % 2 == 0 else {
    FileHandle.standardError.write("usage: demo-passes MODEL_DIR CLIP.wav OUT.json [CLIP.wav OUT.json ...]\n".data(using: .utf8)!)
    exit(2)
}

struct Run: Codable { let kind: String; let audio: Int; let sum: Float; let context: String; let prefix: String; let text: String }
struct Passes: Codable { var runs: [Run] = []; var trims: [String: String] = [:] }
final class Record: @unchecked Sendable {
    var passes = Passes()
    var lines: [Worker.Line] = []
    var last = ""                       // what the last call did, for the log
    var times: [String: [Int]] = [:]    // ms per kind of call
    var reuseChecks = 0
    var encoderOnly = 0                 // of them, runs that reused only the encoder output
    var reuseMismatches = 0
    var agreeChecks = 0
    var agreeMismatches = 0
    var cacheChecks = 0
    var cacheMismatches = 0
    var failed = false
    var checks: [() -> String] = []     // calls to run again, each giving a note for the log ("" if all is well)
    /// Runs the checks waiting and returns their notes.
    func check() -> String {
        let waiting = checks
        checks = []
        return waiting.map { $0() }.joined()
    }
}
func ms(since t0: Date) -> Int { Int(Date().timeIntervalSince(t0) * 1000) }

/// An agreement run of the real model, timed step by step; `finish` gets it once it has finished.
final class Checked: AgreementRun {
    let run: Qwen3AgreementRun
    let audio: [Float]
    private let finish: (Checked) -> Void
    private let fail: (Error) -> Void
    var total = 0                       // ms over all steps, and the longest step
    var longest = 0
    init(_ run: Qwen3AgreementRun, audio: [Float], finish: @escaping (Checked) -> Void, fail: @escaping (Error) -> Void) {
        self.run = run
        self.audio = audio
        self.finish = finish
        self.fail = fail
    }
    /// Whether a blockwise call on `audio` with `context` (as Worker.model makes it) gets this run's encoder output for
    /// its first block, whether or not its keys and values are reused too.
    func standsIn(for audio: [Float], context: String) -> Bool {
        run.finished && run.audioCount == Qwen3ASRModel.blockSamples && audio.count > run.audioCount
            && run.context == context && self.audio.elementsEqual(audio[0..<run.audioCount])
    }
    var text: String { run.text }
    var finished: Bool { run.finished }
    var reuses: Int { run.reuses }
    func step() throws -> Bool {
        let before = run.finished
        let t0 = Date()
        let done: Bool
        do { done = try run.step() } catch { fail(error); throw error }
        let took = ms(since: t0)
        total += took
        longest = max(longest, took)
        if done && !before { finish(self) }
        return done
    }
}

let dir = URL(fileURLWithPath: args[1])
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: Worker.modelId, cacheDir: dir.appendingPathComponent("asr"), offlineMode: true)
let real = Worker.model(model)
let cacheCap = Qwen3ASRMLXMemory.limitCache(to: Worker.cacheLimit)
print("model loaded, \(Worker.memory())")
var failed = false
var totals: [String: [Int]] = [:]
var reuseChecks = 0, encoderOnly = 0, reuseMismatches = 0, agreeChecks = 0, agreeMismatches = 0, cacheChecks = 0, cacheMismatches = 0

for k in stride(from: 2, to: args.count, by: 2) {
    guard let wav = FileManager.default.contents(atPath: args[k]), let samples = Worker.samples(wav: wav) else {
        print("can't read \(args[k])"); exit(2)
    }
    print("=== \(args[k]): \(String(format: "%.1f", Double(samples.count) / 16000)) s")
    let record = Record()
    let seconds = { (audio: [Float]) in String(format: "%.2f", Double(audio.count) / 16000) }
    /// A grey or last run that reused an agreement run's first 8 s: the same run again without it, between the loop's calls.
    func checkReuse(_ text: String, again: @escaping () throws -> String) {
        record.checks.append {
            record.reuseChecks += 1
            let t1 = Date()
            do {
                let plain = try again()
                record.times["  the same, nothing reused", default: []].append(ms(since: t1))
                guard plain != text else { return "" }
                record.reuseMismatches += 1
                return "\n    REUSE MISMATCH: nothing reused \"\(plain)\""
            } catch {
                record.reuseMismatches += 1
                return "\n    REUSE MISMATCH: the run with nothing reused failed: \(error)"
            }
        }
    }
    let worker = Worker(model: Worker.Model(
        grey: { audio, context, prefix, base, cache, onText in
            let checked = base as? Checked
            let inner = checked?.run
            let reuses = inner?.reuses ?? 0
            let hits = cache?.hits ?? 0
            let t0 = Date()
            let text = try real.grey(audio, context, prefix, inner, cache, onText)
            let took = ms(since: t0)
            record.passes.runs.append(Run(kind: "grey", audio: audio.count, sum: audio.reduce(0, +), context: context,
                                          prefix: prefix, text: text))
            let note: String
            if let inner, inner.reuses > reuses {
                note = " (first 8 s reused)"
                record.times["grey or last reusing an agreement run's first 8 s", default: []].append(took)
                checkReuse(text) { try real.grey(audio, context, prefix, nil, nil, { _ in }) }
            } else if let checked, checked.standsIn(for: audio, context: context) {
                note = " (first 8 s's encoder output reused)"
                record.times["grey or last reusing only an agreement run's encoder output", default: []].append(took)
                record.encoderOnly += 1
                checkReuse(text) { try real.grey(audio, context, prefix, nil, nil, { _ in }) }
            } else if let cache, cache.hits > hits {
                note = " (cached prompt)"
                record.times["grey (cached prompt)", default: []].append(took)
                record.checks.append {
                    record.cacheChecks += 1
                    let t1 = Date()
                    do {
                        let plain = try real.grey(audio, context, prefix, nil, nil, { _ in })
                        record.times["  the same without the cache", default: []].append(ms(since: t1))
                        guard plain != text else { return "" }
                        record.cacheMismatches += 1
                        return "\n    CACHE MISMATCH: without the cache \"\(plain)\""
                    } catch {
                        record.cacheMismatches += 1
                        return "\n    CACHE MISMATCH: the run without the cache failed: \(error)"
                    }
                }
            } else {
                note = ""
                record.times["grey, nothing reused", default: []].append(took)
            }
            record.last = "grey, \(seconds(audio)) s, \(context.count) chars of context, from \"\(prefix)\", \(took) ms\(note): \"\(text)\"\n    \(Worker.memory())"
            return text
        },
        agreement: { audio, context in
            Checked(model.agreementRun(audio: audio, context: context, maxTokens: Worker.maxTokens), audio: audio, finish: { c in
                let text = c.run.text
                record.passes.runs.append(Run(kind: "agree", audio: audio.count, sum: audio.reduce(0, +), context: context,
                                              prefix: "", text: text))
                record.times["agreement run, all steps", default: []].append(c.total)
                record.times["agreement run, longest step", default: []].append(c.longest)
                record.last = "agreement, \(seconds(audio)) s, \(context.count) chars of context, \(c.run.steps) steps in \(c.total) ms (longest \(c.longest) ms): \"\(text)\"\n    \(Worker.memory())"
                record.checks.append {
                    var options = Qwen3DecodingOptions()
                    options.maxTokens = Worker.maxTokens
                    options.context = context
                    options.blockwise = true
                    record.agreeChecks += 1
                    let t1 = Date()
                    do {
                        let plain = try model.transcribeThrowing(audio: audio, sampleRate: Worker.rate, options: options)
                        record.times["  the same in one call", default: []].append(ms(since: t1))
                        guard plain != text else { return "" }
                        record.agreeMismatches += 1
                        return "\n    AGREEMENT MISMATCH: in one call \"\(plain)\""
                    } catch {
                        record.agreeMismatches += 1
                        return "\n    AGREEMENT MISMATCH: the call in one go failed: \(error)"
                    }
                }
            }, fail: { error in
                print("AGREEMENT RUN FAILED: \(error)")
                record.failed = true
            })
        },
        last: { audio, context, base, onText in
            let checked = base as? Checked
            let inner = checked?.run
            let reuses = inner?.reuses ?? 0
            let t0 = Date()
            let text = try real.last(audio, context, inner, onText)
            let took = ms(since: t0)
            record.passes.runs.append(Run(kind: "last", audio: audio.count, sum: audio.reduce(0, +), context: context,
                                          prefix: "", text: text))
            record.times["last (after Stop)", default: []].append(took)
            var note = ""
            if let inner, inner.reuses > reuses {
                note = " (first 8 s reused)"
                record.times["grey or last reusing an agreement run's first 8 s", default: []].append(took)
                checkReuse(text) { try real.last(audio, context, nil, { _ in }) }
            } else if let checked, checked.standsIn(for: audio, context: context) {
                note = " (first 8 s's encoder output reused)"
                record.times["grey or last reusing only an agreement run's encoder output", default: []].append(took)
                record.encoderOnly += 1
                checkReuse(text) { try real.last(audio, context, nil, { _ in }) }
            }
            record.last = "last (after Stop), \(seconds(audio)) s, \(context.count) chars of context, \(took) ms\(note): \"\(text)\"\n    \(Worker.memory())"
            return text
        },
        trim: { text in
            let trimmed = real.trim(text)
            record.passes.trims[text] = trimmed
            return trimmed
        }), convert: { $0 })
    worker.onUpdate = { record.lines = $0 }
    // The loop counts a run once its model call is done and before it starts the next one, so the checks run here, on
    // a free model, outside every call the loop times.
    worker.onPass = { n in
        let notes = record.check()
        print("run \(n): \(record.last)\(notes)")
        for l in record.lines { print("  \(l.final ? "■" : "□") \(l.settled)⟨\(l.tail)⟩") }
    }
    worker.onFail = { print("FAILED: \($0)"); record.failed = true }

    worker.begin()
    await worker.play(samples, pace: false)
    worker.end()            // the Stop button: one last run, everything black
    await worker.idle()
    let rest = record.check()   // none should be left: every call ends with a counted run
    if !rest.isEmpty { print("checks after the clip:\(rest)") }

    let out = JSONEncoder()
    out.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes, .sortedKeys]
    try out.encode(record.passes).write(to: URL(fileURLWithPath: args[k + 1]))
    print("\(record.passes.runs.count) calls, \(record.passes.trims.count) trims")
    print("FINAL TRANSCRIPT:")
    for l in record.lines { print("  \(l.settled)\(l.tail)") }
    print("reuse check: \(record.reuseChecks) runs that reused an agreement run's first 8 s (\(record.encoderOnly) only its encoder output) run again without it, \(record.reuseMismatches) differ")
    print("agreement check: \(record.agreeChecks) agreement runs run again in one call, \(record.agreeMismatches) differ")
    print("prompt cache check: \(record.cacheChecks) cached grey runs run again without the cache, \(record.cacheMismatches) differ")
    for (k, v) in record.times { totals[k, default: []] += v }
    reuseChecks += record.reuseChecks
    reuseMismatches += record.reuseMismatches
    agreeChecks += record.agreeChecks
    agreeMismatches += record.agreeMismatches
    cacheChecks += record.cacheChecks
    cacheMismatches += record.cacheMismatches
    // A clip that reaches the first window's end has agreement runs to check; one longer than a window and a step, runs
    // that reuse a finished window's first 8 s.
    let heard = samples.count + Worker.rate
    if heard >= Worker.step, record.agreeChecks == 0 { print("NO AGREEMENT RUN CHECKED"); failed = true }
    if heard > Worker.window + Worker.step, record.reuseChecks == 0 { print("NO REUSE CHECKED"); failed = true }
    encoderOnly += record.encoderOnly
    if record.failed { failed = true }
}

print("=== timings on this Mac (ms)")
for (kind, v) in totals.sorted(by: { $0.key < $1.key }) {
    let s = v.sorted()
    print("\(kind): \(v.count) calls, mean \(v.reduce(0, +) / max(1, v.count)), median \(s[s.count / 2]), max \(s.last ?? 0)")
}
print("BLOCK REUSE: \(reuseChecks) checked (\(encoderOnly) only the encoder output), \(reuseMismatches) mismatches")
print("AGREEMENT RUNS: \(agreeChecks) checked, \(agreeMismatches) mismatches")
print("PROMPT CACHE: \(cacheChecks) checked, \(cacheMismatches) mismatches")
print("MEMORY at the end: \(Worker.memory())")
_ = cacheCap
if reuseMismatches > 0 || agreeMismatches > 0 || cacheMismatches > 0 || failed { exit(1) }
