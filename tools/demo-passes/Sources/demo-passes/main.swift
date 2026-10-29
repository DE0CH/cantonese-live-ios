// demo-passes MODEL_DIR CLIP.wav OUT.json [CLIP.wav OUT.json ...]
//   MODEL_DIR: the app's model folder (ios/Model from ios/fetch-model.sh: asr/)
//   OUT.json:  {runs: [{audio, sum, context, start, text}]}: every model run the app's loop made over the clip (the
//              audio's length and sum, what it was given, the answer), for the simulator's stand-in
//              (testdata/…passes.json). Stops the loop after the clip like the Stop button, so the last run is recorded too.
// Every run that reused the prompt cache is run again without it; the texts must be the same (exit 1 if not). Memory is
// printed after every run, with MLX's buffer cache capped as in the app.
import Foundation
import Qwen3ASR

let args = CommandLine.arguments
guard args.count >= 4, args.count % 2 == 0 else {
    FileHandle.standardError.write("usage: demo-passes MODEL_DIR CLIP.wav OUT.json [CLIP.wav OUT.json ...]\n".data(using: .utf8)!)
    exit(2)
}

struct Run: Codable { let audio: Int; let sum: Float; let context: String; let start: String; let text: String }
struct Passes: Codable { var runs: [Run] = [] }
final class Record: @unchecked Sendable {
    var passes = Passes()
    var lines: [Worker.Line] = []
    var last = ""                       // what the last run did, for the log
    var times: [String: [Int]] = [:]    // ms per kind of call
    var cacheChecks = 0
    var mismatches = 0
    var failed = false
}
func ms(since t0: Date) -> Int { Int(Date().timeIntervalSince(t0) * 1000) }

let dir = URL(fileURLWithPath: args[1])
let model = try await Qwen3ASRModel.fromPretrained(
    modelId: Worker.modelId, cacheDir: dir.appendingPathComponent("asr"), offlineMode: true)
let realASR = Worker.asr(model)
let cacheCap = Qwen3ASRMLXMemory.limitCache(to: Worker.cacheLimit)
print("model loaded, \(Worker.memory())")
var failed = false
var totals: [String: [Int]] = [:]
var checks = 0, mismatches = 0

for k in stride(from: 2, to: args.count, by: 2) {
    guard let wav = FileManager.default.contents(atPath: args[k]), let samples = Worker.samples(wav: wav) else {
        print("can't read \(args[k])"); exit(2)
    }
    print("=== \(args[k]): \(String(format: "%.1f", Double(samples.count) / 16000)) s")
    let record = Record()
    let worker = Worker(asr: { audio, context, start, cache, onText in
        let hits = cache?.hits ?? 0
        let t0 = Date()
        let text = try realASR(audio, context, start, cache, onText)
        let took = ms(since: t0)
        let sum = audio.reduce(0, +)
        record.passes.runs.append(Run(audio: audio.count, sum: sum, context: context, start: start, text: text))
        let kind = cache == nil ? "last (after Stop)" : (start.isEmpty && audio.count == Worker.window ? "slide" : "run")
        var note = ""
        if let cache, cache.hits > hits {
            record.times["run (cached prompt)", default: []].append(took)
            let t1 = Date()
            let plain = try realASR(audio, context, start, nil, { _ in })
            record.times["same run without the cache", default: []].append(ms(since: t1))
            record.cacheChecks += 1
            if plain != text {
                record.mismatches += 1
                note = "\n    CACHE MISMATCH: without the cache \"\(plain)\""
            }
            note = " (cached prompt)" + note
        } else {
            record.times[kind == "run" ? "run (filling the cache)" : kind, default: []].append(took)
        }
        record.last = "\(String(format: "%.2f", Double(audio.count) / 16000)) s, \(context.count) chars of context, from \"\(start)\", \(took) ms\(note): \"\(text)\"\n    \(Worker.memory())"
        return text
    }, convert: { $0 })
    worker.onUpdate = { record.lines = $0 }
    worker.onPass = { n in
        print("run \(n): \(record.last)")
        for l in record.lines { print("  \(l.final ? "■" : "□") \(l.settled)⟨\(l.tail)⟩") }
    }
    worker.onFail = { print("FAILED: \($0)"); record.failed = true }

    worker.begin()
    await worker.play(samples, pace: false)
    worker.end()            // the Stop button: one last run, everything black
    await worker.idle()

    let out = JSONEncoder()
    out.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes, .sortedKeys]
    try out.encode(record.passes).write(to: URL(fileURLWithPath: args[k + 1]))
    print("\(record.passes.runs.count) runs")
    print("FINAL TRANSCRIPT:")
    for l in record.lines { print("  \(l.settled)\(l.tail)") }
    print("prompt cache check: \(record.cacheChecks) cached runs run again without the cache, \(record.mismatches) differ")
    for (k, v) in record.times { totals[k, default: []] += v }
    checks += record.cacheChecks
    if record.failed { failed = true }
    mismatches += record.mismatches
}

print("=== timings on this Mac (ms)")
for (kind, v) in totals.sorted(by: { $0.key < $1.key }) {
    let s = v.sorted()
    print("\(kind): \(v.count) calls, mean \(v.reduce(0, +) / max(1, v.count)), median \(s[s.count / 2]), max \(s.last ?? 0)")
}
print("PROMPT CACHE: \(checks) checked, \(mismatches) mismatches")
print("MEMORY at the end: \(Worker.memory())")
_ = cacheCap
if mismatches > 0 || failed { exit(1) }
