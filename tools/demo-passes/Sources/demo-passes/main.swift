// demo-passes MODEL_DIR CLIP.wav OUT.json [CLIP.wav OUT.json ...]
//   MODEL_DIR: the app's model folder (ios/Model from ios/fetch-model.sh)
//   OUT.json:  {runs: [{audio, sum, context, text}]}: every black run the app's loop made over the clip (the stretch's
//              length and sum, the context it was given, Qwen3-ASR's answer), for the simulator's stand-in
//              (testdata/…passes.json). The clip is played as the UI tests play it (then a second of silence, then Stop).
// Prints each clip's final lines, and how long the grey runs (SenseVoice, CPU) and black runs (Qwen3-ASR, GPU) took in
// all against the clip's length.
import Foundation
import OpenCC

struct Run: Codable { let audio: Int; let sum: Float; let context: String; let text: String }
struct Passes: Codable { var runs: [Run] = [] }

let args = CommandLine.arguments
guard args.count >= 4, args.count % 2 == 0 else {
    FileHandle.standardError.write("usage: demo-passes MODEL_DIR CLIP.wav OUT.json [CLIP.wav OUT.json ...]\n".data(using: .utf8)!)
    exit(2)
}

final class Tally: @unchecked Sendable {
    let lock = NSLock()
    var greyMs = 0, greyRuns = 0, blackMs = 0, blackRuns = 0
    var runs: [Run] = []
    var lines: [Worker.Line] = []
    func add(_ f: (Tally) -> Void) { lock.lock(); f(self); lock.unlock() }
}

let loaded = try await Worker.Models.load(URL(fileURLWithPath: args[1]))
let converter = try ChineseConverter(options: [.traditionalize, .hkStandard])
let convertLock = NSLock()
var failed = false

for i in stride(from: 2, to: args.count, by: 2) {
    guard let samples = Worker.samples(wav: try Data(contentsOf: URL(fileURLWithPath: args[i]))) else {
        print("\(args[i]): not a 16-bit WAV"); exit(1)
    }
    let tally = Tally()
    let models = Worker.Models(
        segmenter: loaded.segmenter,
        grey: { audio in
            let t0 = Date()
            let text = loaded.grey(audio)
            tally.add { $0.greyMs += Int(Date().timeIntervalSince(t0) * 1000); $0.greyRuns += 1 }
            return text
        },
        black: { audio, context in
            let t0 = Date()
            let text = try loaded.black(audio, context)
            tally.add { $0.blackMs += Int(Date().timeIntervalSince(t0) * 1000); $0.blackRuns += 1 }
            return text
        })
    let worker = Worker(models: models, convert: { text in
        convertLock.lock(); defer { convertLock.unlock() }
        return converter.convert(text)
    })
    worker.onBlack = { audio, context, text in
        tally.add { $0.runs.append(Run(audio: audio.count, sum: audio.reduce(0, +), context: context, text: text)) }
    }
    worker.configure(onUpdate: { lines in tally.add { $0.lines = lines } })
    worker.begin()
    await worker.play(samples, pace: false)
    worker.end()
    await worker.settle()

    let seconds = Double(samples.count) / Double(Worker.rate)
    print("== \(args[i]) (\(String(format: "%.1f", seconds)) s)")
    for r in tally.runs { print("black \(String(format: "%5.2f", Double(r.audio) / Double(Worker.rate))) s: \(r.text)") }
    print("FINAL TRANSCRIPT:")
    for l in tally.lines { print("  \(l.settled)\(l.tail.isEmpty ? "" : " [grey: \(l.tail)]")") }
    print(String(format: "TIMES: grey %d runs %d ms, black %d runs %d ms (GPU busy %.0f%% of the clip)",
                 tally.greyRuns, tally.greyMs, tally.blackRuns, tally.blackMs, Double(tally.blackMs) / 10 / seconds))
    if tally.runs.isEmpty || tally.lines.contains(where: { !$0.final }) { print("::error::\(args[i]): no black runs, or grey left"); failed = true }
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try enc.encode(Passes(runs: tally.runs)).write(to: URL(fileURLWithPath: args[i + 1]))
}
exit(failed ? 1 : 0)
