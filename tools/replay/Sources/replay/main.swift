// replay MODEL_DIR CLIP.wav OUT.json
//   MODEL_DIR: the app's model folder (ios/Model/asr from ios/fetch-model.sh)
//   OUT.json:  [{audio, sum, start, text}], one per model job: the audio's length and sum, the text it started from and
//              the model's answer (testdata/demo/demo-16k.passes.json for the demo)
import Foundation
import Qwen3ASR

let args = CommandLine.arguments
guard args.count == 4, let wav = FileManager.default.contents(atPath: args[2]), let samples = Worker.samples(wav: wav) else {
    FileHandle.standardError.write("usage: replay MODEL_DIR CLIP.wav OUT.json\n".data(using: .utf8)!)
    exit(2)
}

struct Pass: Encodable { let audio: Int; let sum: Float; let start: String; let text: String }
final class Record: @unchecked Sendable {
    var passes: [Pass] = []
    var lines: [Worker.Line] = []
}
let record = Record()

let model = try await Qwen3ASRModel.fromPretrained(
    modelId: Worker.modelId, cacheDir: URL(fileURLWithPath: args[1]), offlineMode: true)
let real = Worker.asr(model)
let worker = Worker(asr: { audio, start, onText in
    let text = try real(audio, start, onText)
    record.passes.append(Pass(audio: audio.count, sum: audio.reduce(0, +), start: start, text: text))
    return text
}, convert: { $0 })
worker.onUpdate = { record.lines = $0 }
worker.onPass = { n in
    if let p = record.passes.last { print("pass \(n): \(String(format: "%.1f", Double(p.audio) / 16000)) s from \"\(p.start)\"") }
    for l in record.lines { print("  \(l.final ? "■" : "□") \(l.settled)⟨\(l.tail)⟩") }
}

worker.begin()
await worker.play(samples, pace: false)
let out = JSONEncoder()
out.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
try out.encode(record.passes).write(to: URL(fileURLWithPath: args[3]))
print("\(record.passes.count) passes")
