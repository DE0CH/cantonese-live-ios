import Foundation
import SherpaOnnxShared   // sherpa-onnx's dynamic frameworks (the static ones get embedded unsigned and fail the export)

/// The models the app ships with, in Model/ (ios/fetch-model.sh), both through sherpa-onnx on the CPU:
///   silero_vad.onnx   Silero speech detection
///   sensevoice/       SenseVoice-Small int8: the text
extension Worker.Models {
    static func load(_ dir: URL) -> Worker.Models {
        let sense = SenseVoice(dir: dir.appendingPathComponent("sensevoice"))
        return Worker.Models(segmenter: SileroSegmenter(model: dir.appendingPathComponent("silero_vad.onnx").path),
                             transcribe: sense.transcribe)
    }
}

/// Silero through sherpa-onnx: a stretch ends after 0.5 s of silence, or at 15 s.
final class SileroSegmenter: Segmenter {
    private let vad: SherpaOnnxVoiceActivityDetectorWrapper

    init(model: String) {
        var config = sherpaOnnxVadModelConfig(
            sileroVad: sherpaOnnxSileroVadModelConfig(model: model, threshold: 0.5, minSilenceDuration: 0.5,
                                                      minSpeechDuration: 0.25, windowSize: 512, maxSpeechDuration: 15),
            sampleRate: Int32(Worker.rate))
        vad = SherpaOnnxVoiceActivityDetectorWrapper(config: &config, buffer_size_in_seconds: 60)
    }

    func feed(_ audio: [Float]) -> [(start: Int, audio: [Float])] {
        vad.acceptWaveform(samples: audio)
        return drain()
    }

    func flush() -> [(start: Int, audio: [Float])] {
        vad.flush()
        return drain()
    }

    var speaking: Bool { vad.isSpeechDetected() }

    func reset() { vad.reset() }

    private func drain() -> [(start: Int, audio: [Float])] {
        var out: [(start: Int, audio: [Float])] = []
        while !vad.isEmpty() {
            let s = vad.front()
            out.append((s.start, s.samples))
            vad.pop()
        }
        return out
    }
}

/// SenseVoice-Small through sherpa-onnx, told the speech is Cantonese, with its punctuation and numbers (ITN).
final class SenseVoice: @unchecked Sendable {
    private let recognizer: SherpaOnnxOfflineRecognizer

    init(dir: URL) {
        let model = sherpaOnnxOfflineModelConfig(
            tokens: dir.appendingPathComponent("tokens.txt").path, numThreads: 2,
            senseVoice: sherpaOnnxOfflineSenseVoiceModelConfig(
                model: dir.appendingPathComponent("model.int8.onnx").path, language: "yue",
                useInverseTextNormalization: true))
        var config = sherpaOnnxOfflineRecognizerConfig(featConfig: sherpaOnnxFeatureConfig(), modelConfig: model)
        recognizer = SherpaOnnxOfflineRecognizer(config: &config)
    }

    func transcribe(_ audio: [Float]) -> String {
        recognizer.decode(samples: audio, sampleRate: Worker.rate).text
    }
}
