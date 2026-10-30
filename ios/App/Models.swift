import Foundation
import SherpaOnnx
#if !targetEnvironment(simulator)
import MLX
import Qwen3ASR
#endif

/// The models the app ships with, in Model/ (ios/fetch-model.sh):
///   silero_vad.onnx   Silero speech detection (sherpa-onnx, CPU)
///   sensevoice/       SenseVoice-Small int8 (sherpa-onnx, CPU): the grey text
///   asr/              Qwen3-ASR 1.7B 5-bit (speech-swift on MLX, GPU): the black text
extension Worker.Models {
    static let qwenId = "aufklarer/Qwen3-ASR-1.7B-MLX-5bit"
    /// MLX keeps freed buffers for reuse; a stretch's arrays need well under this, and speech-swift's default (a quarter
    /// of the phone's memory, 2 GB) on top of the weights got the app killed.
    static let mlxCacheLimit = 256 * 1024 * 1024

    /// Silero and SenseVoice from `dir`, and Qwen3-ASR unless `black` stands in for it (the simulator has no Metal
    /// for MLX).
    static func load(_ dir: URL, black: ((_ audio: [Float], _ context: String) throws -> String)? = nil) async throws -> Worker.Models {
        let segmenter = SileroSegmenter(model: dir.appendingPathComponent("silero_vad.onnx").path)
        let sense = SenseVoice(dir: dir.appendingPathComponent("sensevoice"))
        if let black {
            return Worker.Models(segmenter: segmenter, grey: sense.transcribe, black: black)
        }
        #if targetEnvironment(simulator)
        throw CocoaError(.featureUnsupported)
        #else
        let qwen = try await Qwen3ASRModel.fromPretrained(
            modelId: qwenId, cacheDir: dir.appendingPathComponent("asr"), offlineMode: true)
        MLX.Memory.cacheLimit = mlxCacheLimit
        return Worker.Models(segmenter: segmenter, grey: sense.transcribe, black: { audio, context in
            var o = Qwen3DecodingOptions()
            o.context = context
            return try qwen.transcribeThrowing(audio: audio, sampleRate: Worker.rate, options: o)
        })
        #endif
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
