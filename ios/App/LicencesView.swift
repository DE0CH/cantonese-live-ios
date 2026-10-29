import SwiftUI

/// The open-source parts the app ships with (the models and the libraries around them) and their licence texts, which
/// live in Licences/ as bundle resources.
struct LicencesView: View {
    private struct Part: Identifiable {
        let name: String, use: String, licence: String, file: String
        var id: String { name }
    }

    private let parts = [
        Part(name: "Qwen3-ASR 1.7B", use: String(localized: "語音辨識模型（Qwen 團隊；MLX 5-bit 版 aufklarer）"), licence: "Apache 2.0", file: "Apache-2.0"),
        Part(name: "speech-swift", use: String(localized: "喺 iPhone 度行模型（soniqo）"), licence: "Apache 2.0", file: "Apache-2.0"),
        Part(name: "MLX Swift", use: String(localized: "GPU 運算（Apple）"), licence: "MIT", file: "MIT-mlx-swift"),
        Part(name: "MLX", use: String(localized: "GPU 運算（Apple）"), licence: "MIT", file: "MIT-mlx"),
        Part(name: "swift-transformers", use: String(localized: "模型檔案（Hugging Face）"), licence: "Apache 2.0", file: "Apache-2.0"),
        Part(name: "SwiftyOpenCC", use: String(localized: "轉香港繁體（DengXiang）"), licence: "MIT", file: "MIT-SwiftyOpenCC"),
        Part(name: "OpenCC", use: String(localized: "轉香港繁體（BYVoid）"), licence: "Apache 2.0", file: "Apache-2.0"),
        Part(name: "Qwen3-ForcedAligner 0.6B", use: String(localized: "字詞時間對齊模型（Qwen 團隊；MLX 5-bit 版 aufklarer）"), licence: "Apache 2.0", file: "Apache-2.0"),
    ]

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("全部喺部 iPhone 度轉，唔使上網：聲音同文字都唔會離開部手機。")
                }
                Section("開源元件") {
                    ForEach(parts) { p in
                        NavigationLink {
                            ScrollView {
                                Text(text(p.file)).font(.footnote.monospaced()).textSelection(.enabled).padding()
                            }
                            .navigationTitle(p.name)
                            .navigationBarTitleDisplayMode(.inline)
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(p.name)
                                Text("\(p.use) · \(p.licence)").font(.footnote).foregroundStyle(.secondary)
                            }
                        }
                        .accessibilityIdentifier("licence-\(p.name)")
                    }
                }
            }
            .navigationTitle("關於")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("完成") { dismiss() }.accessibilityIdentifier("done") }
        }
    }

    private func text(_ file: String) -> String {
        Bundle.main.url(forResource: file, withExtension: "txt").flatMap { try? String(contentsOf: $0, encoding: .utf8) } ?? ""
    }
}
