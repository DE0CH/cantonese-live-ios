import SwiftUI

/// The sheet shown on the first start: what the app does and that it all runs on the phone.
struct WelcomeView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    Text("歡迎使用廣東話字幕")
                        .font(.largeTitle.bold())
                        .frame(maxWidth: .infinity)
                        .multilineTextAlignment(.center)
                        .padding(.top, 48)
                        .padding(.bottom, 8)
                    row("captions.bubble", "即時字幕", "講緊嗰句先用灰色顯示，講完就定稿。")
                    row("character.bubble", "廣東話、英文、普通話", "混埋講都認得，用香港繁體字寫出嚟。")
                    row("iphone", "全部喺手機度運行", "語音模型喺 iPhone 本地運行，唔使上網。")
                    row("hand.raised", "唔收集任何資料", "聲音同文字都唔會離開部手機。")
                }
                .padding(.horizontal, 32)
            }
            Button { dismiss() } label: {
                Text("繼續").font(.headline).frame(maxWidth: .infinity).padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .accessibilityIdentifier("continue")
            .padding(.horizontal, 32)
            .padding(.bottom, 24)
        }
        .interactiveDismissDisabled()
    }

    private func row(_ symbol: String, _ title: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: symbol)
                .font(.title)
                .foregroundStyle(.tint)
                .frame(width: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text(text).foregroundStyle(.secondary)
            }
        }
    }
}
