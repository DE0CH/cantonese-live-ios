import SwiftUI

struct ContentView: View {
    @EnvironmentObject var t: Transcriber
    private let accent = Color(red: 0.76, green: 0.25, blue: 0.05)

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("粵語即時字幕").font(.headline)
                Spacer()
                Picker("引擎", selection: $t.provider) {
                    ForEach(Transcriber.Provider.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 190)
                .accessibilityIdentifier("provider")
            }
            .padding(.horizontal, 16).padding(.vertical, 12)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading) {
                        if t.committed.isEmpty && t.pending.isEmpty {
                            Text("撳「開始」，對住咪講廣東話，字幕會即時出現。")
                                .font(.body).foregroundStyle(.secondary)
                        }
                        (Text(t.committed) + Text(t.pending).foregroundColor(.secondary))
                            .font(.system(size: 26))
                            .lineSpacing(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                            .accessibilityIdentifier("transcript")
                        Color.clear.frame(height: 1).id("end")
                    }
                    .padding(16)
                }
                .onChange(of: t.committed + t.pending) { _, _ in
                    withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo("end", anchor: .bottom) }
                }
            }
            VStack(spacing: 10) {
                Text(statusText)
                    .font(.footnote)
                    .foregroundStyle(isError ? accent : .secondary)
                    .accessibilityIdentifier("status")
                HStack(spacing: 20) {
                    Button("清除") { t.clear() }
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("clear")
                    Button(t.isRunning ? "停止" : "開始") { t.isRunning ? t.stop() : t.start() }
                        .font(.headline)
                        .frame(minWidth: 150).padding(.vertical, 14)
                        .background(t.isRunning ? Color.primary : accent, in: Capsule())
                        .foregroundStyle(t.isRunning ? Color(uiColor: .systemBackground) : .white)
                        .accessibilityIdentifier("mic")
                }
            }
            .padding(.vertical, 12)
        }
    }

    private var isError: Bool { if case .failed = t.state { return true } else { return false } }
    private var statusText: String {
        switch t.state {
        case .idle: return " "
        case .connecting: return "連接緊…"
        case .listening: return "● 聽緊"
        case .failed(let m): return m
        }
    }
}
