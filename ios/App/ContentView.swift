import SwiftUI

struct ContentView: View {
    @EnvironmentObject var t: Transcriber

    var body: some View {
        NavigationStack {
            ScrollView {
                transcript
                    .font(.title2)
                    .lineSpacing(6)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .accessibilityIdentifier("transcript")
                    .padding()
            }
            .defaultScrollAnchor(.bottom, for: .sizeChanges)   // follow new text
            .overlay {
                if t.lines.isEmpty {
                    ContentUnavailableView("未有字幕", systemImage: "captions.bubble",
                                           description: Text("撳個咪，講廣東話，字幕會即時出現。全部喺部手機度轉，唔使上網。"))
                }
            }
            .navigationTitle("廣東話字幕")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("關於", systemImage: "info.circle") { showLicences = true }
                        .accessibilityIdentifier("about")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("清除", systemImage: "trash") { t.clear() }
                        .disabled(t.lines.isEmpty)
                        .accessibilityIdentifier("clear")
                }
            }
            .bottomBar { controls }
            .sheet(isPresented: $showLicences) { LicencesView() }
            .sheet(isPresented: $showWelcome, onDismiss: { welcomed = true }) { WelcomeView() }
        }
        .onAppear { if !welcomed { showWelcome = true } }
    }

    @State private var showLicences = false
    @State private var showWelcome = false
    @AppStorage("welcomed") private var welcomed = false

    /// Finished lines in the primary colour, the line still being spoken in grey.
    private var transcript: Text {
        t.lines.enumerated().reduce(Text("")) { text, e in
            text + Text((e.offset > 0 ? "\n" : "") + e.element.text)
                .foregroundStyle(e.element.final ? Color.primary : Color.secondary)
        }
    }

    private var controls: some View {
        VStack(spacing: 8) {
            Text(statusText)
                .font(.footnote)
                .foregroundStyle(isError ? Color.red : Color.secondary)
                .accessibilityIdentifier("status")
            Button { t.isRunning ? t.stop() : t.start() } label: {
                Group {
                    if isLoading { ProgressView().controlSize(.large).tint(.white) }
                    else { Image(systemName: t.isRunning ? "stop.fill" : "mic.fill").font(.title) }
                }
                .frame(width: 64, height: 64)
            }
            .prominentGlass()
            .buttonBorderShape(.circle)
            .tint(t.isRunning ? .red : .accentColor)
            .accessibilityLabel(isLoading ? "取消" : t.isRunning ? "停止" : "開始")
            .accessibilityIdentifier("mic")
            .sensoryFeedback(.impact, trigger: t.isRunning)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
    }

    private var isLoading: Bool { if case .loading = t.state { return true } else { return false } }
    private var isError: Bool { if case .failed = t.state { return true } else { return false } }
    private var statusText: String {
        switch t.state {
        case .idle: return " "
        case .loading(let m): return m
        case .listening: return "聽緊"
        case .failed(let m): return m
        }
    }
}

private extension View {
    /// Liquid Glass on iOS 26 (a bar with the scroll edge effect, a glass button); the classic look below it.
    @ViewBuilder func bottomBar(@ViewBuilder _ content: () -> some View) -> some View {
        let bar = content()
        if #available(iOS 26, *) { safeAreaBar(edge: .bottom) { bar } }
        else { safeAreaInset(edge: .bottom) { bar.background(.bar) } }
    }

    @ViewBuilder func prominentGlass() -> some View {
        if #available(iOS 26, *) { buttonStyle(.glassProminent) } else { buttonStyle(.borderedProminent) }
    }
}
