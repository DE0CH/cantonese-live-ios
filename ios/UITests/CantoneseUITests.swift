import XCTest

/// The phone replica: walks every flow on the simulator, streaming the bundled Cantonese clip instead of the mic. The
/// simulator can't run MLX, so a stand-in answers for the models (Transcriber.swift); the loop, OpenCC and the UI
/// are real.
final class CantoneseUITests: XCTestCase {
    func testEveryFlow() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CANTONESE_TEST_AUDIO"] = "1"
        app.launchArguments = ["-welcomed", "NO"] + Self.cantonese   // a first start, whatever ran before
        app.launch()

        // First start: the welcome sheet, once.
        let go = app.buttons["continue"]
        XCTAssertTrue(go.waitForExistence(timeout: 10), "no welcome sheet on the first start")
        shot(app, "welcome")
        go.tap()
        XCTAssertTrue(app.buttons["mic"].waitForExistence(timeout: 10), "welcome sheet didn't close")
        shot(app, "launch")

        XCTAssertFalse(app.buttons["clear"].isEnabled, "clear is on with nothing to clear")
        app.buttons["mic"].tap()
        let ready = Date().addingTimeInterval(3 * 60)
        while Date() < ready, state(app) != "listening" {
            if state(app) == "failed" { break }
            Thread.sleep(forTimeInterval: 1)
        }
        XCTAssertEqual(status(app), "聽緊", "never started listening (\(status(app)))")
        shot(app, "listening")

        // The 7.5 s clip (and a second of silence) goes through the sliding window: grey while it grows, one slide at 8 s.
        let transcript = app.staticTexts["transcript"]
        XCTAssertTrue(wait(60) { transcript.exists && transcript.label.contains("我哋") }, "no captions (\(status(app)))")
        shot(app, "transcribed")
        XCTAssertTrue(wait(60) { transcript.value as? String == "played" }, "the clip didn't finish playing")

        app.buttons["mic"].tap()                  // 停止: one last run, everything black
        XCTAssertTrue(app.buttons["開始"].waitForExistence(timeout: 10), "stop didn't stop")
        Thread.sleep(forTimeInterval: 1)
        let text = transcript.exists ? transcript.label : ""
        print("TRANSCRIPT: \(text)")
        XCTAssertTrue(text.contains("我哋"), "lost the spoken form: \(text) (\(status(app)))")
        XCTAssertTrue(text.contains("飲茶"), "not HK Traditional: \(text)")
        app.buttons["clear"].tap()
        let cleared = app.staticTexts["transcript"]
        XCTAssertTrue(!cleared.exists || cleared.label.trimmingCharacters(in: .whitespaces).isEmpty, "clear left text behind")
        shot(app, "cleared")

        // 關於: every licence the app ships with opens with its text.
        app.buttons["about"].tap()
        let qwen = app.buttons["licence-Qwen3-ASR 1.7B"]
        XCTAssertTrue(qwen.waitForExistence(timeout: 10), "about sheet didn't open")
        shot(app, "about")
        qwen.tap()
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label CONTAINS 'Apache License'")).firstMatch
                        .waitForExistence(timeout: 10), "licence text missing")
        app.navigationBars.buttons.element(boundBy: 0).tap()
        app.buttons["done"].tap()
        XCTAssertTrue(app.buttons["mic"].waitForExistence(timeout: 10), "about sheet didn't close")

        // A second start reuses the loaded models: listening at once.
        app.buttons["mic"].tap()
        XCTAssertTrue(app.staticTexts["聽緊"].waitForExistence(timeout: 60), "second start was slow (\(status(app)))")
        app.buttons["mic"].tap()

        // The welcome sheet doesn't come back.
        app.terminate()
        app.launchArguments = Self.cantonese
        app.launch()
        XCTAssertTrue(app.buttons["mic"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.buttons["continue"].exists, "welcome sheet shown again")
    }

    /// A phone in Hong Kong Chinese gets 廣東話, in Taiwan Chinese 書面語, in Simplified Chinese 简体, otherwise English.
    func testLanguages() throws {
        for (language, empty) in [("zh-Hant-HK", "未有字幕"), ("zh-Hant-TW", "尚無字幕"), ("zh-Hans-CN", "暂无字幕"), ("en-HK", "No captions yet")] {
            let app = XCUIApplication()
            app.launchArguments = ["-welcomed", "YES", "-AppleLanguages", "(\(language))"]
            app.launch()
            XCTAssertTrue(app.staticTexts[empty].waitForExistence(timeout: 10), "\(language): no 「\(empty)」")
            shot(app, "language-\(language)")
            app.terminate()
        }
    }

    /// The App Store demo, once per store language (the app's 廣東話, 简体 and English): a first start, then the demo
    /// conversation (testdata/demo) captioned live with the real model's answers (the stand-in replays every run
    /// the app's loop made over it on a Mac, tools/demo-passes). The appstore workflow makes the store
    /// screenshots from its shots (tools/storeart): the loop stops after two runs (CANTONESE_HOLD), one while only
    /// Cantonese has been said, one once English is mixed in, each showing grey text.
    static let holds = (chinese: 108, mixed: 274)   // the runs in testdata/demo/demo-16k.passes.json
    static let cantonese = ["-AppleLanguages", "(zh-Hant-HK)"]
    func testDemo() throws {
        for (name, language) in [("zh-HK", "zh-Hant-HK"), ("zh-Hans", "zh-Hans-CN"), ("en", "en-HK")] {
            try demo(name, language)
        }
    }

    private func demo(_ name: String, _ language: String) throws {
        let app = XCUIApplication()
        app.launchEnvironment["CANTONESE_TEST_AUDIO"] = "demo"
        app.launchEnvironment["CANTONESE_HOLD"] = "\(Self.holds.chinese),\(Self.holds.mixed)"
        app.launchArguments = ["-welcomed", "NO", "-AppleLanguages", "(\(language))"]
        app.launch()
        let go = app.buttons["continue"]
        XCTAssertTrue(go.waitForExistence(timeout: 10), "no welcome sheet")
        Thread.sleep(forTimeInterval: 3)
        shot(app, "demo-welcome-\(name)")
        go.tap()
        Thread.sleep(forTimeInterval: 1)
        app.buttons["mic"].tap()
        XCTAssertTrue(wait(120) { state(app) == "listening" }, "never started listening (\(status(app)))")
        let transcript = app.staticTexts["transcript"]
        for shotName in ["demo-chinese", "demo-mixed"] {
            XCTAssertTrue(wait(60) { transcript.exists && transcript.value as? String == "held" }, "the loop didn't stop for \(shotName)")
            Thread.sleep(forTimeInterval: 0.5)   // the screen takes the newest lines every 0.1 s
            shot(app, "\(shotName)-\(name)")
            print("\(shotName)-\(name): \(transcript.label)")
            XCTAssertTrue(wait(30) { !transcript.exists || transcript.value as? String != "held" })
        }
        // The rest of the conversation and a second of silence, then Stop: one last run, everything black.
        XCTAssertTrue(wait(120) { transcript.value as? String == "played" }, "the demo didn't finish playing")
        app.buttons["mic"].tap()
        Thread.sleep(forTimeInterval: 2)
        let text = transcript.label
        print("DEMO TRANSCRIPT \(name): \(text)")
        for words in ["Central", "proposal", "send"] { XCTAssertTrue(text.contains(words), "demo captions missing \(words): \(text)") }
        app.terminate()
    }

    private func wait(_ seconds: TimeInterval, until: () -> Bool) -> Bool {
        let end = Date().addingTimeInterval(seconds)
        while Date() < end {
            if until() { return true }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return until()
    }

    private func status(_ app: XCUIApplication) -> String { app.staticTexts["status"].label }
    private func state(_ app: XCUIApplication) -> String { app.staticTexts["status"].value as? String ?? "" }

    private func shot(_ app: XCUIApplication, _ name: String) {
        let a = XCTAttachment(screenshot: app.screenshot())
        a.name = name
        a.lifetime = .keepAlways
        add(a)
    }
}
