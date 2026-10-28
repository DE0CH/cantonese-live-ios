import XCTest

/// The phone replica: walks every flow on the simulator, streaming the bundled Cantonese clip instead of the mic. The
/// simulator can't run MLX, so a stand-in answers for the model (Transcriber.swift); line cuts, OpenCC and the UI
/// are real.
final class CantoneseUITests: XCTestCase {
    func testEveryFlow() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CANTONESE_TEST_AUDIO"] = "1"
        app.launchArguments = ["-welcomed", "NO"]   // a first start, whatever ran before
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
        while Date() < ready, status(app) != "聽緊" {
            if status(app).hasPrefix("用唔到") { break }
            Thread.sleep(forTimeInterval: 1)
        }
        XCTAssertEqual(status(app), "聽緊", "never started listening (\(status(app)))")
        shot(app, "listening")

        // The 7.5 s clip is cut into lines at its quietest point near 6 s; the first lands as a finished line.
        let transcript = app.staticTexts["transcript"]
        let deadline = Date().addingTimeInterval(60)
        while Date() < deadline, !(transcript.exists && transcript.label.contains("我哋")) {
            Thread.sleep(forTimeInterval: 2)
        }
        shot(app, "transcribed")
        let text = transcript.exists ? transcript.label : ""
        print("TRANSCRIPT: \(text)")
        XCTAssertTrue(text.contains("我哋"), "lost the spoken form: \(text) (\(status(app)))")
        XCTAssertTrue(text.contains("飲茶"), "not HK Traditional: \(text)")

        app.buttons["mic"].tap()                  // 停止
        XCTAssertTrue(app.buttons["開始"].waitForExistence(timeout: 10), "stop didn't stop")
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
        app.launchArguments = []
        app.launch()
        XCTAssertTrue(app.buttons["mic"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.buttons["continue"].exists, "welcome sheet shown again")
    }

    /// The App Store demo: a first start, then the demo conversation (testdata/demo) captioned live. The appstore workflow
    /// makes the store screenshots from its shots (tools/storeart).
    func testDemo() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CANTONESE_TEST_AUDIO"] = "demo"
        app.launchArguments = ["-welcomed", "NO"]
        app.launch()
        let go = app.buttons["continue"]
        XCTAssertTrue(go.waitForExistence(timeout: 10), "no welcome sheet")
        Thread.sleep(forTimeInterval: 3)
        shot(app, "demo-welcome")
        go.tap()
        Thread.sleep(forTimeInterval: 1)
        app.buttons["mic"].tap()
        let ready = Date().addingTimeInterval(120)
        while Date() < ready, status(app) != "聽緊" { Thread.sleep(forTimeInterval: 0.2) }
        XCTAssertEqual(status(app), "聽緊")
        // The conversation is 19.2 s, then a second of silence; the last line is finished soon after.
        Thread.sleep(forTimeInterval: 11)
        shot(app, "demo-middle")
        Thread.sleep(forTimeInterval: 11)
        shot(app, "demo-captions")
        Thread.sleep(forTimeInterval: 3)
        let text = app.staticTexts["transcript"].label
        print("DEMO TRANSCRIPT: \(text)")
        XCTAssertTrue(text.contains("Central") && text.contains("MTR"), "demo captions missing: \(text)")
        app.buttons["mic"].tap()
    }

    private func status(_ app: XCUIApplication) -> String { app.staticTexts["status"].label }

    private func shot(_ app: XCUIApplication, _ name: String) {
        let a = XCTAttachment(screenshot: app.screenshot())
        a.name = name
        a.lifetime = .keepAlways
        add(a)
    }
}
