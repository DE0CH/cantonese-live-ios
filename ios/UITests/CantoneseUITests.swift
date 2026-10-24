import XCTest

/// The phone replica: walks every flow against the live relay, streaming the bundled Cantonese clip instead of the mic.
final class CantoneseUITests: XCTestCase {
    func testEveryFlow() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CANTONESE_TEST_AUDIO"] = "1"
        app.launch()
        shot(app, "launch")

        app.buttons["clear"].tap()
        app.buttons["mic"].tap()
        XCTAssertTrue(app.staticTexts["● 聽緊"].waitForExistence(timeout: 20), "never started listening (\(status(app)))")
        let transcript = app.staticTexts["transcript"]
        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline, transcript.label.unicodeScalars.filter({ (0x4E00...0x9FFF).contains($0.value) }).count < 6 {
            Thread.sleep(forTimeInterval: 0.5)
        }
        shot(app, "live")
        Thread.sleep(forTimeInterval: 6)          // the rest of the 7.5 s clip
        app.buttons["mic"].tap()                  // 停止
        Thread.sleep(forTimeInterval: 3)
        shot(app, "stopped")
        let text = transcript.label
        print("TRANSCRIPT: \(text)")
        XCTAssertTrue(text.contains("我哋"), "lost the spoken form: \(text)")
        XCTAssertGreaterThanOrEqual(text.unicodeScalars.filter { (0x4E00...0x9FFF).contains($0.value) }.count, 6,
                                    "no Cantonese text (\(status(app)))")
        app.buttons["clear"].tap()
        let cleared = app.staticTexts["transcript"]
        XCTAssertTrue(!cleared.exists || cleared.label.trimmingCharacters(in: .whitespaces).isEmpty, "clear left text behind")
        shot(app, "cleared")
    }

    /// The network goes down for 3 s, 3 s into the clip: nothing shows it, and the transcript has every word exactly once
    /// (the backlog is over 1.5 s, so this also walks the split into a catch-up and a live segment).
    func testDropIsInvisible() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CANTONESE_TEST_AUDIO"] = "1"
        app.launchEnvironment["CANTONESE_TEST_DROP"] = "3"
        app.launch()
        app.buttons["clear"].tap()
        app.buttons["mic"].tap()
        XCTAssertTrue(app.staticTexts["● 聽緊"].waitForExistence(timeout: 20), "never started listening (\(status(app)))")
        var statuses: Set<String> = []
        let deadline = Date().addingTimeInterval(14)   // the 7.5 s clip, the outage and the catch-up
        while Date() < deadline {
            statuses.insert(status(app))
            XCTAssertEqual(app.buttons["mic"].label, "停止", "the drop stopped the session")
            Thread.sleep(forTimeInterval: 0.2)
        }
        shot(app, "after-drop")
        app.buttons["mic"].tap()
        Thread.sleep(forTimeInterval: 4)
        shot(app, "stopped-after-drop")
        let text = app.staticTexts["transcript"].label
        print("TRANSCRIPT (drop): \(text) STATUSES: \(statuses)")
        XCTAssertEqual(statuses, ["● 聽緊"], "the drop showed in the status")
        XCTAssertEqual(text.components(separatedBy: "飲茶").count - 1, 1, "lost or repeated words: \(text)")
        XCTAssertTrue(text.contains("聽日"), "lost the audio after the drop: \(text)")
    }

    private func status(_ app: XCUIApplication) -> String { app.staticTexts["status"].label }

    private func shot(_ app: XCUIApplication, _ name: String) {
        let a = XCTAttachment(screenshot: app.screenshot())
        a.name = name
        a.lifetime = .keepAlways
        add(a)
    }
}
