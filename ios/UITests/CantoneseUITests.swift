import XCTest

/// The phone replica: walks every flow against the live relay, streaming the bundled Cantonese clip instead of the mic.
final class CantoneseUITests: XCTestCase {
    func testEveryFlow() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CANTONESE_TEST_AUDIO"] = "1"
        app.launch()
        shot(app, "launch")

        // Spoken Cantonese on both engines, then 書面語 on both: the finished lines must turn into written Chinese.
        for (style, engine) in [("口語", "Soniox"), ("口語", "Fun-ASR"), ("書面語", "Soniox"), ("書面語", "Fun-ASR")] {
            app.segmentedControls["style"].buttons[style].tap()
            app.segmentedControls["provider"].buttons[engine].tap()
            app.buttons["clear"].tap()
            app.buttons["mic"].tap()
            XCTAssertTrue(app.staticTexts["● 聽緊"].waitForExistence(timeout: 20), "\(engine): never started listening (\(status(app)))")
            let transcript = app.staticTexts["transcript"]
            let deadline = Date().addingTimeInterval(30)
            while Date() < deadline, transcript.label.unicodeScalars.filter({ (0x4E00...0x9FFF).contains($0.value) }).count < 6 {
                Thread.sleep(forTimeInterval: 0.5)
            }
            shot(app, "\(style)-\(engine)-live")
            Thread.sleep(forTimeInterval: 6)          // the rest of the 7.5 s clip
            app.buttons["mic"].tap()                  // 停止
            Thread.sleep(forTimeInterval: 3)
            if style == "書面語" {
                let deadline = Date().addingTimeInterval(15)
                while Date() < deadline, !transcript.label.contains("我們") { Thread.sleep(forTimeInterval: 0.5) }
            }
            shot(app, "\(style)-\(engine)-stopped")
            let text = transcript.label
            print("TRANSCRIPT \(style) \(engine): \(text)")
            if style == "書面語" {
                XCTAssertTrue(text.contains("我們") && !text.contains("我哋"), "\(engine) 書面語 not rewritten: \(text)")
            } else {
                XCTAssertTrue(text.contains("我哋"), "\(engine) 口語 lost the spoken form: \(text)")
            }
            XCTAssertGreaterThanOrEqual(text.unicodeScalars.filter { (0x4E00...0x9FFF).contains($0.value) }.count, 6,
                                        "\(engine) returned no Cantonese text (\(status(app)))")
        }
        app.buttons["clear"].tap()
        let cleared = app.staticTexts["transcript"]
        XCTAssertTrue(!cleared.exists || cleared.label.trimmingCharacters(in: .whitespaces).isEmpty, "clear left text behind")
        shot(app, "cleared")
    }

    private func status(_ app: XCUIApplication) -> String { app.staticTexts["status"].label }

    private func shot(_ app: XCUIApplication, _ name: String) {
        let a = XCTAttachment(screenshot: app.screenshot())
        a.name = name
        a.lifetime = .keepAlways
        add(a)
    }
}
