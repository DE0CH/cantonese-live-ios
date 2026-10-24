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

    /// The connection is cut 3 s into the clip: the app says it is reconnecting, comes back by itself, and the audio
    /// from the gap still reaches the transcript.
    func testReconnect() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CANTONESE_TEST_AUDIO"] = "1"
        app.launchEnvironment["CANTONESE_TEST_DROP"] = "1"
        app.launch()
        app.buttons["clear"].tap()
        app.buttons["mic"].tap()
        XCTAssertTrue(app.staticTexts["連線斷咗，重新連接緊…"].waitForExistence(timeout: 20), "never reconnected (\(status(app)))")
        shot(app, "reconnecting")
        XCTAssertTrue(app.buttons["mic"].label == "停止", "a dropped connection stopped the session")
        XCTAssertTrue(app.staticTexts["● 聽緊"].waitForExistence(timeout: 20), "never came back (\(status(app)))")
        Thread.sleep(forTimeInterval: 10)         // the held audio and the rest of the clip
        app.buttons["mic"].tap()
        Thread.sleep(forTimeInterval: 3)
        shot(app, "reconnected")
        let text = app.staticTexts["transcript"].label
        print("TRANSCRIPT (reconnect): \(text)")
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
