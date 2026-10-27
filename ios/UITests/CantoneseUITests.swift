import XCTest

/// The phone replica: walks every flow on the simulator, streaming the bundled Cantonese clip instead of the mic. The
/// simulator runs CoreML on the CPU, too slow for the model, so a stand-in answers for it (Transcriber.swift); VAD,
/// segments, OpenCC and the UI are real.
final class CantoneseUITests: XCTestCase {
    func testEveryFlow() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CANTONESE_TEST_AUDIO"] = "1"
        app.launch()
        shot(app, "launch")

        app.buttons["clear"].tap()
        app.buttons["mic"].tap()
        let ready = Date().addingTimeInterval(3 * 60)
        while Date() < ready, status(app) != "● 聽緊" {
            if status(app).hasPrefix("用唔到") { break }
            Thread.sleep(forTimeInterval: 1)
        }
        XCTAssertEqual(status(app), "● 聽緊", "never started listening (\(status(app)))")
        shot(app, "listening")

        // The 7.5 s clip and a second of silence end one segment; its transcription lands as a committed line.
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

        // A second start reuses the loaded models: listening at once.
        app.buttons["mic"].tap()
        XCTAssertTrue(app.staticTexts["● 聽緊"].waitForExistence(timeout: 60), "second start was slow (\(status(app)))")
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
