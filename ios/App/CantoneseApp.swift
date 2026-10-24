import SwiftUI

@main
struct CantoneseApp: App {
    @StateObject private var transcriber = Transcriber()
    var body: some Scene {
        WindowGroup { ContentView().environmentObject(transcriber) }
    }
}
