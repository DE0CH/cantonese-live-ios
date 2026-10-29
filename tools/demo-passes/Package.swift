// swift-tools-version: 6.0
// (Not named "replay": swift-huggingface depends on a package of that name, mattt/Replay.)
// The app's own loop (ios/App/Worker.swift, linked in) with the real model on a Mac, over a recorded clip: every pass it
// makes is written out for the simulator's stand-in (the App Store demo), and the screen after each pass is printed.
// The speech-swift revision is the app's (ios/project.yml); the model-check workflow checks they match.
import PackageDescription

let package = Package(
    name: "demo-passes",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/DE0CH/speech-swift", revision: "6e7bdff83c8cc638083f150501268f9581bd5a39"),
    ],
    targets: [
        .executableTarget(name: "demo-passes", dependencies: [.product(name: "Qwen3ASR", package: "speech-swift")]),
    ],
    swiftLanguageModes: [.v5]   // like the app (SWIFT_VERSION 5.9)
)
