// swift-tools-version: 6.0
// (Not named "replay": swift-huggingface depends on a package of that name, mattt/Replay.)
// The app's own loop (ios/App/Worker.swift, linked in) with the real models on a Mac, over recorded clips: every run it
// makes is written out for the simulator's stand-in (UI tests, App Store demo), and the screen after each run is printed.
// The speech-swift revision is the app's (ios/project.yml); the model-check workflow checks they match.
import PackageDescription

let package = Package(
    name: "demo-passes",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/DE0CH/speech-swift", revision: "fb4c9f55cc1ad8b4df685840fdfd6581644eda37"),
    ],
    targets: [
        .executableTarget(name: "demo-passes", dependencies: [.product(name: "Qwen3ASR", package: "speech-swift")]),
    ],
    swiftLanguageModes: [.v5]   // like the app (SWIFT_VERSION 5.9)
)
