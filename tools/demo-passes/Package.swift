// swift-tools-version: 6.0
// (Not named "replay": swift-huggingface depends on a package of that name, mattt/Replay.)
// The app's own loop and models (ios/App/Worker.swift and Models.swift, linked in) with the real models on a Mac, over
// recorded clips: every black run it makes is written out for the simulator's stand-in (UI tests, App Store demo).
// The package revisions are the app's (ios/project.yml); the model-check workflow checks speech-swift's matches.
import PackageDescription

let package = Package(
    name: "demo-passes",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/soniqo/speech-swift", revision: "d53bcb333882ebbd7444ae059b270127565594a0"),
        .package(url: "https://github.com/k2-fsa/sherpa-onnx", exact: "2.0.1"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/ddddxxx/SwiftyOpenCC", revision: "1d8105a0f7199c90af722bff62728050c858e777"),
    ],
    targets: [
        .executableTarget(name: "demo-passes", dependencies: [
            .product(name: "Qwen3ASR", package: "speech-swift"),
            .product(name: "sherpa-onnx", package: "sherpa-onnx"),
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "OpenCC", package: "SwiftyOpenCC"),
        ]),
    ],
    swiftLanguageModes: [.v5]   // like the app (SWIFT_VERSION 5.9)
)
