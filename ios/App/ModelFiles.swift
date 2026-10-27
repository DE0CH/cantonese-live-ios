import CoreML
import Foundation

/// Puts the converted model in place the first time: downloads each .mlpackage file (and the tokenizer and config) from
/// the Hugging Face repo, compiles each package on the phone into the .mlmodelc speech-swift loads, and deletes the
/// package. A file already in place is skipped, so an interrupted download resumes at the next start.
enum ModelFiles {
    struct Entry: Decodable { let type: String; let path: String; let size: Int? }

    static func ensure(repo: String, in dir: URL, progress: @escaping (Double) -> Void) async throws {
        let fm = FileManager.default
        let packages = ["encoder", "embedding", "decoder_part1", "decoder_part2"]
        let extras = ["config.json", "vocab.json", "merges.txt", "tokenizer_config.json"]
        if (packages.map { "\($0).mlmodelc" } + extras).allSatisfy({ fm.fileExists(atPath: dir.appendingPathComponent($0).path) }) {
            return
        }
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        let listURL = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main?recursive=true")!
        let (data, _) = try await URLSession.shared.data(from: listURL)
        let files = try JSONDecoder().decode([Entry].self, from: data).filter { entry in
            entry.type == "file" && (extras.contains(entry.path)
                || packages.contains { name in
                    entry.path.hasPrefix("\(name).mlpackage/")
                        && !fm.fileExists(atPath: dir.appendingPathComponent("\(name).mlmodelc").path)
                })
        }
        let total = Double(max(1, files.reduce(0) { $0 + ($1.size ?? 0) }))
        var done = 0.0
        for f in files {
            let dest = dir.appendingPathComponent(f.path)
            if let size = f.size, (try? dest.resourceValues(forKeys: [.fileSizeKey]).fileSize) == size {
                done += Double(size); continue
            }
            try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(f.path)")!
            let base = done
            let tmp = try await Download.run(url) { got in progress(0.95 * (base + Double(got)) / total) }
            try? fm.removeItem(at: dest)
            try fm.moveItem(at: tmp, to: dest)
            done += Double(f.size ?? 0)
        }

        for name in packages {
            let compiled = dir.appendingPathComponent("\(name).mlmodelc")
            let package = dir.appendingPathComponent("\(name).mlpackage")
            guard !fm.fileExists(atPath: compiled.path) else { continue }
            let tmp = try await MLModel.compileModel(at: package)
            try fm.moveItem(at: tmp, to: compiled)
            try? fm.removeItem(at: package)
            progress(0.95 + 0.05 * Double(packages.firstIndex(of: name)! + 1) / Double(packages.count))
        }
    }
}

/// One file download to a temporary file, reporting bytes received.
private final class Download: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private var continuation: CheckedContinuation<URL, Error>?
    private let onBytes: (Int64) -> Void

    private init(onBytes: @escaping (Int64) -> Void) { self.onBytes = onBytes }

    static func run(_ url: URL, onBytes: @escaping (Int64) -> Void) async throws -> URL {
        let d = Download(onBytes: onBytes)
        let session = URLSession(configuration: .default, delegate: d, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return try await withCheckedThrowingContinuation { c in
            d.continuation = c
            session.downloadTask(with: url).resume()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        onBytes(totalBytesWritten)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // The file at `location` is deleted when this returns: move it somewhere that lasts.
        let keep = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
                throw URLError(.badServerResponse)
            }
            try FileManager.default.moveItem(at: location, to: keep)
            continuation?.resume(returning: keep)
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { continuation?.resume(throwing: error); continuation = nil }
    }
}
