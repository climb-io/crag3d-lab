// 戶外 3D Topo 實驗：Apple Object Capture 建岩壁模型，輸出 OBJ 與相機位置。
// 給遮罩資料夾時（sky.py 產生，每張照片一個同名 PNG，白色建模、黑色是天空），照片連同遮罩一張張交給 Apple，天空不建成表面、不貼進貼圖。
import CoreImage
import Foundation
import ImageIO
import RealityKit

@main struct Capture {
    static func main() async throws {
        setbuf(stdout, nil)
        guard PhotogrammetrySession.isSupported else {
            print("這台電腦不支援 PhotogrammetrySession")
            exit(2)
        }
        let args = CommandLine.arguments, options = args.dropFirst(5)
        guard args.count >= 5 else {
            print("用法：capture_lite 照片資料夾 OBJ輸出資料夾 最多面數 貼圖邊長(1024|2048|4096|8192) [sequential] [masks=遮罩資料夾]")
            print("sequential：照片檔名照拍攝順序、前後張重疊（站點拍法），Apple 只配前後的照片")
            exit(2)
        }
        let input = URL(fileURLWithPath: args[1], isDirectory: true)
        let output = URL(fileURLWithPath: args[2], isDirectory: true)
        let masks = options.first { $0.hasPrefix("masks=") }.map { URL(fileURLWithPath: String($0.dropFirst(6)), isDirectory: true) }
        var config = PhotogrammetrySession.Configuration()
        config.sampleOrdering = options.contains("sequential") ? .sequential : .unordered
        config.featureSensitivity = .high
        config.checkpointDirectory = output.deletingLastPathComponent().appendingPathComponent("apple-checkpoint", isDirectory: true)
        try? FileManager.default.removeItem(at: config.checkpointDirectory!)   // 上次中斷留下的續跑暫存會讓 Apple 略過照片，每次從頭建
        config.isObjectMaskingEnabled = masks != nil   // 有遮罩才用（每張都有遮罩，Apple 不自己猜物體）
        config.ignoreBoundingBox = true
        config.customDetailSpecification.maximumPolygonCount = UInt(args[3])!
        config.customDetailSpecification.maximumTextureDimension = ["2048": .twoK, "4096": .fourK, "8192": .eightK][args[4]] ?? .oneK
        config.customDetailSpecification.outputTextureMaps = [.diffuseColor]
        config.customDetailSpecification.textureFormat = .jpeg(compressionQuality: 0.8)
        let started = Date()
        let photos = try FileManager.default.contentsOfDirectory(at: input, includingPropertiesForKeys: nil)
            .filter { ["jpg", "jpeg", "heic", "png"].contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let session = try masks.map { dir in
            try PhotogrammetrySession(input: photos.enumerated().lazy.compactMap { id, url in sample(id, url, dir.appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".png")) }, configuration: config)
        } ?? PhotogrammetrySession(input: input, configuration: config)
        try session.process(requests: [.poses, .modelFile(url: output, detail: .custom)])
        for try await event in session.outputs {
            switch event {
            case .requestProgress(_, let fraction): print("進度 \(fraction)")
            case .requestComplete(_, let result):
                if case .poses(let poses) = result {
                    let rows: [[String: Any]] = poses.posesBySample.compactMap { id, pose in
                        guard let url = poses.urlsBySample[id] ?? (photos.indices.contains(id) ? photos[id] : nil) else { return nil }
                        let p = pose.translation
                        let q = pose.rotation.vector
                        return ["name": url.lastPathComponent, "position": [p.x, p.y, p.z], "quaternion": [q.x, q.y, q.z, q.w]]
                    }
                    let json = try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys])
                    try json.write(to: output.deletingLastPathComponent().appendingPathComponent("apple-poses.json"))
                    print("相機數 \(rows.count)")
                } else {
                    print("完成 \(result)")
                }
            case .requestError(_, let error): throw error
            case .processingComplete:
                print("總秒數 \(Date().timeIntervalSince(started))")
                try? FileManager.default.removeItem(at: config.checkpointDirectory!)   // 建完就刪中斷續跑的暫存（一次約 2 GB）
                return
            case .invalidSample(let id, let reason): print("無效照片 \(id)：\(reason)")
            case .skippedSample(let id): print("略過照片 \(id)")
            default: break
            }
        }
    }

    static let context = CIContext()

    // 照片照 EXIF 轉正（跟 sky.py 的遮罩同一個方向），EXIF 其他欄位（焦距、GPS）照樣交給 Apple。
    // Apple 會把所有照片留在記憶體：用 YUV（每像素 1.5 位元組，BGRA 要 4），172 張長邊 3024 的照片 BGRA 要 3.5 GB
    static func sample(_ id: Int, _ url: URL, _ maskURL: URL) -> PhotogrammetrySample? {
        guard let image = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]), let mask = CIImage(contentsOf: maskURL),
              let pixels = buffer(image, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange), let maskPixels = buffer(mask, kCVPixelFormatType_OneComponent8) else {
            print("讀不到照片或遮罩：\(url.lastPathComponent)")
            return nil
        }
        var s = PhotogrammetrySample(id: id, image: pixels)
        s.objectMask = maskPixels
        if let source = CGImageSourceCreateWithURL(url as CFURL, nil), var meta = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] {
            meta[kCGImagePropertyOrientation as String] = nil   // 已經轉正
            if var tiff = meta[kCGImagePropertyTIFFDictionary as String] as? [String: Any] { tiff[kCGImagePropertyTIFFOrientation as String] = nil; meta[kCGImagePropertyTIFFDictionary as String] = tiff }
            s.metadata = meta
        }
        return s
    }

    static func buffer(_ image: CIImage, _ format: OSType) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, Int(image.extent.width), Int(image.extent.height), format, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
        guard let pb else { return nil }
        context.render(image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY)), to: pb)
        return pb
    }
}
