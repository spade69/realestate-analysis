import Foundation
import Vision
import AppKit

guard CommandLine.arguments.count >= 2 else {
    print("Usage: ocr <image>")
    exit(1)
}
let path = CommandLine.arguments[1]
guard let img = NSImage(contentsOfFile: path),
      let tiff = img.tiffRepresentation,
      let rep = NSBitmapImageRep(data: tiff),
      let cg = rep.cgImage else {
    print("Failed to load image: \(path)")
    exit(2)
}

let req = VNRecognizeTextRequest { req, err in
    guard let obs = req.results as? [VNRecognizedTextObservation] else { return }
    // 按 y 坐标降序（Vision 里 y=0 在底部，越高代表越靠上）
    let sorted = obs.sorted { $0.boundingBox.midY > $1.boundingBox.midY }
    for o in sorted {
        guard let top = o.topCandidates(1).first else { continue }
        let bb = o.boundingBox
        // 输出：y_mid x_mid text
        print(String(format: "%.4f\t%.4f\t%@", bb.midY, bb.midX, top.string))
    }
}
req.recognitionLevel = .accurate
req.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
req.usesLanguageCorrection = false  // 数字/年份别被"矫正"

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
try? handler.perform([req])
