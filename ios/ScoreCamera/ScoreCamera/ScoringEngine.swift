import CoreML
import CoreImage
import CoreVideo
import UIKit
import Vision

struct ScoringResult {
    let scores: [Int]
    let questionCount: Int
    let digitCounts: [Int]
    let scoreConfidence: Float
    let usedDocumentCorrection: Bool
    let answerImage: CGImage

    var digitTotal: Int {
        digitCounts.reduce(0, +)
    }

    var diagnosticText: String {
        let correctionText = usedDocumentCorrection ? "文書補正あり" : "文書補正なし"
        let averageConfidence = digitTotal > 0 ? Int((scoreConfidence / Float(digitTotal)) * 100) : 0
        return "設問検出: \(questionCount)個 / 数字検出: \(digitTotal)個 / 平均信頼度: \(averageConfidence)% / \(correctionText)"
    }

    var qualityScore: Float {
        // まず数字の個数を最優先し、同数なら検出信頼度が高い結果を採用する。
        Float(digitTotal * 1000 + questionCount * 10) + scoreConfidence
    }
}

/// 既存all.pyと同じQR確認、領域切り出し、数字認識を端末内で実行する。
final class ScoringEngine {
    private let modelInputSize = 640
    private let qrText = "fujishima startup QRcode"
    private let cropper: MLModel
    private let detector: MLModel
    private let ciContext = CIContext()

    init() throws {
        cropper = try Self.loadModel(named: "QuestionCropper")
        detector = try Self.loadModel(named: "ScoreDetector")
    }

    /// QRコードの下にある答案を切り出し、各設問の得点一覧を返す。
    ///
    /// 採点処理フロー:
    /// 1. 元画像と、必要に応じた文書補正後画像を候補として用意する。
    /// 2. 各候補でQRコードを確認し、QRコードより下の答案領域を切り出す。
    /// 3. QuestionCropperで設問ごとの解答欄を検出する。
    /// 4. 各解答欄をScoreDetectorへ渡し、左から最大3桁の数字を得点へ変換する。
    /// 5. 数字検出数と信頼度が最も高い候補を最終結果として返す。
    func score(image: CGImage) throws -> ScoringResult {
        let candidates = imageCandidates(from: image)
        var bestResult: ScoringResult?
        var lastError: Error?

        for candidate in candidates {
            do {
                let result = try scoreCandidate(
                    image: candidate.image,
                    usedDocumentCorrection: candidate.usedDocumentCorrection
                )

                // 元画像と文書補正後画像の両方を試し、数字検出数と信頼度が高い方を採用する。
                if bestResult == nil || result.qualityScore > bestResult!.qualityScore {
                    bestResult = result
                }
            } catch {
                lastError = error
            }
        }

        if let bestResult {
            return bestResult
        }
        throw lastError ?? ScoringError.questionNotFound
    }

    private func scoreCandidate(image: CGImage, usedDocumentCorrection: Bool) throws -> ScoringResult {
        let answerImage = try cropAnswerArea(from: image)
        let questionBoxes = try detectWithFallback(
            with: cropper,
            image: answerImage,
            classCount: 1,
            confidenceThresholds: [0.50, 0.35, 0.20]
        ).sorted { $0.rect.minX < $1.rect.minX }

        guard !questionBoxes.isEmpty else {
            throw ScoringError.questionNotFound
        }

        var scores: [Int] = []
        var digitCounts: [Int] = []
        var scoreConfidence: Float = 0

        for question in questionBoxes {
            let questionRect = expandedRect(
                question.rect,
                xRatio: 0.10,
                yRatio: 0.18,
                imageWidth: answerImage.width,
                imageHeight: answerImage.height
            )
            guard let questionImage = crop(answerImage, to: questionRect) else {
                continue
            }
            let questionScore = try scoreQuestion(questionImage)
            scores.append(questionScore.score)
            digitCounts.append(questionScore.digitCount)
            scoreConfidence += questionScore.confidence
        }

        guard !scores.isEmpty else {
            throw ScoringError.questionNotFound
        }

        let result = ScoringResult(
            scores: scores,
            questionCount: questionBoxes.count,
            digitCounts: digitCounts,
            scoreConfidence: scoreConfidence,
            usedDocumentCorrection: usedDocumentCorrection,
            answerImage: answerImage
        )
        print("[ScoreCamera] \(usedDocumentCorrection ? "corrected" : "original") \(result.diagnosticText)")

        return result
    }

    /// 撮影画像では紙が傾くため、元画像に加えて文書補正後の画像も候補にする。
    private func imageCandidates(from image: CGImage) -> [ImageCandidate] {
        var candidates = [ImageCandidate(image: image, usedDocumentCorrection: false)]
        if let corrected = try? correctedDocumentImage(from: image) {
            candidates.append(ImageCandidate(image: corrected, usedDocumentCorrection: true))
        }
        return candidates
    }

    /// Visionで用紙らしい四角形を検出し、斜めから撮った画像を正面画像へ補正する。
    private func correctedDocumentImage(from image: CGImage) throws -> CGImage? {
        let request = VNDetectRectanglesRequest()
        request.maximumObservations = 1
        request.minimumConfidence = 0.60
        request.minimumSize = 0.35
        request.quadratureTolerance = 25

        //推論
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        guard let rectangle = request.results?.first else {
            return nil
        }

        let ciImage = CIImage(cgImage: image)
        let width = CGFloat(image.width)
        let height = CGFloat(image.height)

        func vector(_ point: CGPoint) -> CIVector {
            CIVector(x: point.x * width, y: point.y * height)
        }

        guard let filter = CIFilter(name: "CIPerspectiveCorrection") else {
            return nil
        }
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(vector(rectangle.topLeft), forKey: "inputTopLeft")
        filter.setValue(vector(rectangle.topRight), forKey: "inputTopRight")
        filter.setValue(vector(rectangle.bottomLeft), forKey: "inputBottomLeft")
        filter.setValue(vector(rectangle.bottomRight), forKey: "inputBottomRight")

        guard let outputImage = filter.outputImage else {
            return nil
        }
        return ciContext.createCGImage(outputImage, from: outputImage.extent)
    }

    /// QRコードの文字列を確認し、QRコードより下の領域を採点対象として切り出す。
    private func cropAnswerArea(from image: CGImage) throws -> CGImage {
        var qrObservations: [VNBarcodeObservation] = []
        let request = VNDetectBarcodesRequest { request, _ in
            qrObservations = (request.results as? [VNBarcodeObservation]) ?? []
        }
        request.symbologies = [.qr]
        
        //推論
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])

        guard let qr = qrObservations.first(where: { $0.payloadStringValue == qrText }) else {
            throw ScoringError.qrNotFound
        }

        // Visionの座標系は左下原点なので、画像座標系（左上原点）に変換する。
        let imageWidth = CGFloat(image.width)
        let imageHeight = CGFloat(image.height)
        let cropStartX = max(0, qr.boundingBox.minX * imageWidth)
        let cropStartY = max(0, (1 - qr.boundingBox.minY) * imageHeight)
        let cropRect = CGRect(
            x: cropStartX,
            y: cropStartY,
            width: imageWidth - cropStartX,
            height: imageHeight - cropStartY
        )

        guard let answerImage = crop(image, to: cropRect) else {
            throw ScoringError.answerAreaNotFound
        }
        return answerImage
    }

    /// 1つの設問画像に含まれる最大3桁の数字を左から順に得点へ変換する。
    private func scoreQuestion(_ image: CGImage) throws -> QuestionScore {
        var bestScore = QuestionScore(score: 0, digitCount: 0, confidence: 0)

        // 撮影画像は明るさ・影・印刷濃度の差が出るため、原画像と補正画像を比較して一番よい結果を採用する。
        for candidate in questionImageCandidates(from: image) {
            let digitDetections = try detectBestDigits(in: candidate)
            let selectedDetections = digitDetections
                .sorted { $0.confidence > $1.confidence }
                .prefix(3)
                .sorted { $0.rect.minX < $1.rect.minX }

            let digits = selectedDetections.map(\.classIndex)
            let confidence = selectedDetections.reduce(Float(0)) { $0 + $1.confidence }
            let score = QuestionScore(
                score: digits.reduce(0) { $0 * 10 + $1 },
                digitCount: digits.count,
                confidence: confidence
            )

            if score.qualityScore > bestScore.qualityScore {
                bestScore = score
            }
        }

        return bestScore
    }

    /// 撮影条件で信頼度が下がる場合に備え、高い閾値から順に試す。
    private func detectWithFallback(
        with model: MLModel,
        image: CGImage,
        classCount: Int,
        confidenceThresholds: [Double]
    ) throws -> [Detection] {
        var lastDetections: [Detection] = []
        for threshold in confidenceThresholds {
            let detections = try detect(
                with: model,
                image: image,
                classCount: classCount,
                confidenceThreshold: threshold
            )
            if !detections.isEmpty {
                return detections
            }
            lastDetections = detections
        }
        return lastDetections
    }

    /// 数字検出では「最初に1個見つかったら終了」ではなく、閾値を複数試して最も情報量の多い結果を採用する。
    private func detectBestDigits(in image: CGImage) throws -> [Detection] {
        var bestDetections: [Detection] = []

        for threshold in [0.75, 0.60, 0.45] {
            let detections = try detect(
                with: detector,
                image: image,
                classCount: 10,
                confidenceThreshold: threshold
            )
            let filtered = filterDigitDetections(detections, image: image)
            if digitDetectionQuality(filtered) > digitDetectionQuality(bestDetections) {
                bestDetections = filtered
            }
        }

        return bestDetections
    }

    /// 極端に小さい/大きい枠はノイズの可能性が高いため除外する。
    private func filterDigitDetections(_ detections: [Detection], image: CGImage) -> [Detection] {
        let imageWidth = CGFloat(image.width)
        let imageHeight = CGFloat(image.height)
        return detections.filter { detection in
            let widthRatio = detection.rect.width / imageWidth
            let heightRatio = detection.rect.height / imageHeight
            return widthRatio >= 0.03
                && widthRatio <= 0.70
                && heightRatio >= 0.12
                && heightRatio <= 0.95
        }
    }

    private func digitDetectionQuality(_ detections: [Detection]) -> Float {
        let selected = detections
            .sorted { $0.confidence > $1.confidence }
            .prefix(3)
        let confidence = selected.reduce(Float(0)) { $0 + $1.confidence }
        return Float(selected.count * 100) + confidence
    }

    /// Core MLモデルのNMS出力を画像座標へ戻して検出結果を作成する。
    private func detect(
        with model: MLModel,
        image: CGImage,
        classCount: Int,
        confidenceThreshold: Double
    ) throws -> [Detection] {
        let letterboxed = try makeLetterboxedImage(from: image)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: letterboxed.pixelBuffer),
            "iouThreshold": MLFeatureValue(double: 0.70),
            "confidenceThreshold": MLFeatureValue(double: confidenceThreshold),
        ])
        let output = try model.prediction(from: input)
        guard let coordinates = output.featureValue(for: "coordinates")?.multiArrayValue,
              let confidences = output.featureValue(for: "confidence")?.multiArrayValue else {
            throw ScoringError.invalidModelOutput
        }

        let rowCount = coordinates.shape.first?.intValue ?? 0
        let confidenceColumns = confidences.shape.count > 1 ? confidences.shape[1].intValue : 0
        var detections: [Detection] = []

        for row in 0 ..< rowCount {
            var bestClass = 0
            var bestConfidence: Float = 0
            for classIndex in 0 ..< min(classCount, confidenceColumns) {
                let confidence = value(in: confidences, row: row, column: classIndex)
                if confidence > bestConfidence {
                    bestConfidence = confidence
                    bestClass = classIndex
                }
            }
            guard bestConfidence >= Float(confidenceThreshold) else {
                continue
            }

            let centerX = CGFloat(value(in: coordinates, row: row, column: 0)) * CGFloat(modelInputSize)
            let centerY = CGFloat(value(in: coordinates, row: row, column: 1)) * CGFloat(modelInputSize)
            let width = CGFloat(value(in: coordinates, row: row, column: 2)) * CGFloat(modelInputSize)
            let height = CGFloat(value(in: coordinates, row: row, column: 3)) * CGFloat(modelInputSize)
            let modelRect = CGRect(
                x: centerX - width / 2,
                y: centerY - height / 2,
                width: width,
                height: height
            )
            let imageRect = letterboxed.originalRect(from: modelRect)
            guard imageRect.width > 1, imageRect.height > 1 else {
                continue
            }
            detections.append(Detection(rect: imageRect, classIndex: bestClass, confidence: bestConfidence))
        }
        return detections
    }

    /// 縦横比を維持して640×640のグレー背景画像へ配置し、Python版YOLOの前処理に合わせる。
    private func makeLetterboxedImage(from image: CGImage) throws -> LetterboxedImage {
        let sourceWidth = CGFloat(image.width)
        let sourceHeight = CGFloat(image.height)
        let scale = min(CGFloat(modelInputSize) / sourceWidth, CGFloat(modelInputSize) / sourceHeight)
        let scaledWidth = max(1, Int((sourceWidth * scale).rounded()))
        let scaledHeight = max(1, Int((sourceHeight * scale).rounded()))
        let paddingX = CGFloat(modelInputSize - scaledWidth) / 2
        let paddingY = CGFloat(modelInputSize - scaledHeight) / 2

        var maybePixelBuffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            modelInputSize,
            modelInputSize,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &maybePixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
            throw ScoringError.imageConversionFailed
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: modelInputSize,
            height: modelInputSize,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw ScoringError.imageConversionFailed
        }

        context.setFillColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: modelInputSize, height: modelInputSize))
        context.interpolationQuality = .high
        // Core MLへ渡すPixelBufferは上から下へ並ぶため、ここで上下反転しない。
        // 反転すると数字の2が5のように見え、数字検出が大きく崩れる。
        context.draw(
            image,
            in: CGRect(x: paddingX, y: paddingY, width: CGFloat(scaledWidth), height: CGFloat(scaledHeight))
        )

        return LetterboxedImage(
            pixelBuffer: pixelBuffer,
            scale: scale,
            paddingX: paddingX,
            paddingY: paddingY,
            originalSize: CGSize(width: sourceWidth, height: sourceHeight)
        )
    }

    /// NMSで返される2次元MLMultiArrayの値をFloatへ変換する。
    private func value(in array: MLMultiArray, row: Int, column: Int) -> Float {
        if array.dataType == .float32 {
            let offset = row * array.strides[0].intValue + column * array.strides[1].intValue
            return array.dataPointer.assumingMemoryBound(to: Float32.self)[offset]
        }
        return array[[NSNumber(value: row), NSNumber(value: column)]].floatValue
    }

    /// Xcodeがコンパイルしてアプリへ組み込んだCore MLモデルを読み込む。
    private static func loadModel(named name: String) throws -> MLModel {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw ScoringError.modelNotFound(name)
        }
        return try MLModel(contentsOf: url)
    }

    private func crop(_ image: CGImage, to rect: CGRect) -> CGImage? {
        let bounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let clippedRect = rect.integral.intersection(bounds)
        guard clippedRect.width > 1, clippedRect.height > 1 else {
            return nil
        }
        return image.cropping(to: clippedRect)
    }

    private func expandedRect(
        _ rect: CGRect,
        xRatio: CGFloat,
        yRatio: CGFloat,
        imageWidth: Int,
        imageHeight: Int
    ) -> CGRect {
        let dx = rect.width * xRatio
        let dy = rect.height * yRatio
        return rect
            .insetBy(dx: -dx, dy: -dy)
            .intersection(CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight))
    }

    private func questionImageCandidates(from image: CGImage) -> [CGImage] {
        var candidates = [image]

        if let contrastImage = adjustedImage(
            from: image,
            contrast: 1.35,
            brightness: 0.02,
            saturation: 1.0,
            sharpness: 0.25
        ) {
            candidates.append(contrastImage)
        }

        if let monochromeImage = adjustedImage(
            from: image,
            contrast: 1.55,
            brightness: 0.03,
            saturation: 0.0,
            sharpness: 0.35
        ) {
            candidates.append(monochromeImage)
        }

        return candidates
    }

    private func adjustedImage(
        from image: CGImage,
        contrast: Float,
        brightness: Float,
        saturation: Float,
        sharpness: Float
    ) -> CGImage? {
        let inputImage = CIImage(cgImage: image)
        var outputImage = inputImage.applyingFilter(
            "CIColorControls",
            parameters: [
                kCIInputContrastKey: contrast,
                kCIInputBrightnessKey: brightness,
                kCIInputSaturationKey: saturation,
            ]
        )
        outputImage = outputImage.applyingFilter(
            "CISharpenLuminance",
            parameters: [
                kCIInputSharpnessKey: sharpness,
            ]
        )
        return ciContext.createCGImage(outputImage, from: inputImage.extent)
    }
}

private struct ImageCandidate {
    let image: CGImage
    let usedDocumentCorrection: Bool
}

private struct QuestionScore {
    let score: Int
    let digitCount: Int
    let confidence: Float

    var qualityScore: Float {
        Float(digitCount * 100) + confidence
    }
}

private struct Detection {
    let rect: CGRect
    let classIndex: Int
    let confidence: Float
}

private struct LetterboxedImage {
    let pixelBuffer: CVPixelBuffer
    let scale: CGFloat
    let paddingX: CGFloat
    let paddingY: CGFloat
    let originalSize: CGSize

    /// 640×640入力上の座標を元画像の座標へ戻す。
    func originalRect(from modelRect: CGRect) -> CGRect {
        let rect = CGRect(
            x: (modelRect.minX - paddingX) / scale,
            y: (modelRect.minY - paddingY) / scale,
            width: modelRect.width / scale,
            height: modelRect.height / scale
        )
        return rect.intersection(CGRect(origin: .zero, size: originalSize))
    }
}

private enum ScoringError: LocalizedError {
    case modelNotFound(String)
    case qrNotFound
    case answerAreaNotFound
    case questionNotFound
    case invalidModelOutput
    case imageConversionFailed

    var errorDescription: String? {
        switch self {
        case let .modelNotFound(name):
            return "採点モデル（\(name)）を読み込めませんでした。"
        case .qrNotFound:
            return "指定されたQRコードを画像から読み取れませんでした。"
        case .answerAreaNotFound:
            return "採点対象の答案領域を切り出せませんでした。"
        case .questionNotFound:
            return "設問の解答欄を検出できませんでした。"
        case .invalidModelOutput:
            return "採点モデルの出力形式が不正です。"
        case .imageConversionFailed:
            return "画像を採点用の形式へ変換できませんでした。"
        }
    }
}
