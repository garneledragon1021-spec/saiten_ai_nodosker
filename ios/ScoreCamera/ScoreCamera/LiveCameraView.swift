import AVFoundation
import CoreImage
import SwiftUI
import UIKit

/// AVCaptureSessionのプレビューをSwiftUIへ埋め込み、一定間隔でカメラ画像を採点する。
struct LiveCameraView: UIViewRepresentable {
    let onResult: (ScoringResult) -> Void
    let onError: (String) -> Void

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        view.videoPreviewLayer.session = context.coordinator.session
        context.coordinator.configureAndStart()
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    static func dismantleUIView(_ uiView: PreviewView, coordinator: Coordinator) {
        coordinator.stop()
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(onResult: onResult, onError: onError)
    }

    final class PreviewView: UIView {
        override class var layerClass: AnyClass {
            AVCaptureVideoPreviewLayer.self
        }

        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            if let connection = videoPreviewLayer.connection {
                Coordinator.setPortraitOrientation(on: connection)
            }
        }
    }

    final class Coordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
        let session = AVCaptureSession()

        // カメラ開始/停止はsessionQueue、フレーム解析はanalysisQueueに分け、UIスレッドを塞がない。
        // 処理フロー: 権限確認 -> セッション構成 -> フレーム受信 -> 1秒間隔でCGImage化 -> ScoringEngineで採点 -> メインスレッドへ結果通知。
        private let sessionQueue = DispatchQueue(label: "jp.saitenai.ScoreCamera.session")
        private let analysisQueue = DispatchQueue(label: "jp.saitenai.ScoreCamera.analysis", qos: .userInitiated)
        private let ciContext = CIContext()
        private let minimumAnalysisInterval: TimeInterval = 1.0
        private let onResult: (ScoringResult) -> Void
        private let onError: (String) -> Void

        private var engine: ScoringEngine?
        private var isConfigured = false
        private var isProcessingFrame = false
        private var lastAnalysisTime: CFTimeInterval = 0
        private var lastErrorMessage: String?

        init(onResult: @escaping (ScoringResult) -> Void, onError: @escaping (String) -> Void) {
            self.onResult = onResult
            self.onError = onError
            super.init()
        }

        func configureAndStart() {
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized:
                sessionQueue.async { self.configureSessionIfNeededAndStart() }
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    if granted {
                        self.sessionQueue.async { self.configureSessionIfNeededAndStart() }
                    } else {
                        self.publishError("カメラの使用が許可されていません。")
                    }
                }
            case .denied, .restricted:
                publishError("カメラの使用が許可されていません。設定アプリでカメラを許可してください。")
            @unknown default:
                publishError("カメラの状態を確認できませんでした。")
            }
        }

        func stop() {
            sessionQueue.async {
                if self.session.isRunning {
                    self.session.stopRunning()
                }
            }
        }

        private func configureSessionIfNeededAndStart() {
            if !isConfigured {
                do {
                    try configureSession()
                    isConfigured = true
                } catch {
                    publishError(error.localizedDescription)
                    return
                }
            }

            if !session.isRunning {
                session.startRunning()
            }
        }

        private func configureSession() throws {
            session.beginConfiguration()
            defer { session.commitConfiguration() }

            if session.canSetSessionPreset(.hd1920x1080) {
                session.sessionPreset = .hd1920x1080
            } else if session.canSetSessionPreset(.hd1280x720) {
                session.sessionPreset = .hd1280x720
            } else {
                session.sessionPreset = .high
            }

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                throw LiveCameraError.cameraUnavailable
            }

            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else {
                throw LiveCameraError.cameraInputUnavailable
            }
            session.addInput(input)

            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            ]
            output.setSampleBufferDelegate(self, queue: analysisQueue)

            guard session.canAddOutput(output) else {
                throw LiveCameraError.cameraOutputUnavailable
            }
            session.addOutput(output)

            if let connection = output.connection(with: .video) {
                Self.setPortraitOrientation(on: connection)
                if connection.isVideoMirroringSupported {
                    connection.isVideoMirrored = false
                }
            }
        }

        static func setPortraitOrientation(on connection: AVCaptureConnection) {
            let portraitRotationAngle: CGFloat = 90
            if connection.isVideoRotationAngleSupported(portraitRotationAngle) {
                connection.videoRotationAngle = portraitRotationAngle
            }
        }

        func captureOutput(
            _ output: AVCaptureOutput,
            didOutput sampleBuffer: CMSampleBuffer,
            from connection: AVCaptureConnection
        ) {
            // analysisQueue上で呼ばれる。連続フレームをすべて採点すると端末負荷が高いため、
            // 前回から1秒以上経ち、前の採点が終わっているフレームだけを処理する。
            let now = CACurrentMediaTime()
            guard now - lastAnalysisTime >= minimumAnalysisInterval,
                  !isProcessingFrame else {
                return
            }

            lastAnalysisTime = now
            isProcessingFrame = true
            defer { isProcessingFrame = false }

            guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
                  let cgImage = makeCGImage(from: imageBuffer) else {
                publishError("カメラ画像を読み取れませんでした。")
                return
            }

            do {
                if engine == nil {
                    engine = try ScoringEngine()
                }
                let result = try engine!.score(image: cgImage)
                lastErrorMessage = nil
                DispatchQueue.main.async {
                    self.onResult(result)
                }
            } catch {
                publishError(error.localizedDescription)
            }
        }

        private func makeCGImage(from imageBuffer: CVImageBuffer) -> CGImage? {
            let image = CIImage(cvImageBuffer: imageBuffer)
            return ciContext.createCGImage(image, from: image.extent)
        }

        private func publishError(_ message: String) {
            guard message != lastErrorMessage else {
                return
            }
            lastErrorMessage = message
            DispatchQueue.main.async {
                self.onError(message)
            }
        }
    }
}

private enum LiveCameraError: LocalizedError {
    case cameraUnavailable
    case cameraInputUnavailable
    case cameraOutputUnavailable

    var errorDescription: String? {
        switch self {
        case .cameraUnavailable:
            return "背面カメラを使用できません。"
        case .cameraInputUnavailable:
            return "カメラ入力を開始できません。"
        case .cameraOutputUnavailable:
            return "カメラ画像の取得を開始できません。"
        }
    }
}
