import SwiftUI
import UIKit

private enum ImageInputSource: Identifiable {
    case camera
    case photoLibrary

    var id: String {
        switch self {
        case .camera: "camera"
        case .photoLibrary: "photoLibrary"
        }
    }

    var pickerSourceType: UIImagePickerController.SourceType {
        switch self {
        case .camera: .camera
        case .photoLibrary: .photoLibrary
        }
    }
}

struct ContentView: View {
    @State private var inputSource: ImageInputSource?
    @State private var isShowingLiveCamera = false
    @State private var isScoring = false
    @State private var capturedImage: UIImage?
    @State private var answerPreviewImage: UIImage?
    @State private var scores: [Int]?
    @State private var diagnosticMessage: String?
    @State private var errorMessage: String?

    private var total: Int? {
        scores?.reduce(0, +)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    VStack(spacing: 8) {
                        Text("合計点")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                        Text(total.map { "\($0) 点" } ?? "-- 点")
                            .font(.system(size: 52, weight: .bold, design: .rounded))
                            .contentTransition(.numericText())
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 30)
                    .background(.blue.opacity(0.1), in: RoundedRectangle(cornerRadius: 20))

                    if let capturedImage {
                        Image(uiImage: capturedImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 200)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }

                    if let answerPreviewImage {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("採点対象として切り出した画像")
                                .font(.headline)
                            Image(uiImage: answerPreviewImage)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 180)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    if let scores {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("設問ごとの得点")
                                .font(.headline)
                            ForEach(Array(scores.enumerated()), id: \.offset) { index, score in
                                HStack {
                                    Text("question\(index + 1)")
                                    Spacer()
                                    Text("\(score) 点")
                                        .fontWeight(.semibold)
                                }
                            }
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.background, in: RoundedRectangle(cornerRadius: 14))
                        .shadow(color: .black.opacity(0.08), radius: 4, y: 2)
                    }

                    if let diagnosticMessage {
                        Label(diagnosticMessage, systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    if let errorMessage {
                        Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    VStack(spacing: 12) {
                        Button {
                            isShowingLiveCamera = true
                        } label: {
                            Label("リアルタイムカメラで採点", systemImage: "camera.viewfinder")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .disabled(isScoring || !UIImagePickerController.isSourceTypeAvailable(.camera))

                        Button {
                            inputSource = .camera
                        } label: {
                            Label(isScoring ? "採点中..." : "写真を撮って採点", systemImage: "camera.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .disabled(isScoring || !UIImagePickerController.isSourceTypeAvailable(.camera))

                        Button {
                            inputSource = .photoLibrary
                        } label: {
                            Label("写真から読み込んで採点", systemImage: "photo.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .disabled(isScoring || !UIImagePickerController.isSourceTypeAvailable(.photoLibrary))
                    }

                    if !UIImagePickerController.isSourceTypeAvailable(.camera) {
                        Text("カメラは実機のiPhoneでのみ利用できます。シミュレータでは写真選択で画面確認できます。")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    Text("通常の写真として撮影・読み込みます。QRコード確認、設問検出、数字認識、合計表示はすべてiPhone内で実行します。画像は外部へ送信しません。")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding()
            }
            .navigationTitle("答案採点")
            .fullScreenCover(isPresented: $isShowingLiveCamera) {
                LiveScoringView {
                    isShowingLiveCamera = false
                } onResult: { result in
                    scores = result.scores
                    answerPreviewImage = UIImage(cgImage: result.answerImage)
                    diagnosticMessage = result.diagnosticText
                    errorMessage = nil
                    capturedImage = nil
                }
            }
            .fullScreenCover(item: $inputSource) { source in
                ImagePicker(sourceType: source.pickerSourceType) { image in
                    capturedImage = image
                    score(image)
                }
                .ignoresSafeArea()
            }
        }
    }

    /// 撮影・選択した画像をバックグラウンドで端末内のCore ML採点処理へ渡す。
    ///
    /// 処理フロー:
    /// 1. UIで受け取ったUIImageを向き補正済みのCGImageへ変換する。
    /// 2. メインスレッドで採点中表示に切り替え、古い結果やエラーをクリアする。
    /// 3. 重いCore ML/Vision処理はグローバルキューへ逃がし、画面操作を止めない。
    /// 4. ScoringEngineがQR確認、答案領域切り出し、設問検出、数字認識を順に実行する。
    /// 5. 結果またはエラーだけをメインスレッドへ戻し、SwiftUIの状態を更新する。
    @MainActor
    private func score(_ image: UIImage) {
        guard let normalizedImage = image.normalizedCGImage() else {
            errorMessage = "画像を読み取れませんでした。"
            return
        }

        isScoring = true
        scores = nil
        answerPreviewImage = nil
        diagnosticMessage = nil
        errorMessage = nil

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let result = try ScoringEngine().score(image: normalizedImage)
                DispatchQueue.main.async {
                    scores = result.scores
                    answerPreviewImage = UIImage(cgImage: result.answerImage)
                    diagnosticMessage = result.diagnosticText
                    isScoring = false
                }
            } catch {
                DispatchQueue.main.async {
                    errorMessage = error.localizedDescription
                    isScoring = false
                }
            }
        }
    }
}

private extension UIImage {
    /// 画像の向きを反映したCGImageを作成する。
    func normalizedCGImage() -> CGImage? {
        let format = UIGraphicsImageRendererFormat.default()
        // 端末の画面倍率で勝手に拡大せず、写真本来のピクセル数を維持する。
        format.scale = scale
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }.cgImage
    }
}

#Preview {
    ContentView()
}
