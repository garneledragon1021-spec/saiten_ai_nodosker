import SwiftUI

/// カメラ映像を表示しながら、一定間隔で採点結果を画面上に重ねて表示する。
struct LiveScoringView: View {
    let onClose: () -> Void
    let onResult: (ScoringResult) -> Void

    @State private var latestResult: ScoringResult?
    @State private var latestErrorMessage: String?
    @State private var lastUpdatedAt: Date?

    private var total: Int? {
        latestResult?.scores.reduce(0, +)
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                LiveCameraView { result in
                    latestResult = result
                    latestErrorMessage = nil
                    lastUpdatedAt = Date()
                    onResult(result)
                } onError: { message in
                    latestErrorMessage = message
                }
                .ignoresSafeArea()

                VStack(spacing: 0) {
                    topBar
                        .padding(.top, max(geometry.safeAreaInsets.top, 12) + 6)
                        .padding(.horizontal, 16)

                    Spacer(minLength: 16)

                    resultPanel(maxHeight: min(320, geometry.size.height * 0.42))
                        .padding(.horizontal, 12)
                        .padding(.bottom, max(geometry.safeAreaInsets.bottom, 12) + 6)
                }
            }
            .background(.black)
        }
    }

    private var topBar: some View {
        HStack {
            Label("リアルタイム解析中", systemImage: "camera.viewfinder")
                .font(.headline)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())

            Spacer()

            Button {
                onClose()
            } label: {
                Image(systemName: "xmark")
                    .font(.headline)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
        }
        .foregroundStyle(.white)
    }

    private func resultPanel(maxHeight: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("合計点")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(total.map { "\($0) 点" } ?? "-- 点")
                        .font(.system(size: 44, weight: .bold, design: .rounded))
                        .contentTransition(.numericText())
                }
                Spacer()
                Text(lastUpdatedText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let latestResult {
                diagnosticChips(for: latestResult)

                ScrollView {
                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(), spacing: 8),
                            GridItem(.flexible(), spacing: 8),
                        ],
                        spacing: 8
                    ) {
                        ForEach(Array(latestResult.scores.enumerated()), id: \.offset) { index, score in
                            scoreCell(index: index, score: score)
                        }
                    }
                }
                .scrollIndicators(.hidden)
                .frame(maxHeight: max(76, maxHeight - 150))
            } else if let latestErrorMessage {
                Label(latestErrorMessage, systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            } else {
                Text("答案全体とQRコードを画面内に入れてください。検出できると自動で合計点を表示します。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(maxHeight: maxHeight)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .foregroundStyle(.primary)
    }

    private func diagnosticChips(for result: ScoringResult) -> some View {
        HStack(spacing: 8) {
            chip("設問 \(result.questionCount)")
            chip("数字 \(result.digitTotal)")
            chip("信頼度 \(averageConfidencePercent(for: result))%")
        }
        .font(.caption)
    }

    private func chip(_ text: String) -> some View {
        Text(text)
            .lineLimit(1)
            .minimumScaleFactor(0.8)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(.black.opacity(0.12), in: Capsule())
    }

    private func scoreCell(index: Int, score: Int) -> some View {
        HStack {
            Text("Q\(index + 1)")
                .foregroundStyle(.secondary)
            Spacer(minLength: 6)
            Text("\(score)")
                .fontWeight(.semibold)
        }
        .font(.subheadline)
        .lineLimit(1)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.black.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    private func averageConfidencePercent(for result: ScoringResult) -> Int {
        guard result.digitTotal > 0 else {
            return 0
        }
        return Int((result.scoreConfidence / Float(result.digitTotal)) * 100)
    }

    private var lastUpdatedText: String {
        guard let lastUpdatedAt else {
            return "未検出"
        }
        let elapsed = max(0, Int(Date().timeIntervalSince(lastUpdatedAt)))
        return "\(elapsed)秒前"
    }
}
