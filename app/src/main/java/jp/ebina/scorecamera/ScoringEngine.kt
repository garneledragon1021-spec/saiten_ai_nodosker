package jp.ebina.scorecamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.PointF
import android.graphics.Matrix
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.imgproc.Imgproc
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil

data class ScoringResult(
    val scores: List<Int?>,
    val questionCount: Int,
    val digitCount: Int,
    val confidenceTotal: Float,
    val answerImage: Bitmap,
    val rotationDegrees: Int = 0,
    val usedFrameDetection: Boolean = false,
    val ocrFieldCount: Int = 0,
    val processingMs: Long = 0,
) {
    val total: Int? get() = if (scores.all { it != null }) scores.filterNotNull().sum() else null
    val diagnosticText: String
        get() {
            val unread = scores.count { it == null }
            return "得点枠: ${questionCount}個 / 数字: ${digitCount}個 / 向き・枠の傾き・明るさ補正あり / ${processingMs / 1000.0}秒" +
                if (unread > 0) " / ${unread}欄は読取不能（合計未確定）" else ""
        }
}

class ScoringEngine(private val context: Context) {
    private val environment: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private val digitModel: OrtSession by lazy {
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(2)
            environment.createSession(assetFilePath("score_detector.onnx"), options)
        }
    }
    private val manualDigitModel: OrtSession by lazy {
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(2)
            environment.createSession(assetFilePath("score_detector_manual_v1.onnx"), options)
        }
    }
    private val textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val inkRecognizer by lazy { InkDigitRecognizer(context.applicationContext) }
    private val qrScanner = BarcodeScanning.getClient(
        BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE).build()
    )

    @Synchronized
    fun score(source: Bitmap, expectedQuestionCount: Int? = null, isCancelled: () -> Boolean = { false }): ScoringResult {
        checkCancelled(isCancelled)
        val started = android.os.SystemClock.elapsedRealtime()
        val barcodes = Tasks.await(qrScanner.process(InputImage.fromBitmap(source, 0)), 15, TimeUnit.SECONDS)
        checkCancelled(isCancelled)
        val qr = barcodes.firstOrNull { it.rawValue == EXPECTED_QR }
            ?: throw ScoringException("指定QRを読み取れません。QRを大きく写し、ピントを合わせてください。")
        val corners = qr.cornerPoints?.map { PointF(it.x.toFloat(), it.y.toFloat()) }
            ?: throw ScoringException("QRコードの四隅を取得できませんでした。")
        return scoreWithQr(source, corners, expectedQuestionCount, isCancelled = isCancelled).copy(processingMs = android.os.SystemClock.elapsedRealtime() - started)
    }

    /** QR locations are injectable for regression tests that isolate cropping from decoding. */
    @Synchronized
    internal fun scoreWithQr(source: Bitmap, corners: List<PointF>, expectedQuestionCount: Int? = null,
        trace: ((String) -> Unit)? = null, isCancelled: () -> Boolean = { false }): ScoringResult {
        checkCancelled(isCancelled)
        require(corners.size == 4)
        val edge = corners[1]
        var tilt = Math.toDegrees(atan2((edge.y - corners[0].y).toDouble(), (edge.x - corners[0].x).toDouble())).toFloat()
        while (tilt > 45f) tilt -= 90f
        while (tilt < -45f) tilt += 90f
        var best: AnswerCandidate? = null
        var foundCount = 0
        fun quality(candidate: AnswerCandidate): Float {
            val heights = candidate.frames.map { it.height() }.average()
            val spread = candidate.frames.maxOf { (it.top + it.bottom) / 2 } - candidate.frames.minOf { (it.top + it.bottom) / 2 }
            return candidate.frames.size * 100f - spread / heights.toFloat().coerceAtLeast(1f)
        }
        for (rotation in listOf(0, 90, 180, 270)) {
            val matrix = Matrix().apply { setRotate(rotation - tilt) }
            val bounds = RectF(0f, 0f, source.width.toFloat(), source.height.toFloat())
            matrix.mapRect(bounds)
            matrix.postTranslate(-bounds.left, -bounds.top)
            val normalized = Bitmap.createBitmap(ceil(bounds.width()).toInt(), ceil(bounds.height()).toInt(), Bitmap.Config.ARGB_8888)
            Canvas(normalized).apply { drawColor(Color.WHITE); drawBitmap(source, matrix, Paint(Paint.FILTER_BITMAP_FLAG)) }
            val points = corners.flatMap { listOf(it.x, it.y) }.toFloatArray()
            matrix.mapPoints(points)
            val xs = (0..3).map { points[it * 2] }
            val ys = (0..3).map { points[it * 2 + 1] }
            val qr = RectF(xs.min(), ys.min(), xs.max(), ys.max())
            val left = (qr.left - qr.width() * .15f).toInt().coerceAtLeast(0)
            val top = qr.bottom.toInt().coerceAtLeast(0)
            if (top >= normalized.height - 10 || left >= normalized.width - 10) { normalized.recycle(); continue }
            val answer = Bitmap.createBitmap(normalized, left, top, normalized.width - left, normalized.height - top)
            if (answer !== normalized) normalized.recycle()
            val frames = detectFrames(answer, qr.height())
            trace?.invoke("rotation=$rotation frames=${frames.map { listOf(it.left, it.top, it.right, it.bottom, it.confidence) }}")
            foundCount = max(foundCount, frames.size)
            if (frames.size >= 2 && (expectedQuestionCount == null || frames.size == expectedQuestionCount)) {
                val candidate = AnswerCandidate(answer, frames, rotation, true)
                val previous = best
                if (previous == null || quality(candidate) > quality(previous)) {
                    previous?.image?.recycle()
                    best = candidate
                } else answer.recycle()
            } else answer.recycle()
        }
        val chosen = best ?: throw ScoringException(if (foundCount >= 2 && expectedQuestionCount != null)
            "得点枠は${foundCount}個です。指定の${expectedQuestionCount}個すべてが写るよう撮り直すか、得点欄数を確認してください。"
            else "QRは確認できましたが、横に並ぶ得点枠が見つかりません。得点欄全体を写してください。")
        return scoreCandidate(chosen, trace, isCancelled)
    }

    private fun checkCancelled(isCancelled: () -> Boolean) {
        if (isCancelled()) throw java.util.concurrent.CancellationException("答案の読取を切り替えました。")
    }

    private fun scoreCandidate(candidate: AnswerCandidate, trace: ((String) -> Unit)?, isCancelled: () -> Boolean): ScoringResult {
        val answer = candidate.image
        try {
        val questions = candidate.frames.sortedBy { it.left }
        val scores = mutableListOf<Int?>()
        var digitCount = 0
        var confidenceTotal = 0f
        var ocrFields = 0
        questions.forEach { detection ->
            // Finish the current native call/field safely; do not process the old paper's remaining fields.
            checkCancelled(isCancelled)
            val expanded = detection.expanded(.10f, .18f, answer.width, answer.height)
            val questionImage = Bitmap.createBitmap(answer, expanded.left, expanded.top, expanded.width(), expanded.height())
            val field = FieldPreprocessor.rectify(answer, Rect(detection.left, detection.top, detection.right, detection.bottom), detection.corners)
            val prepared = FieldPreprocessor.prepare(field)
            val normalized = prepared.bitmap
            trace?.invoke("frame=${detection.corners}")
            val ink = inkRecognizer.read(normalized, trace, prepared.colorSeparated)
            // With no neutral-color glyph, every resolver path below necessarily returns null.
            // Keep the color-separated grayscale recovery path unchanged (it can find missed ink).
            if (ink.count == 0 && !prepared.colorSeparated) {
                scores += null
                trace?.invoke("field=${scores.size}, blank-neutral: skip whole-field recognition")
                questionImage.recycle()
                normalized.recycle(); if (field !== answer) field.recycle()
                return@forEach
            }
            // Use full-field detectors as corroboration; a confident subset is still a missing-digit failure.
            val digits = bestDigits(questionImage)
            val neuralScore = if (digits.isEmpty()) null else digits.fold(0) { value, digit -> value * 10 + digit.classIndex }
            val ocrScore = readTextScore(normalized)
            val neuralConfidence = digits.minOfOrNull { it.confidence } ?: 0f
            var score = ScoreResolver.resolve(ink, neuralScore, digits.size, neuralConfidence, ocrScore)
            if (prepared.colorSeparated && score != null && score != neuralScore && score != ocrScore) score = null
            if (score == null) {
                // Retry only ambiguous fields, keeping the usual g06 CPU cost unchanged for clear readings.
                val alternate = bestDigits(field)
                val value = if (alternate.isEmpty()) null else alternate.fold(0) { total, digit -> total * 10 + digit.classIndex }
                score = ScoreResolver.recoverFromAlternate(ink, digits.size, neuralConfidence,
                    value, alternate.size, alternate.minOfOrNull { it.confidence } ?: 0f, ocrScore)
                trace?.invoke("alternate=$value/${alternate.size}, selected=$score")
                if (score == null) {
                    // The original manually annotated model is complementary, not an unconditional override.
                    val legacy = bestDigits(questionImage, manualDigitModel)
                    val legacyValue = if (legacy.isEmpty()) null else legacy.fold(0) { total, digit -> total * 10 + digit.classIndex }
                    val observedCount = maxOf(if (neuralConfidence >= .45f) digits.size else 0,
                        if ((alternate.minOfOrNull { it.confidence } ?: 0f) >= .45f) alternate.size else 0)
                    score = ScoreResolver.recoverFromAlternate(ink, observedCount, if (observedCount > 0) 1f else 0f,
                        legacyValue, legacy.size, legacy.minOfOrNull { it.confidence } ?: 0f, ocrScore)
                    trace?.invoke("manual-model=$legacyValue/${legacy.size}, selected=$score")
                    if (score == null && prepared.colorSeparated) {
                        val gray = FieldPreprocessor.prepare(field, allowColorSeparation = false)
                        try {
                            val grayInk = inkRecognizer.read(gray.bitmap, trace)
                            val grayOcr = readTextScore(gray.bitmap)
                            val readers = listOf(digits, alternate, legacy)
                            val maximumCount = readers.filter { (it.minOfOrNull { d -> d.confidence } ?: 0f) >= .45f }
                                .maxOfOrNull { it.size } ?: 0
                            val recovered = readers.mapNotNull { reading ->
                                val v = if (reading.isEmpty()) null else reading.fold(0) { total, d -> total * 10 + d.classIndex }
                                ScoreResolver.recoverFromPreprocessing(ink, grayInk, maximumCount,
                                    v, reading.size, reading.minOfOrNull { it.confidence } ?: 0f, grayOcr)
                            }.distinct()
                            score = recovered.singleOrNull()
                            trace?.invoke("grayscale=$grayInk ocr=$grayOcr candidates=$recovered selected=$score")
                        } finally { gray.bitmap.recycle() }
                    }
                }
            }
            trace?.invoke("field=${scores.size + 1}, ink=$ink, neural=$neuralScore/${digits.size}, ocr=$ocrScore, selected=$score")
            if (neuralScore == null && ocrScore != null) ocrFields++
            scores += score
            digitCount += score?.toString()?.length ?: 0
            confidenceTotal += digits.sumOf { it.confidence.toDouble() }.toFloat()
            questionImage.recycle()
            normalized.recycle(); if (field !== answer) field.recycle()
        }
        checkCancelled(isCancelled)
        // Show the compact scoring strip, not the entire remaining page/background.
        val bottom = (questions.maxOf { it.bottom } + questions.maxOf { it.height() } * .12f).toInt().coerceAtMost(answer.height)
        val right = (questions.maxOf { it.right } + questions.maxOf { it.width() } * .12f).toInt().coerceAtMost(answer.width)
        val preview = Bitmap.createBitmap(answer, 0, 0, right, bottom)
        if (preview !== answer) answer.recycle()
        return ScoringResult(scores, questions.size, digitCount, confidenceTotal, preview, candidate.rotation, candidate.geometric, ocrFields)
        } catch (error: Exception) {
            if (!answer.isRecycled) answer.recycle()
            throw error
        }
    }

    private fun readTextScore(bitmap: Bitmap): Int? {
        val scale = 320f / max(bitmap.width, bitmap.height)
        val scaled = Bitmap.createBitmap((bitmap.width * scale).toInt() + 64, (bitmap.height * scale).toInt() + 64, Bitmap.Config.ARGB_8888)
        Canvas(scaled).apply {
            drawColor(Color.WHITE)
            drawBitmap(bitmap, null, RectF(32f, 32f, scaled.width - 32f, scaled.height - 32f), Paint(Paint.FILTER_BITMAP_FLAG))
        }
        try {
            val text = Tasks.await(textRecognizer.process(InputImage.fromBitmap(scaled, 0)), 10, TimeUnit.SECONDS)
            val numbers = text.textBlocks.flatMap { it.lines }.mapNotNull { line ->
                val value = line.text.replace(" ", "").trim('(', ')', '[', ']')
                val rect = line.boundingBox ?: return@mapNotNull null
                if (value.matches(Regex("[0-9]{1,3}")) && rect.height() > scaled.height * .20f &&
                    rect.centerY() in (scaled.height * .15f).toInt()..(scaled.height * .85f).toInt()) value.toInt() else null
            }
            return numbers.singleOrNull()
        } catch (_: java.util.concurrent.TimeoutException) {
            return null
        } finally { if (scaled !== bitmap) scaled.recycle() }
    }

    private fun detectFrames(bitmap: Bitmap, qrHeight: Float): List<Detection> {
        check(OpenCVLoader.initLocal()) { "得点枠検出を初期化できませんでした。" }
        val rgba = Mat(); val gray = Mat(); val mask = Mat(); val hierarchy = Mat()
        val contours = mutableListOf<MatOfPoint>()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            val grayPixels = ByteArray(gray.rows() * gray.cols())
            gray.get(0, 0, grayPixels)
            Imgproc.adaptiveThreshold(gray, mask, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 31, 8.0)
            Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            val frames = contours.mapNotNull { contour ->
                val rect = Imgproc.boundingRect(contour)
                val ratio = rect.width.toFloat() / rect.height.coerceAtLeast(1)
                if (rect.width < 12 || rect.height < 12 || rect.height < qrHeight * .32f || rect.height > qrHeight * 3.5f ||
                    ratio !in .50f..2.5f || rect.width > bitmap.width * .6f) return@mapNotNull null
                val curve = MatOfPoint2f(*contour.toArray()); val polygon = MatOfPoint2f()
                try {
                    Imgproc.approxPolyDP(curve, polygon, Imgproc.arcLength(curve, true) * .035, true)
                    if (polygon.total() != 4L || Imgproc.contourArea(contour) / (rect.width.toDouble() * rect.height) < .65) null
                    else {
                        val points = polygon.toArray()
                        // A pasted image's grey boundary is not the printed score box. Prefer the darkest actual perimeter.
                        var darkness = 0.0
                        for (side in 0..3) for (step in 0 until 16) {
                            val a = points[side]; val b = points[(side + 1) % 4]; val t = step / 16.0
                            val px = (a.x + (b.x - a.x) * t).toInt(); val py = (a.y + (b.y - a.y) * t).toInt()
                            var darkest = 255.0
                            for (dy in -2..2) for (dx in -2..2) {
                                val x = (px + dx).coerceIn(0, gray.cols() - 1); val y = (py + dy).coerceIn(0, gray.rows() - 1)
                                darkest = min(darkest, (grayPixels[y * gray.cols() + x].toInt() and 255).toDouble())
                            }
                            darkness += 255.0 - darkest
                        }
                        val borderConfidence = (darkness / (64 * 255)).toFloat()
                        if (borderConfidence < .2f) null else Detection(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, 0, borderConfidence,
                            points.map { PointF(it.x.toFloat(), it.y.toFloat()) })
                    }
                } finally { curve.release(); polygon.release() }
            }
            // A handwritten 0 may itself approximate a quadrilateral inside its real score frame.
            // IoU alone does not suppress a small contained shape; require containment in BOTH dimensions.
            val unnested = frames.filter { child -> frames.none { parent ->
                parent !== child && child.width() < parent.width() * .65f && child.height() < parent.height() * .65f &&
                    child.left >= parent.left && child.top >= parent.top && child.right <= parent.right && child.bottom <= parent.bottom &&
                    parent.confidence >= child.confidence
            } }
            val distinct = nonMaximumSuppression(unnested.sortedByDescending { it.width() * it.height() }, .35f)
            return distinct.map { anchor ->
                distinct.filter { frame ->
                    val centerDistance = abs((frame.top + frame.bottom) - (anchor.top + anchor.bottom)) / 2f
                    centerDistance < max(frame.height(), anchor.height()) * 1.1f &&
                        frame.height().toFloat() / anchor.height() in .5f..2f
                }
            }.filter { it.size >= 2 }.maxByOrNull { it.size * 1000 - it.minOf { frame -> frame.top } }?.sortedBy { it.left } ?: emptyList()
        } finally {
            contours.forEach { it.release() }; rgba.release(); gray.release(); mask.release(); hierarchy.release()
        }
    }

    private fun bestDigits(bitmap: Bitmap, model: OrtSession = digitModel): List<Detection> {
        val adjusted = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val contrast = 1.35f
        val offset = 128f * (1f - contrast) + 6f
        val colors = ColorMatrix(floatArrayOf(contrast,0f,0f,0f,offset, 0f,contrast,0f,0f,offset, 0f,0f,contrast,0f,offset, 0f,0f,0f,1f,0f))
        Canvas(adjusted).drawBitmap(bitmap, 0f, 0f, Paint().apply { colorFilter = ColorMatrixColorFilter(colors) })
        try {
            return listOf(bitmap, adjusted).map { image ->
                detectWithFallback(model, image, 10, floatArrayOf(.45f)).filter {
                    it.width().toFloat() / image.width in .035f.. .70f &&
                        it.height().toFloat() / image.height in .20f.. .95f &&
                        (it.left + it.right) / 2f in image.width * .08f..image.width * .92f
                }.sortedByDescending { it.confidence }.take(3).sortedBy { it.left }
            }.maxBy { digits -> digits.size * 2f + digits.sumOf { it.confidence.toDouble() }.toFloat() }
        } finally { adjusted.recycle() }
    }

    private fun detectWithFallback(
        module: OrtSession,
        bitmap: Bitmap,
        classCount: Int,
        thresholds: FloatArray,
    ): List<Detection> {
        val prepared = letterbox(bitmap)
        val inputValues = bitmapToNchw(prepared.bitmap).also { prepared.bitmap.recycle() }
        val inputTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(inputValues),
            longArrayOf(1, 3, MODEL_SIZE.toLong(), MODEL_SIZE.toLong()),
        )
        val (shape, values) = inputTensor.use { tensor ->
            module.run(mapOf(module.inputNames.first() to tensor)).use { results ->
                val output = results.get(module.outputNames.first()).orElseThrow {
                    ScoringException("採点モデルの出力を取得できませんでした。")
                } as OnnxTensor
                val outputShape = output.info.shape
                val outputValues = FloatArray(outputShape.fold(1L) { total, size -> total * size }.toInt())
                output.floatBuffer.get(outputValues)
                outputShape to outputValues
            }
        }
        if (shape.size != 3 || shape[0] != 1L) throw ScoringException("採点モデルの出力形式が不正です。")
        val channels = shape[1].toInt()
        val anchors = shape[2].toInt()
        if (channels < 4 + classCount) throw ScoringException("採点モデルのクラス数が不正です。")

        thresholds.forEach { threshold ->
            val candidates = ArrayList<Detection>()
            for (anchor in 0 until anchors) {
                var bestClass = 0
                var bestConfidence = values[4 * anchors + anchor]
                for (classIndex in 1 until classCount) {
                    val confidence = values[(4 + classIndex) * anchors + anchor]
                    if (confidence > bestConfidence) {
                        bestConfidence = confidence
                        bestClass = classIndex
                    }
                }
                if (bestConfidence < threshold) continue
                val centerX = values[anchor]
                val centerY = values[anchors + anchor]
                val width = values[2 * anchors + anchor]
                val height = values[3 * anchors + anchor]
                val modelLeft = centerX - width / 2f
                val modelTop = centerY - height / 2f
                candidates += prepared.toOriginal(modelLeft, modelTop, width, height, bestClass, bestConfidence)
            }
            val selected = nonMaximumSuppression(candidates, .45f, classCount > 1)
            if (selected.isNotEmpty()) return selected
        }
        return emptyList()
    }

    private fun bitmapToNchw(bitmap: Bitmap): FloatArray {
        val pixelCount = MODEL_SIZE * MODEL_SIZE
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)
        val values = FloatArray(pixelCount * 3)
        pixels.forEachIndexed { index, color ->
            values[index] = ((color shr 16) and 0xFF) / 255f
            values[pixelCount + index] = ((color shr 8) and 0xFF) / 255f
            values[pixelCount * 2 + index] = (color and 0xFF) / 255f
        }
        return values
    }

    private fun letterbox(source: Bitmap): LetterboxedBitmap {
        val scale = min(MODEL_SIZE.toFloat() / source.width, MODEL_SIZE.toFloat() / source.height)
        val width = max(1, (source.width * scale).toInt())
        val height = max(1, (source.height * scale).toInt())
        val left = (MODEL_SIZE - width) / 2f
        val top = (MODEL_SIZE - height) / 2f
        val output = Bitmap.createBitmap(MODEL_SIZE, MODEL_SIZE, Bitmap.Config.ARGB_8888)
        Canvas(output).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(source, null, android.graphics.RectF(left, top, left + width, top + height), Paint(Paint.FILTER_BITMAP_FLAG))
        }
        return LetterboxedBitmap(output, scale, left, top, source.width, source.height)
    }

    private fun nonMaximumSuppression(source: List<Detection>, threshold: Float, classAgnostic: Boolean = false): List<Detection> {
        val pending = source.sortedByDescending { it.confidence }.toMutableList()
        val selected = mutableListOf<Detection>()
        while (pending.isNotEmpty()) {
            val best = pending.removeAt(0)
            selected += best
            pending.removeAll { (classAgnostic || it.classIndex == best.classIndex) && intersectionOverUnion(it, best) > threshold }
        }
        return selected
    }

    private fun intersectionOverUnion(a: Detection, b: Detection): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0, right - left) * max(0, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0) 0f else intersection.toFloat() / union
    }

    private fun assetFilePath(name: String): String {
        val target = File(context.filesDir, name)
        if (!target.exists() || target.length() == 0L) {
            context.assets.open(name).use { input ->
                FileOutputStream(target).use { output -> input.copyTo(output) }
            }
        }
        return target.absolutePath
    }

    companion object {
        private const val MODEL_SIZE = 640
        private const val EXPECTED_QR = "fujishima startup QRcode"
        @Volatile private var shared: ScoringEngine? = null
        fun get(context: Context): ScoringEngine = shared ?: synchronized(this) {
            shared ?: ScoringEngine(context.applicationContext).also { shared = it }
        }
    }
}

private data class AnswerCandidate(val image: Bitmap, val frames: List<Detection>, val rotation: Int, val geometric: Boolean)

private data class LetterboxedBitmap(
    val bitmap: Bitmap,
    val scale: Float,
    val paddingLeft: Float,
    val paddingTop: Float,
    val originalWidth: Int,
    val originalHeight: Int,
) {
    fun toOriginal(left: Float, top: Float, width: Float, height: Float, classIndex: Int, confidence: Float): Detection {
        val originalLeft = ((left - paddingLeft) / scale).toInt().coerceIn(0, originalWidth - 1)
        val originalTop = ((top - paddingTop) / scale).toInt().coerceIn(0, originalHeight - 1)
        val originalRight = ((left + width - paddingLeft) / scale).toInt().coerceIn(originalLeft + 1, originalWidth)
        val originalBottom = ((top + height - paddingTop) / scale).toInt().coerceIn(originalTop + 1, originalHeight)
        return Detection(originalLeft, originalTop, originalRight, originalBottom, classIndex, confidence)
    }
}

private data class Detection(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    val classIndex: Int,
    val confidence: Float,
    val corners: List<PointF>? = null,
) {
    fun width() = right - left
    fun height() = bottom - top

    fun expanded(xRatio: Float, yRatio: Float, maxWidth: Int, maxHeight: Int): Rect {
        val dx = (width() * xRatio).toInt()
        val dy = (height() * yRatio).toInt()
        return Rect(
            (left - dx).coerceAtLeast(0),
            (top - dy).coerceAtLeast(0),
            (right + dx).coerceAtMost(maxWidth),
            (bottom + dy).coerceAtMost(maxHeight),
        )
    }
}

private class ScoringException(message: String) : Exception(message)
