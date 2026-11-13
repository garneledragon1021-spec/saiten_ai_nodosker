package jp.ebina.scorecamera

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.abs

internal data class InkReading(val value: Int?, val count: Int, val confidence: Float, val reliableCount: Boolean, val consistentMasks: Boolean = false)

/** Counts visible glyphs before interpreting them: missing/uncertain glyphs invalidate the entire field. */
internal class InkDigitRecognizer(context: Context) {
    private val environment = OrtEnvironment.getEnvironment()
    private val session by lazy {
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(2)
            environment.createSession(context.assets.open("handwritten_digits_v1.onnx").use { it.readBytes() }, options)
        }
    }

    fun read(bitmap: Bitmap, trace: ((String) -> Unit)? = null, preserveEdgeInk: Boolean = false): InkReading {
        val rgba = Mat(); val gray = Mat()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            val readings = (0..2).map { method ->
                val mask = Mat()
                try {
                    if (method == 0) Imgproc.threshold(gray, mask, 0.0, 255.0, Imgproc.THRESH_BINARY_INV or Imgproc.THRESH_OTSU)
                    else Imgproc.adaptiveThreshold(gray, mask, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 31, if (method == 1) 9.0 else 5.0)
                    if (preserveEdgeInk) {
                        removeBorderLines(mask)
                        val padded = Mat()
                        try {
                            Core.copyMakeBorder(mask, padded, 4, 4, 4, 4, Core.BORDER_CONSTANT, org.opencv.core.Scalar(0.0))
                            padded.copyTo(mask)
                        } finally { padded.release() }
                    }
                    // Suppress a thin remainder of the printed frame, never a percentage of a digit's own bounding box.
                    mask.row(0).useMat { it.setTo(org.opencv.core.Scalar(0.0)) }
                    mask.row(mask.rows() - 1).useMat { it.setTo(org.opencv.core.Scalar(0.0)) }
                    mask.col(0).useMat { it.setTo(org.opencv.core.Scalar(0.0)) }
                    mask.col(mask.cols() - 1).useMat { it.setTo(org.opencv.core.Scalar(0.0)) }
                    val boxes = glyphBoxes(mask)
                    trace?.invoke("mask=$method size=${mask.cols()}x${mask.rows()} glyphs=$boxes")
                    if (boxes.size !in 1..3) InkReading(null, boxes.size, 0f, true)
                    else {
                        val predictions = boxes.map { classify(mask, it) }
                        val minimum = predictions.minOf { it.second }
                        val value = if (minimum >= .80f) predictions.fold(0) { total, prediction -> total * 10 + prediction.first } else null
                        InkReading(value, boxes.size, minimum, true)
                    }
                } finally { mask.release() }
            }
            return InkConsensus.combine(readings)
        } finally { rgba.release(); gray.release() }
    }

    private fun removeBorderLines(mask: Mat) {
        val horizontal = Mat(); val vertical = Mat()
        val hk = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(max(8.0, mask.cols() * .65), 1.0))
        val vk = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, max(8.0, mask.rows() * .65)))
        try {
            Imgproc.morphologyEx(mask, horizontal, Imgproc.MORPH_OPEN, hk)
            Imgproc.morphologyEx(mask, vertical, Imgproc.MORPH_OPEN, vk)
            val values = ByteArray(mask.rows() * mask.cols()); mask.get(0, 0, values)
            val hs = ByteArray(values.size); horizontal.get(0, 0, hs)
            val vs = ByteArray(values.size); vertical.get(0, 0, vs)
            val band = max(2, (min(mask.cols(), mask.rows()) * .08).toInt())
            for (y in 0 until mask.rows()) for (x in 0 until mask.cols()) {
                val i = y * mask.cols() + x
                if (((y < band || y >= mask.rows() - band) && hs[i].toInt() != 0) ||
                    ((x < band || x >= mask.cols() - band) && vs[i].toInt() != 0)) values[i] = 0
            }
            mask.put(0, 0, values)
        } finally { horizontal.release(); vertical.release(); hk.release(); vk.release() }
    }

    private fun glyphBoxes(mask: Mat): List<Rect> {
        val labels = Mat(); val stats = Mat(); val centroids = Mat()
        try {
            val count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids, 8, CvType.CV_32S)
            val boxes = (1 until count).mapNotNull { index ->
                val x = stats.get(index, Imgproc.CC_STAT_LEFT)[0].toInt()
                val y = stats.get(index, Imgproc.CC_STAT_TOP)[0].toInt()
                val w = stats.get(index, Imgproc.CC_STAT_WIDTH)[0].toInt()
                val h = stats.get(index, Imgproc.CC_STAT_HEIGHT)[0].toInt()
                val area = stats.get(index, Imgproc.CC_STAT_AREA)[0]
                if (area < max(3.0, mask.total() * .0015) || h < mask.rows() * .18 || w > mask.cols() * .95 ||
                    x <= 1 || y <= 1 || x + w >= mask.cols() - 1 || y + h >= mask.rows() - 1) null
                else Rect(x, y, w, h)
            }.sortedBy { it.x }.toMutableList()
            // Join detached strokes with overlapping x ranges (e.g. a broken handwritten 4).
            var index = 0
            while (index < boxes.size - 1) {
                val a = boxes[index]; val b = boxes[index + 1]
                val overlap = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
                if (overlap > min(a.width, b.width) * .20) {
                    val left = min(a.x, b.x); val top = min(a.y, b.y)
                    boxes[index] = Rect(left, top, max(a.x + a.width, b.x + b.width) - left, max(a.y + a.height, b.y + b.height) - top)
                    boxes.removeAt(index + 1)
                } else index++
            }
            // Parentheses are not digits. Only discard a curved, symmetric edge stroke, not a thin straight 1.
            return boxes.filterIndexed { i, box ->
                !((i == 0 || i == boxes.lastIndex) && boxes.size > 1 && isParenthesis(mask, box))
            }
        } finally { labels.release(); stats.release(); centroids.release() }
    }

    private fun isParenthesis(mask: Mat, box: Rect): Boolean {
        if (box.width > box.height * .38 || box.width < 3) return false
        fun middleX(from: Float, to: Float): Double {
            val xs = mutableListOf<Int>()
            for (y in box.y + (box.height * from).toInt() until box.y + (box.height * to).toInt())
                for (x in box.x until box.x + box.width) if (mask.get(y, x)[0] > 0) xs += x
            return if (xs.isEmpty()) box.x.toDouble() else xs.average()
        }
        val top = middleX(0f, .2f); val mid = middleX(.4f, .6f); val bottom = middleX(.8f, 1f)
        return abs(top - bottom) < box.width * .35 && abs((top + bottom) / 2 - mid) > box.width * .35
    }

    private fun classify(mask: Mat, box: Rect): Pair<Int, Float> {
        val glyph = mask.submat(box); val resized = Mat(); val canvas = Mat.zeros(28, 28, CvType.CV_8UC1)
        val centered = Mat()
        try {
            val scale = 20.0 / max(box.width, box.height)
            val w = max(1, (box.width * scale).toInt()); val h = max(1, (box.height * scale).toInt())
            Imgproc.resize(glyph, resized, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            canvas.submat(Rect((28 - w) / 2, (28 - h) / 2, w, h)).useMat { resized.copyTo(it) }
            val moments = Imgproc.moments(canvas)
            val matrix = Mat.zeros(2, 3, CvType.CV_64F)
            try {
                matrix.put(0, 0, 1.0, 0.0, 13.5 - moments.m10 / max(1.0, moments.m00))
                matrix.put(1, 0, 0.0, 1.0, 13.5 - moments.m01 / max(1.0, moments.m00))
                Imgproc.warpAffine(canvas, centered, matrix, Size(28.0, 28.0))
            } finally { matrix.release() }
            // MNIST strokes are often thicker than pencil strokes. Compare slight thickening,
            // but never silently choose between two different high-confidence digit identities.
            val predictions = mutableListOf(predict(centered))
            val thickened = Mat()
            val kernels = listOf(Mat.ones(2, 2, CvType.CV_8U), Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0)))
            try {
                for (kernel in kernels) {
                    Imgproc.dilate(centered, thickened, kernel)
                    predictions += predict(thickened)
                }
            } finally { thickened.release(); kernels.forEach { it.release() } }
            if (predictions.filter { it.second >= .80f }.map { it.first }.distinct().size > 1) return 0 to 0f
            return predictions.maxBy { it.second }
        } finally { glyph.release(); resized.release(); canvas.release(); centered.release() }
    }

    private fun predict(image: Mat): Pair<Int, Float> {
        val bytes = ByteArray(28 * 28); image.get(0, 0, bytes)
        val values = FloatArray(bytes.size) { (bytes[it].toInt() and 255) / 255f }
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), longArrayOf(1, 1, 28, 28)).use { tensor ->
            session.run(mapOf("image" to tensor)).use { result ->
                val logits = FloatArray(10); (result[0] as OnnxTensor).floatBuffer.get(logits)
                val peak = logits.max(); val weights = logits.map { exp((it - peak).toDouble()) }
                val digit = weights.indices.maxBy { weights[it] }
                return digit to (weights[digit] / weights.sum()).toFloat()
            }
        }
    }
}

private inline fun <T> Mat.useMat(block: (Mat) -> T): T = try { block(this) } finally { release() }
