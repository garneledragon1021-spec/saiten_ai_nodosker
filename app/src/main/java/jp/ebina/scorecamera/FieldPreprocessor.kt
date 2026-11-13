package jp.ebina.scorecamera

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/** Works on one score box at a time to keep the moto g06 memory footprint bounded. */
internal object FieldPreprocessor {
    data class Prepared(val bitmap: Bitmap, val colorSeparated: Boolean)
    fun rectify(source: Bitmap, bounds: Rect, corners: List<PointF>?): Bitmap {
        val crop = Bitmap.createBitmap(source, bounds.left, bounds.top, bounds.width(), bounds.height())
        if (corners == null || corners.size != 4) return crop
        val points = corners.map { Point((it.x - bounds.left).toDouble(), (it.y - bounds.top).toDouble()) }
        val ordered = listOf(points.minBy { it.x + it.y }, points.maxBy { it.x - it.y },
            points.maxBy { it.x + it.y }, points.minBy { it.x - it.y })
        fun distance(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)
        val width = max(distance(ordered[0], ordered[1]), distance(ordered[3], ordered[2])).toInt().coerceAtLeast(12)
        val height = max(distance(ordered[0], ordered[3]), distance(ordered[1], ordered[2])).toInt().coerceAtLeast(12)
        val ratio = min(384.0 / max(width, height), 1.0)
        val w = max(12, (width * ratio).toInt()); val h = max(12, (height * ratio).toInt())
        val input = Mat(); val output = Mat()
        val from = MatOfPoint2f(*ordered.toTypedArray())
        val to = MatOfPoint2f(Point(0.0, 0.0), Point(w - 1.0, 0.0), Point(w - 1.0, h - 1.0), Point(0.0, h - 1.0))
        val transform = Imgproc.getPerspectiveTransform(from, to)
        try {
            Utils.bitmapToMat(crop, input)
            Imgproc.warpPerspective(input, output, transform, Size(w.toDouble(), h.toDouble()), Imgproc.INTER_LINEAR)
            return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(output, it) }
        } finally {
            input.release(); output.release(); from.release(); to.release(); transform.release()
            if (crop !== source) crop.recycle()
        }
    }

    fun normalize(frame: Bitmap): Bitmap = prepare(frame).bitmap

    fun prepare(frame: Bitmap, allowColorSeparation: Boolean = true): Prepared {
        // Keep handwriting near the printed frame; the previous 7.5% trim cut off real digits.
        val x = max(1, (frame.width * .025f).toInt())
        val y = max(1, (frame.height * .025f).toInt())
        var inner = Bitmap.createBitmap(frame, x, y, frame.width - 2 * x, frame.height - 2 * y)
        val rgba = Mat(); val gray = Mat(); val background = Mat(); val normalized = Mat(); val enhanced = Mat()
        val clahe = Imgproc.createCLAHE(2.0, Size(4.0, 4.0))
        try {
            val pixels = IntArray(inner.width * inner.height)
            inner.getPixels(pixels, 0, inner.width, 0, 0, inner.width, inner.height)
            val redPixels = pixels.count { pixel ->
                val r = (pixel shr 16) and 255; val g = (pixel shr 8) and 255; val b = pixel and 255
                r - max(g, b) > max(12, r / 10)
            }
            val redInk = allowColorSeparation && redPixels > pixels.size * .002 && redPixels < pixels.size * .20
            if (redInk) {
                // Chromatic ink survives even when it touches a black frame. Neutral pencil uses the grayscale path.
                val signal = ByteArray(pixels.size) { i ->
                    val p = pixels[i]; val r = (p shr 16) and 255; val g = (p shr 8) and 255; val b = p and 255
                    (255 - ((r - max(g, b) - 8).coerceAtLeast(0) * 4).coerceAtMost(255)).toByte()
                }
                gray.create(inner.height, inner.width, CvType.CV_8UC1); gray.put(0, 0, signal)
            } else {
                // Preserve the already-validated neutral-pencil path; chromatic separation is optional.
                if (inner !== frame) inner.recycle()
                val gx = max(2, (frame.width * .075f).toInt()); val gy = max(2, (frame.height * .075f).toInt())
                inner = Bitmap.createBitmap(frame, gx, gy, frame.width - 2 * gx, frame.height - 2 * gy)
                Utils.bitmapToMat(inner, rgba)
                Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            }
            // Estimate illumination locally; division removes broad shadows without thresholding away faint strokes.
            val kernel = max(15, (min(gray.cols(), gray.rows()) * 2 / 3) or 1)
            Imgproc.GaussianBlur(gray, background, Size(kernel.toDouble(), kernel.toDouble()), 0.0)
            Core.divide(gray, background, normalized, 245.0, CvType.CV_8U)
            clahe.apply(normalized, enhanced)
            // Pad the foreground mask AFTER thresholding, not this grayscale image:
            // a white image border can otherwise make Otsu mistake the entire paper for ink.
            return Prepared(Bitmap.createBitmap(inner.width, inner.height, Bitmap.Config.ARGB_8888).also {
                Utils.matToBitmap(enhanced, it)
            }, redInk)
        } finally {
            rgba.release(); gray.release(); background.release(); normalized.release(); enhanced.release(); clahe.collectGarbage()
            if (inner !== frame) inner.recycle()
        }
    }
}
