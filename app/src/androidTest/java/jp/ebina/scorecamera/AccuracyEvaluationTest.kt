package jp.ebina.scorecamera

import android.graphics.BitmapFactory
import android.graphics.PointF
import android.graphics.ImageDecoder
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.common.InputImage
import java.util.concurrent.TimeUnit

/** Measures the unchanged production engine; inaccurate outputs are reported, not hidden as test failures. */
@RunWith(AndroidJUnit4::class)
class AccuracyEvaluationTest {
    private fun decodeCase(dir: File, case: JSONObject): android.graphics.Bitmap {
        val imageFile = File(dir, case.getString("file"))
        return if (case.has("max_side")) ImageDecoder.decodeBitmap(ImageDecoder.createSource(imageFile)) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val longest = maxOf(info.size.width, info.size.height)
            val limit = case.getInt("max_side")
            if (longest > limit) {
                val scale = limit.toFloat() / longest
                decoder.setTargetSize((info.size.width * scale).toInt(), (info.size.height * scale).toInt())
            }
        } else BitmapFactory.decodeFile(imageFile.absolutePath)
    }
    @Test fun traceObservedFailures() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(context.getExternalFilesDir(null), "accuracy-evaluation")
        assumeTrue(File(dir, "manifest.json").isFile)
        val cases = JSONArray(File(dir, "manifest.json").readText())
        val selected = InstrumentationRegistry.getArguments().getString("trace_ids")?.split(",")
            ?: listOf("row17-baseline", "row17-low_resolution")
        val scanner = BarcodeScanning.getClient()
        try {
            File(dir, "failure-traces.jsonl").bufferedWriter().use { writer ->
                for (i in 0 until cases.length()) {
                    val case = cases.getJSONObject(i)
                    if (case.getString("id") !in selected) continue
                    val image = decodeCase(dir, case)
                    try {
                        val qr = Tasks.await(scanner.process(InputImage.fromBitmap(image, 0)), 15, TimeUnit.SECONDS)
                            .first { it.rawValue == "fujishima startup QRcode" }.cornerPoints!!
                        val trace = JSONArray()
                        val result = try { ScoringEngine.get(context).scoreWithQr(image,
                            qr.map { PointF(it.x.toFloat(), it.y.toFloat()) }, case.getJSONArray("expected").length(),
                            trace = { trace.put(it) }) } catch (error: Exception) {
                                case.put("trace", trace); case.put("error", error.message)
                                writer.appendLine(case.toString()); writer.flush(); continue
                            }
                        case.put("trace", trace)
                        case.put("scores", JSONArray().apply { result.scores.forEach { put(it ?: JSONObject.NULL) } })
                        writer.appendLine(case.toString()); writer.flush()
                        result.answerImage.recycle()
                    } finally { image.recycle() }
                }
            }
        } finally { scanner.close() }
    }
    @Test fun evaluateFrozenCases() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(context.getExternalFilesDir(null), "accuracy-evaluation")
        assumeTrue("Prepare and transfer the evaluation fixture before running the accuracy benchmark", File(dir, "manifest.json").isFile)
        val cases = JSONArray(File(dir, "manifest.json").readText())
        val engine = ScoringEngine.get(context)
        val output = File(dir, "results.jsonl")
        assertTrue(cases.length() > 0)
        output.bufferedWriter().use { writer ->
            for (i in 0 until cases.length()) {
                val case = cases.getJSONObject(i)
                val row = JSONObject(case.toString())
                val image = decodeCase(dir, case)
                val started = SystemClock.elapsedRealtime()
                try {
                    val count = case.getJSONArray("expected").length()
                    val result = if (case.has("qr")) {
                        val qr = case.getJSONArray("qr")
                        engine.scoreWithQr(image, (0 until qr.length()).map {
                            val p = qr.getJSONArray(it); PointF(p.getDouble(0).toFloat(), p.getDouble(1).toFloat())
                        }, count)
                    } else engine.score(image, count)
                    row.put("scores", JSONArray().apply { result.scores.forEach { put(it ?: JSONObject.NULL) } })
                    row.put("total", result.total ?: JSONObject.NULL)
                    row.put("rotation", result.rotationDegrees)
                    row.put("diagnostic", result.diagnosticText)
                    result.answerImage.recycle()
                } catch (error: Exception) {
                    row.put("error", "${error.javaClass.simpleName}: ${error.message}")
                } finally {
                    image.recycle()
                    row.put("elapsed_ms", SystemClock.elapsedRealtime() - started)
                }
                writer.appendLine(row.toString()); writer.flush()
                android.util.Log.i("AccuracyEvaluation", "${i+1}/${cases.length()} ${row.getString("id")}")
            }
        }
    }
}
