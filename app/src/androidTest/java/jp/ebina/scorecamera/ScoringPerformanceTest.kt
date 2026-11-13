package jp.ebina.scorecamera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Same inputs and engine API can run against both installed 1.4 and 1.5 APKs. */
@RunWith(AndroidJUnit4::class)
class ScoringPerformanceTest {
    @Test fun benchmarkWarmEngineOnWrittenAndBlankSheets() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val engine = ScoringEngine.get(instrumentation.targetContext)
        val sample = instrumentation.context.assets.open("sample-answer.jpg").use { BitmapFactory.decodeStream(it) }
        val blank = Bitmap.createBitmap(800, 300, Bitmap.Config.ARGB_8888)
        Canvas(blank).apply {
            drawColor(Color.WHITE)
            val pen = Paint().apply { color = Color.BLACK; style = Paint.Style.STROKE; strokeWidth = 3f }
            for (i in 0..3) drawRect(130f + 145 * i, 130f, 240f + 145 * i, 240f, pen)
        }
        val corners = listOf(PointF(20f, 20f), PointF(80f, 20f), PointF(80f, 80f), PointF(20f, 80f))
        val output = JSONObject()
        try {
            for (name in listOf("written", "blank")) {
                val times = JSONArray()
                for (iteration in 0..5) {
                    val started = SystemClock.elapsedRealtime()
                    val result = if (name == "written") engine.score(sample, 3) else engine.scoreWithQr(blank, corners, 4)
                    val elapsed = SystemClock.elapsedRealtime() - started
                    try {
                        assertEquals(if (name == "written") listOf(12, 95, 84) else listOf(null, null, null, null), result.scores)
                        if (iteration > 0) times.put(elapsed) // Exclude initialization/warm-up.
                    } finally { result.answerImage.recycle() }
                }
                output.put(name, times)
            }
            File(instrumentation.targetContext.getExternalFilesDir(null), "performance.json").writeText(output.toString(2))
            android.util.Log.i("ScorePerformance", output.toString())
        } finally { sample.recycle(); blank.recycle() }
    }
}
