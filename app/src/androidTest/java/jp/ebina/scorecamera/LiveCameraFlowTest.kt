package jp.ebina.scorecamera

import android.content.Intent
import android.graphics.BitmapFactory
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LiveCameraFlowTest {
    @Test fun provisionalConfirmationAndNextPaperAreDisplayedCorrectly() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, "android.permission.CAMERA")
        val activity = instrumentation.startActivitySync(
            Intent(context, LiveCameraActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as LiveCameraActivity
        try {
            instrumentation.runOnMainSync {
                val label = activity.findViewById<TextView>(R.id.liveTotalLabel)
                val total = activity.findViewById<TextView>(R.id.liveTotal)
                val next = activity.findViewById<android.view.View>(R.id.nextAnswerButton)
                val scores = listOf(6, 13, 42, 10)
                activity.showReading(0, scores)
                assertEquals("71 点", total.text.toString())
                assertEquals("暫定合計（確認中）", label.text.toString())
                activity.showReading(0, scores)
                assertEquals("合計点（確認済み）", label.text.toString())
                activity.showReading(0, listOf(0, 0, 0, 0))
                assertEquals("71 点", total.text.toString())
                assertTrue(next.isEnabled)
                next.performClick()
                assertEquals("-- 点", total.text.toString())
                assertEquals("合計点（読取待ち）", label.text.toString())
                activity.showReading(0, scores) // Late result from the preceding paper.
                assertEquals("-- 点", total.text.toString())
                activity.showReading(1, listOf(8, 20, 27, 18))
                assertEquals("73 点", total.text.toString())
                assertEquals("暫定合計（確認中）", label.text.toString())
                activity.showReading(1, listOf(8, null, 27, 18))
                assertEquals("要確認", total.text.toString())
                assertEquals("合計点（未読欄あり）", label.text.toString())
            }
        } finally { instrumentation.runOnMainSync { activity.finish() } }
    }

    @Test fun bothScoringScreensKeepTheDisplayOn() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, "android.permission.CAMERA")
        for (screen in listOf(MainActivity::class.java, LiveCameraActivity::class.java)) {
            val activity = instrumentation.startActivitySync(Intent(context, screen).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            try {
                instrumentation.runOnMainSync {
                    assertTrue(activity.window.attributes.flags and android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON != 0)
                    assertFalse(activity.isFinishing)
                }
            } finally { instrumentation.runOnMainSync { activity.finish() } }
        }
    }

    @Test fun switchingPapersStopsBeforeReadingTheOldPapersRemainingFields() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val image = instrumentation.context.assets.open("sample-answer.jpg").use { BitmapFactory.decodeStream(it) }
        var checks = 0
        try {
            ScoringEngine.get(instrumentation.targetContext).score(image, 3) { ++checks >= 5 }
            fail("Expected cancellation before the second score field")
        } catch (_: java.util.concurrent.CancellationException) {
            assertEquals(5, checks)
            assertFalse(image.isRecycled)
        } finally { image.recycle() }
    }
}
