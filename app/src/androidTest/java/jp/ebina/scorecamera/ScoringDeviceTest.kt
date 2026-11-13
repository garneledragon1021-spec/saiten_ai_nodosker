package jp.ebina.scorecamera

import android.graphics.BitmapFactory
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Shader
import android.content.Intent
import android.content.pm.PackageManager
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Exercises the bundled QR reader and both native ONNX models on Android. */
@RunWith(AndroidJUnit4::class)
class ScoringDeviceTest {
    @Test
    fun aNestedZeroLikeContourIsNotAnExtraScoreFrame() {
        val image = Bitmap.createBitmap(800,300,Bitmap.Config.ARGB_8888)
        Canvas(image).apply {
            drawColor(Color.WHITE)
            val pen = Paint().apply { color=Color.BLACK; style=Paint.Style.STROKE; strokeWidth=3f }
            for (i in 0..3) drawRect(130f+145*i,130f,240f+145*i,240f,pen)
            drawRect(475f,158f,505f,214f,pen)
        }
        val result = ScoringEngine.get(InstrumentationRegistry.getInstrumentation().targetContext).scoreWithQr(image,
            listOf(PointF(20f,20f),PointF(80f,20f),PointF(80f,80f),PointF(20f,80f)),4)
        assertEquals(4,result.questionCount)
        result.answerImage.recycle(); image.recycle()
    }
    @Test
    fun normalizationKeepsRedInkNearTheBottomFrame() {
        assertTrue(org.opencv.android.OpenCVLoader.initLocal())
        val frame = Bitmap.createBitmap(120,120,Bitmap.Config.ARGB_8888)
        Canvas(frame).apply {
            drawColor(Color.WHITE)
            drawRect(2f,2f,118f,118f,Paint().apply { color=Color.BLACK; style=Paint.Style.STROKE; strokeWidth=2f })
            drawLine(25f,80f,25f,115f,Paint().apply { color=Color.RED; strokeWidth=3f })
        }
        val normalized = FieldPreprocessor.normalize(frame)
        // Original pixel (25,114), minus 3px trim. Padding is applied to masks, not grayscale images.
        assertTrue("Do not crop away handwriting beside the frame", Color.red(normalized.getPixel(22,111)) < 180)
        normalized.recycle(); frame.recycle()
    }
    @Test
    fun emptyPrintedScoreBoxHasNoDigitGlyphs() {
        assertTrue(org.opencv.android.OpenCVLoader.initLocal())
        val frame = Bitmap.createBitmap(120, 120, Bitmap.Config.ARGB_8888)
        Canvas(frame).apply {
            drawColor(Color.WHITE)
            drawRect(3f,3f,116f,116f,Paint().apply { color = Color.BLACK; style = Paint.Style.STROKE; strokeWidth = 3f })
        }
        val normalized = FieldPreprocessor.normalize(frame)
        val reading = InkDigitRecognizer(InstrumentationRegistry.getInstrumentation().targetContext).read(normalized)
        assertEquals(0, reading.count)
        assertEquals(null, reading.value)
        frame.recycle(); normalized.recycle()
    }
    @Test
    fun liveCameraScreenStartsAndClosesWithoutCrashing() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, "android.permission.CAMERA")
        val activity = instrumentation.startActivitySync(
            Intent(context, LiveCameraActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        )
        instrumentation.waitForIdleSync()
        instrumentation.runOnMainSync {
            assertTrue(activity.findViewById<android.view.View>(R.id.closeButton).isEnabled)
            activity.findViewById<android.view.View>(R.id.closeButton).performClick()
        }
    }
    @Test
    fun darkShadowedAndPerspectivePhotosKeepEveryDigit() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val original = instrumentation.context.assets.open("sample-answer.jpg").use { BitmapFactory.decodeStream(it) }
        val w = original.width.toFloat(); val h = original.height.toFloat()
        val dark = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        Canvas(dark).drawBitmap(original, 0f, 0f, Paint().apply {
            colorFilter = ColorMatrixColorFilter(ColorMatrix(floatArrayOf(.5f,0f,0f,0f,0f, 0f,.5f,0f,0f,0f, 0f,0f,.5f,0f,0f, 0f,0f,0f,1f,0f)))
        })
        val shadow = original.copy(Bitmap.Config.ARGB_8888, true)
        Canvas(shadow).drawRect(0f, 0f, w, h, Paint().apply {
            shader = LinearGradient(0f, 0f, w, h, 0x99000000.toInt(), Color.TRANSPARENT, Shader.TileMode.CLAMP)
        })
        val perspective = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        val transform = Matrix().apply { setPolyToPoly(floatArrayOf(0f,0f,w,0f,w,h,0f,h),0,
            floatArrayOf(w*.08f,h*.02f,w*.96f,h*.08f,w*.93f,h*.94f,w*.02f,h*.96f),0,4) }
        Canvas(perspective).apply { drawColor(Color.WHITE); drawBitmap(original, transform, Paint(Paint.FILTER_BITMAP_FLAG)) }
        val engine = ScoringEngine.get(instrumentation.targetContext)
        for ((name, image) in listOf("dark" to dark, "shadow" to shadow, "perspective" to perspective)) {
            val result = engine.score(image, 3)
            assertEquals(name, listOf(12,95,84), result.scores)
            assertEquals(191, result.total)
            image.recycle(); result.answerImage.recycle()
        }
    }
    @Test
    fun unreadableFieldsCanBeConfirmedWithoutTreatingThemAsZero() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val activity = instrumentation.startActivitySync(
            Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as MainActivity
        instrumentation.runOnMainSync {
            activity.showScores(listOf(null, null, null, null), null)
            val total = activity.findViewById<TextView>(R.id.totalScore)
            assertEquals("要確認", total.text.toString())
            val rows = activity.findViewById<android.widget.LinearLayout>(R.id.scoreList)
            listOf(6, 13, 42, 10).forEachIndexed { index, score ->
                ((rows.getChildAt(index) as android.widget.LinearLayout).getChildAt(1) as android.widget.EditText).setText(score.toString())
            }
            assertEquals("71 点", total.text.toString())
            ((rows.getChildAt(0) as android.widget.LinearLayout).getChildAt(1) as android.widget.EditText).setText("")
            assertEquals("要確認", total.text.toString())
            activity.finish()
        }
    }

    @Test
    fun missingExpectedFramesMustNotProduceATotal() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val original = instrumentation.context.assets.open("sample-answer.jpg").use { BitmapFactory.decodeStream(it) }
        try {
            ScoringEngine(instrumentation.targetContext).score(original, 4)
            org.junit.Assert.fail("A three-frame sheet must not be confirmed as a four-frame sheet")
        } catch (error: Exception) {
            assertTrue(error.message.orEmpty().contains("指定の4個"))
        }
    }
    @Test
    fun appStartsWithImageAndCameraActionsAndNoNetworkPermission() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assertEquals(PackageManager.PERMISSION_DENIED,
            context.packageManager.checkPermission("android.permission.INTERNET", context.packageName))
        val activity = instrumentation.startActivitySync(
            Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        )
        instrumentation.runOnMainSync {
            assertEquals("-- 点", activity.findViewById<TextView>(R.id.totalScore).text.toString())
            assertTrue(activity.findViewById<android.view.View>(R.id.liveButton).isEnabled)
            assertTrue(activity.findViewById<android.view.View>(R.id.cameraButton).isEnabled)
            assertTrue(activity.findViewById<android.view.View>(R.id.galleryButton).isEnabled)
            activity.finish()
        }
    }

    @Test
    fun sampleAnswerIsScoredEntirelyOnDevice() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val image = instrumentation.context.assets.open("sample-answer.jpg").use {
            BitmapFactory.decodeStream(it)
        }
        val result = ScoringEngine(instrumentation.targetContext).score(image)
        assertEquals(listOf(12, 95, 84), result.scores)
        assertEquals(191, result.total)
        assertEquals(3, result.questionCount)
        assertTrue(result.digitCount >= 6)
    }

    @Test
    fun sampleScoresArePreservedInAllFourOrientations() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val original = instrumentation.context.assets.open("sample-answer.jpg").use { BitmapFactory.decodeStream(it) }
        val engine = ScoringEngine(instrumentation.targetContext)
        for (degrees in listOf(90, 180, 270)) {
            val rotated = Bitmap.createBitmap(original, 0, 0, original.width, original.height,
                Matrix().apply { postRotate(degrees.toFloat()) }, true)
            val result = engine.score(rotated)
            assertEquals("rotation=$degrees", listOf(12, 95, 84), result.scores)
            assertEquals(191, result.total)
            rotated.recycle()
        }
    }

    @Test
    fun slightlyTiltedSampleKeepsBothDigits() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val original = instrumentation.context.assets.open("sample-answer.jpg").use { BitmapFactory.decodeStream(it) }
        val rotated = Bitmap.createBitmap(original, 0, 0, original.width, original.height,
            Matrix().apply { postRotate(3f) }, true)
        val result = ScoringEngine(instrumentation.targetContext).score(rotated)
        assertEquals(listOf(12, 95, 84), result.scores)
        assertEquals(191, result.total)
    }

    @Test
    fun reportedScreenshotCropsFourScoreFramesInsteadOfQuestions() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val screenshot = instrumentation.context.assets.open("reported-live.png").use { BitmapFactory.decodeStream(it) }
        // The screenshot QR is blurred, so inject its visible location to test cropping independently.
        val result = ScoringEngine(instrumentation.targetContext).scoreWithQr(screenshot, listOf(
            PointF(17f, 1137f), PointF(77f, 1137f), PointF(77f, 1200f), PointF(17f, 1200f)
        ), trace = { android.util.Log.i("ScoreRegression", it) })
        assertEquals(4, result.questionCount)
        assertEquals(0, result.rotationDegrees)
        assertTrue("The preview must contain only the scoring strip", result.answerImage.height < 150)
        android.util.Log.i("ScoreRegression", "reported screenshot: ${result.scores}, ${result.diagnosticText}")
        assertEquals(listOf(6, 13, 42, 10), result.scores)
        assertEquals(71, result.total)
    }
}
