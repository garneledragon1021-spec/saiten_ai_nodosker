package jp.ebina.scorecamera

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.view.MotionEvent
import androidx.camera.core.Camera
import androidx.camera.core.FocusMeteringAction
import java.util.concurrent.TimeUnit
import android.util.Size
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.OnBackPressedCallback
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import jp.ebina.scorecamera.databinding.ActivityLiveCameraBinding
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class LiveCameraActivity : AppCompatActivity() {
    private lateinit var binding: ActivityLiveCameraBinding
    private lateinit var scoringEngine: ScoringEngine
    private val executor = Executors.newSingleThreadExecutor()
    private val scoring = AtomicBoolean(false)
    private val session = LiveScoreSession()
    private var camera: Camera? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var torchEnabled = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        binding = ActivityLiveCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = finishWithResult()
        })
        scoringEngine = ScoringEngine.get(applicationContext)
        binding.closeButton.setOnClickListener { finishWithResult() }
        binding.nextAnswerButton.setOnClickListener {
            resetReading("次の答案のQRと得点欄を写してください。前の結果はクリアしました。")
            // Drop buffered frames; keep the camera open instead of rebinding it.
            imageAnalysis?.let { it.clearAnalyzer(); it.setAnalyzer(executor, ::analyze) }
        }
        binding.flashButton.setOnClickListener {
            camera?.let { bound ->
                torchEnabled = !torchEnabled
                bound.cameraControl.enableTorch(torchEnabled)
                binding.flashButton.text = if (torchEnabled) "ライト切" else "ライト"
                resetReading("ライトを変更しました。QRと得点欄を写してください。")
            }
        }
        binding.previewView.setOnTouchListener { view, event ->
            if (event.action == MotionEvent.ACTION_UP) {
                val point = binding.previewView.meteringPointFactory.createPoint(event.x, event.y)
                camera?.cameraControl?.startFocusAndMetering(FocusMeteringAction.Builder(point,
                    FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE).setAutoCancelDuration(6, TimeUnit.SECONDS).build())
                resetReading("ピントを調整しています。QRと得点欄を写してください。")
                view.performClick()
            }
            true
        }
        startCamera()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (isFinishing || isDestroyed) return@addListener
            try {
                val provider = future.get()
                val preview = Preview.Builder().build().also { it.surfaceProvider = binding.previewView.surfaceProvider }
                var lastFailure: Exception? = null
                for (size in listOf(Size(1920, 1080), Size(1280, 720))) {
                    val analysis = ImageAnalysis.Builder()
                        .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                            ResolutionStrategy(size, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                        ).build())
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                    try {
                        provider.unbindAll()
                        camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                        imageAnalysis = analysis
                        analysis.setAnalyzer(executor, ::analyze)
                        lastFailure = null
                        break
                    } catch (error: IllegalArgumentException) { analysis.clearAnalyzer(); lastFailure = error }
                }
                lastFailure?.let { throw it }
                binding.flashButton.isEnabled = camera?.cameraInfo?.hasFlashUnit() == true
            } catch (_: Exception) {
                binding.liveScores.text = "カメラを開始できません。閉じて写真を撮るか、写真から読み込んでください。"
                binding.flashButton.isEnabled = false
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        val now = android.os.SystemClock.elapsedRealtime()
        if (session.isHeld || session.coolingDown(now) || !scoring.compareAndSet(false, true)) {
            image.close()
            return
        }
        val generation = session.generation
        var bitmap: Bitmap? = null
        runOnUiThread {
            if (!isFinishing && !isDestroyed && generation == session.generation) binding.liveProgress.visibility = View.VISIBLE
        }
        try {
            val rotation = image.imageInfo.rotationDegrees
            val rawBitmap = image.toBitmap()
            image.close()
            val upright = if (rotation == 0) rawBitmap else Bitmap.createBitmap(
                rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height,
                Matrix().apply { postRotate(rotation.toFloat()) }, true,
            ).also { if (it !== rawBitmap) rawBitmap.recycle() }
            bitmap = upright
            val result = scoringEngine.score(upright, intent.getIntExtra("expectedQuestionCount", 4)) {
                generation != session.generation
            }
            runOnUiThread {
                if (!isFinishing && !isDestroyed) {
                    showReading(generation, result.scores)
                }
            }
            result.answerImage.recycle()
        } catch (error: Exception) {
            runOnUiThread {
                if (!isFinishing && !isDestroyed && session.failed(generation)) {
                    binding.liveTotalLabel.text = "合計点（読取待ち）"
                    binding.liveTotal.text = "-- 点"
                    binding.liveScores.text = error.message ?: "読取に失敗しました。"
                }
            }
        } finally {
            image.close()
            bitmap?.recycle()
            // Leave a cooling/interaction interval after inference on the g06 instead of continuously saturating its CPU.
            // A new paper must not inherit the previous paper's cooling interval.
            session.completed(generation, android.os.SystemClock.elapsedRealtime())
            scoring.set(false)
            runOnUiThread {
                if (!isFinishing && !isDestroyed && generation == session.generation) binding.liveProgress.visibility = View.GONE
            }
        }
    }

    private fun resetReading(message: String) {
        session.reset()
        binding.liveTotalLabel.text = "合計点（読取待ち）"
        binding.liveTotal.text = "-- 点"
        binding.liveScores.text = message
        binding.liveProgress.visibility = View.GONE
    }

    // Kept separate from camera delivery so provisional/confirmed UI can be regression tested.
    internal fun showReading(generation: Int, scores: List<Int?>) {
        val reading = session.accept(generation, scores, android.os.SystemClock.elapsedRealtime()) ?: return
        binding.liveTotalLabel.text = when {
            reading.confirmed -> "合計点（確認済み）"
            reading.total != null -> "暫定合計（確認中）"
            else -> "合計点（未読欄あり）"
        }
        binding.liveTotal.text = reading.total?.let { "$it 点" } ?: "要確認"
        binding.liveScores.text = (when {
            reading.confirmed -> "結果を保持中。「次の答案」で再開します。\n"
            reading.total != null -> "同じ結果をもう一度確認しています。\n"
            else -> ""
        }) +
            reading.scores.mapIndexed { index, score -> "大問${index + 1}: ${score?.let { "${it}点" } ?: "読取不能"}" }.joinToString("　")
    }

    private fun finishWithResult() {
        setResult(RESULT_OK, Intent().apply { session.resultScores()?.let { putExtra(EXTRA_SCORES, it) } })
        finish()
    }

    override fun onDestroy() {
        session.reset()
        imageAnalysis?.clearAnalyzer()
        super.onDestroy()
        executor.shutdown()
    }

    companion object {
        const val EXTRA_SCORES = "scores"
    }
}
