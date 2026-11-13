package jp.ebina.scorecamera

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.LinearLayout
import android.text.InputType
import android.text.InputFilter
import androidx.core.widget.doAfterTextChanged
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import jp.ebina.scorecamera.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var scoringEngine: ScoringEngine
    private var pendingCaptureUri: Uri? = null
    private var pendingCameraAction: (() -> Unit)? = null

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) pendingCameraAction?.invoke() else showError("カメラの使用が許可されていません。")
        pendingCameraAction = null
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { saved ->
        if (saved) pendingCaptureUri?.let(::loadAndScore)
    }

    private val pickPhoto = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let(::loadAndScore)
    }

    private val liveCamera = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val scores = result.data?.getIntArrayExtra(LiveCameraActivity.EXTRA_SCORES) ?: return@registerForActivityResult
        showScores(scores.map { if (it < 0) null else it }, null)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Window-scoped: prevents idle lock only while this screen is visible; no wake lock permission.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        scoringEngine = ScoringEngine.get(applicationContext)
        binding.fieldCount.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, (2..20).map { it.toString() })
        binding.fieldCount.setSelection(2)

        binding.liveButton.setOnClickListener { requestCameraThen {
            liveCamera.launch(Intent(this, LiveCameraActivity::class.java).putExtra("expectedQuestionCount", expectedQuestionCount()))
        } }
        binding.cameraButton.setOnClickListener { requestCameraThen(::launchCamera) }
        binding.galleryButton.setOnClickListener {
            pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
    }

    private fun requestCameraThen(action: () -> Unit) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            action()
        } else {
            pendingCameraAction = action
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun launchCamera() {
        val directory = File(cacheDir, "captures").apply { mkdirs() }
        val file = File(directory, "answer-${System.currentTimeMillis()}.jpg")
        pendingCaptureUri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        pendingCaptureUri?.let { takePicture.launch(it) }
    }

    private fun loadAndScore(uri: Uri) {
        val expectedCount = expectedQuestionCount()
        lifecycleScope.launch {
            setBusy(true)
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, info, _ ->
                        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                        val longestSide = maxOf(info.size.width, info.size.height)
                        if (longestSide > 2400) {
                            val scale = 2400f / longestSide
                            decoder.setTargetSize((info.size.width * scale).toInt(), (info.size.height * scale).toInt())
                        }
                    }
                }
                binding.sourcePreview.setImageBitmap(bitmap)
                binding.sourcePreview.visibility = View.VISIBLE
                val result = withContext(Dispatchers.Default) { scoringEngine.score(bitmap, expectedCount) }
                showScores(result.scores, result)
            } catch (error: Exception) {
                showError(error.message ?: "採点中にエラーが発生しました。")
            } finally {
                setBusy(false)
            }
        }
    }

    private fun expectedQuestionCount() = binding.fieldCount.selectedItem.toString().toInt()

    internal fun showScores(scores: List<Int?>, result: ScoringResult?) {
        val confirmedScores = scores.toMutableList()
        binding.totalScore.text = if (scores.all { it != null }) "${scores.filterNotNull().sum()} 点" else "要確認"
        binding.scoreList.removeAllViews()
        scores.forEachIndexed { index, score ->
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(12, 8, 12, 8) }
            row.addView(TextView(this).apply {
                text = "大問${index + 1}"
                textSize = 16f
                setTextColor(ContextCompat.getColor(context, R.color.text_primary))
            }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            row.addView(EditText(this).apply {
                inputType = InputType.TYPE_CLASS_NUMBER
                filters = arrayOf(InputFilter.LengthFilter(3))
                hint = "読取不能"
                contentDescription = "大問${index + 1}の得点"
                textSize = 16f
                setText(score?.toString() ?: "")
                doAfterTextChanged { value ->
                    confirmedScores[index] = value?.toString()?.toIntOrNull()
                    binding.totalScore.text = if (confirmedScores.all { it != null }) "${confirmedScores.filterNotNull().sum()} 点" else "要確認"
                }
            }, LinearLayout.LayoutParams((120 * resources.displayMetrics.density).toInt(), LinearLayout.LayoutParams.WRAP_CONTENT))
            binding.scoreList.addView(row)
        }
        binding.scoreTitle.visibility = View.VISIBLE
        if (result != null) {
            binding.cropPreview.setImageBitmap(result.answerImage)
            binding.cropPreview.visibility = View.VISIBLE
            binding.cropTitle.visibility = View.VISIBLE
            binding.statusMessage.text = result.diagnosticText
            binding.statusMessage.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
            binding.statusMessage.visibility = View.VISIBLE
        } else {
            binding.sourcePreview.visibility = View.GONE
            binding.cropPreview.visibility = View.GONE
            binding.cropTitle.visibility = View.GONE
            binding.statusMessage.text = "得点欄をタップすると修正できます。"
            binding.statusMessage.visibility = View.VISIBLE
        }
    }

    private fun showError(message: String) {
        binding.statusMessage.text = message
        binding.statusMessage.setTextColor(0xFFB3261E.toInt())
        binding.statusMessage.visibility = View.VISIBLE
    }

    private fun setBusy(busy: Boolean) {
        binding.progress.visibility = if (busy) View.VISIBLE else View.GONE
        binding.liveButton.isEnabled = !busy
        binding.cameraButton.isEnabled = !busy
        binding.galleryButton.isEnabled = !busy
        binding.fieldCount.isEnabled = !busy
        if (busy) {
            binding.totalScore.text = "-- 点"
            binding.scoreList.removeAllViews()
            binding.scoreTitle.visibility = View.GONE
            binding.cropTitle.visibility = View.GONE
            binding.cropPreview.visibility = View.GONE
            binding.statusMessage.text = "端末内AIで採点しています…"
            binding.statusMessage.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
            binding.statusMessage.visibility = View.VISIBLE
        }
    }
}
