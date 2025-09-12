package com.example.routex  // ← change to your namespace if different

import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext


import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.location.Location
import android.os.Build
import android.os.Bundle
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.routex.databinding.ActivityMainBinding
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.CancellationTokenSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {
    // top of class
    private lateinit var uploader: FirebaseUploader
    private lateinit var infer: ResnetInference
    private lateinit var overlayText: android.widget.TextView


    private lateinit var binding: ActivityMainBinding

    private lateinit var previewView: PreviewView
    private var imageCapture: ImageCapture? = null

    private var snapJob: Job? = null
    private val snapIntervalMs = 2000L

    private val requestPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { perms ->
        val granted = perms.values.all { it }
        if (granted) startCamera() else {
            Toast.makeText(this, "Permissions denied", Toast.LENGTH_LONG).show()
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        deleteCachedAsset(this, "resnet_scripted.pt")
        // Delete any cached model from filesDir (one-time fix)
        listOf("resnet_scripted.pt", "resnet_scripted.ptl").forEach { name ->
            try { java.io.File(filesDir, name).delete() } catch (_: Exception) {}
        }

        setContentView(binding.root)
        uploader = FirebaseUploader(this)
        infer = ResnetInference(this)



        // Add PreviewView programmatically (safer for Layout Preview)
        previewView = PreviewView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        binding.cameraContainer.addView(previewView)

        overlayText = android.widget.TextView(this).apply {
            text = "—"
            setPadding(24, 16, 24, 16)
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundColor(0x88000000.toInt()) // semi-transparent black
            textSize = 20f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
        (binding.cameraContainer as android.widget.FrameLayout).addView(
            overlayText,
            android.widget.FrameLayout.LayoutParams(
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = android.view.Gravity.TOP or android.view.Gravity.CENTER_HORIZONTAL
                topMargin = 24
            }
        )


        // Request runtime permissions (Camera + Fine Location; Write ext storage on P and below)
        val needs = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION
        )
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            needs.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
        requestPermissions.launch(needs.toTypedArray())

        // Buttons
        binding.btnStart.setOnClickListener { startSnappingLoop() }
        binding.btnStop.setOnClickListener { stopSnappingLoop() }
        binding.btnSnap.setOnClickListener { snapOnce() }
    }

    private fun testFirebaseOnce() {
        lifecycleScope.launch(Dispatchers.IO) {
            val db = FirebaseFirestore.getInstance()
            val storage = FirebaseStorage.getInstance().reference

            try {
                val now = System.currentTimeMillis()

                // 1) Firestore test write
                db.collection("captures")
                    .document("_healthcheck_$now")
                    .set(mapOf("ok" to true, "ts" to now, "note" to "hello firestore"))
                    .await()

                // 2) Storage test upload
                val bytes = "hello storage @ $now".toByteArray()
                storage.child("healthchecks/hello_$now.txt").putBytes(bytes).await()

                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Firebase OK ✅", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Firebase FAIL: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val selector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, selector, preview, imageCapture)
            } catch (e: Exception) {
                e.printStackTrace()
                Toast.makeText(this, "Failed to bind camera: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startSnappingLoop() {
        if (imageCapture == null) {
            Toast.makeText(this, "Camera not ready yet", Toast.LENGTH_SHORT).show()
            return
        }
        snapJob?.cancel()
        snapJob = lifecycleScope.launch(Dispatchers.Main) {
            Toast.makeText(this@MainActivity, "Auto-snap started", Toast.LENGTH_SHORT).show()
            while (isActive) {
                snapOnce()
                delay(snapIntervalMs)
            }
        }
    }

    private fun stopSnappingLoop() {
        snapJob?.cancel()
        snapJob = null
        Toast.makeText(this, "Stopped", Toast.LENGTH_SHORT).show()
    }

    private fun snapOnce() {
        val capture = imageCapture ?: run {
            Toast.makeText(this, "ImageCapture not ready", Toast.LENGTH_SHORT).show()
            return
        }

        val photoFile = outFile("jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        capture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Toast.makeText(this@MainActivity, "Capture failed: ${exc.message}", Toast.LENGTH_SHORT).show()
                }

                override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                    lifecycleScope.launch(Dispatchers.IO) {
                        val loc = getCurrentLocationOrNull()
                        val lat = loc?.latitude
                        val lon = loc?.longitude

                        // Decode the saved JPG into a moderate bitmap (prevents OOM)
                        val bmp = ImageUtils.decodeDownsampled(photoFile, 640, 640)

// Run on-device inference (ResNet)
                        // Top-K predictions
                        val preds = infer.classifyWithProbs(bmp, topK = 3)
                        val top1 = preds.first()
                        val label = top1.label
                        val conf = top1.prob

// Log full dist to Logcat
                        android.util.Log.d("ML", "preds = " + preds.joinToString { "${it.label}=${"%.2f".format(it.prob)}" })

// Update overlay + Toast on UI
                        launch(Dispatchers.Main) {
                            overlayText.text = "${label}  (${(conf*100).toInt()}%)"
                            Toast.makeText(this@MainActivity, "Label: $label (${(conf*100).toInt()}%)", Toast.LENGTH_SHORT).show()
                        }

                        val base = photoFile.nameWithoutExtension
                        val metaTxt = "lat=${lat ?: ""}, lon=${lon ?: ""}, label=$label, conf=${"%.4f".format(conf)}"

                        // write sidecar *.txt locally
                        File(photoFile.parentFile, "$base.txt").writeText(metaTxt)


                        // 🔼 upload JPG + TXT + metadata doc
                        try {
                            uploader.uploadImageWithTxt(
                                imageFile = photoFile,
                                txtContent = metaTxt,
                                meta = mapOf(
                                    "lat" to (lat ?: Double.NaN),
                                    "lon" to (lon ?: Double.NaN),
                                    "label" to label,
                                    "conf" to conf,
                                    "ts" to System.currentTimeMillis()
                                )
                            )
                            launch(Dispatchers.Main) {
                                Toast.makeText(this@MainActivity, "Uploaded ${photoFile.name}", Toast.LENGTH_SHORT).show()
                            }
                        } catch (e: Exception) {
                            launch(Dispatchers.Main) {
                                Toast.makeText(this@MainActivity, "Upload failed: ${e.message}", Toast.LENGTH_LONG).show()
                            }
                        }
                    }
                }
            }
        )
    }

    private fun outFile(ext: String): File {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US)
        val name = "RC_${sdf.format(Date())}.$ext"
        val dir = File(getExternalFilesDir(null), "captures").apply { mkdirs() }
        return File(dir, name)
    }

    @SuppressLint("MissingPermission")
    private suspend fun getCurrentLocationOrNull(): Location? {
        val client = LocationServices.getFusedLocationProviderClient(this)

        // Try fast: last known
        val last = try {
            client.lastLocation.addOnFailureListener { }.awaitOrNull()
        } catch (_: Exception) { null }
        if (last != null) return last

        // Fallback: request a fresh current location
        return try {
            val cts = CancellationTokenSource()
            client.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, cts.token)
                .addOnFailureListener { }
                .awaitOrNull()
        } catch (_: Exception) { null }
    }
}

/* ---------- Simple Task<T>.awaitOrNull() helper without extra deps ---------- */
private fun <T> com.google.android.gms.tasks.Task<T>.awaitOrNull(): T? {
    // NOTE: This blocks the calling thread until completion. We call it from Dispatchers.IO.
    return try {
        com.google.android.gms.tasks.Tasks.await(this)
    } catch (_: Exception) {
        null
    }
}
