package com.example.routex

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.example.routex.databinding.ActivityMainBinding
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.Date
import kotlin.math.abs
import androidx.lifecycle.lifecycleScope
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var previewView: PreviewView

    // CameraX
    private var videoCapture: VideoCapture<Recorder>? = null
    private var imageCapture: ImageCapture? = null
    private var currentRecording: Recording? = null
    private var videoStartBootNs: Long = 0L

    // Pose service
    private var poseBuf: PoseBuffer? = null
    private var svcConn: ServiceConnection? = null

    // REC badge
    private lateinit var recBadge: android.widget.TextView
    private var isRecording = false

    // Uploader (used by SNAP path)
    private val uploader by lazy { FirebaseUploader() }

    // Sampling & matching
    private val sampleFpsExpr = "1"             // UDTIP video frame sampling (1 Hz default)
    private val maxPoseDeltaNs = 5_000_000_000L // ±5 s tolerance when pairing frame ↔ pose

    // Permissions
    private val requestPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { perms ->
        val granted = perms.values.all { it }
        if (!granted) {
            Toast.makeText(this, "Permissions denied", Toast.LENGTH_LONG).show()
            return@registerForActivityResult
        }
        startSensorService()
        startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCvManager.ensureInit() // safe no-op if OpenCV not bundled

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Preview
        previewView = PreviewView(this).apply {
            // Fill center to reduce visible bars in preview (frames match target aspect below)
            scaleType = PreviewView.ScaleType.FILL_CENTER
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        binding.cameraContainer.addView(previewView)

        lifecycleScope.launch {
            try {
                if (Firebase.auth.currentUser == null) {
                    Firebase.auth.signInAnonymously().await()
                }
                TDML.start(this@MainActivity, datasetId = "routex3-2025-busan")
                ML.start(this@MainActivity,   datasetId = "routex3-2025-busan")
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }



        // REC badge
        recBadge = android.widget.TextView(this).apply {
            text = "● REC"
            setPadding(24, 12, 24, 12)
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundColor(0x88CC0000.toInt())
            textSize = 16f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            visibility = View.GONE
        }
        (binding.cameraContainer as FrameLayout).addView(
            recBadge,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = android.view.Gravity.TOP or android.view.Gravity.END
                topMargin = 24
                rightMargin = 24
            }
        )

        // Ask permissions
        val needs = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            needs.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
        requestPermissions.launch(needs.toTypedArray())

        // Buttons
        binding.btnStart.setOnClickListener { startRecordingVideo() }
        binding.btnStop.setOnClickListener { stopRecordingVideo() }
        binding.btnSnap.setOnClickListener { snapStill() }
        findViewById<FloatingActionButton>(R.id.fabBack).setOnClickListener { finish() }
    }

    override fun onDestroy() {
        super.onDestroy()
        currentRecording?.stop()
        svcConn?.let { unbindService(it) }
        SensorCaptureService.stop(this)
    }

    // ---- Sensor service

    private fun startSensorService() {
        SensorCaptureService.start(this)
        svcConn = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                poseBuf = (binder as? SensorCaptureService.LocalBinder)?.getBuffer()
                Toast.makeText(this@MainActivity, "Pose service connected", Toast.LENGTH_SHORT).show()
            }
            override fun onServiceDisconnected(name: ComponentName?) { poseBuf = null }
        }
        bindService(Intent(this, SensorCaptureService::class.java), svcConn!!, BIND_AUTO_CREATE)
    }

    // ---- CameraX

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3) // avoid bars on emulator
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val recorder = Recorder.Builder()
                .setQualitySelector(
                    // Prefer 4:3 SD on emulator; on phones CameraX picks HD/FHD automatically
                    QualitySelector.fromOrderedList(listOf(Quality.SD, Quality.HD, Quality.FHD))
                )
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val selector = CameraSelector.DEFAULT_BACK_CAMERA
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, videoCapture, imageCapture)
        }, ContextCompat.getMainExecutor(this))
    }

    // ---- Recording

    private fun startRecordingVideo() {
        val vc = videoCapture ?: run {
            Toast.makeText(this, "Camera not ready", Toast.LENGTH_SHORT).show()
            return
        }
        if (currentRecording != null) {
            Toast.makeText(this, "Already recording", Toast.LENGTH_SHORT).show()
            return
        }

        val videoFile = outVideoFile()
        val fileOut = FileOutputOptions.Builder(videoFile).build()

        currentRecording = vc.output
            .prepareRecording(this, fileOut)
            // .withAudioEnabled() // keep disabled; not needed here
            .start(ContextCompat.getMainExecutor(this)) { ev ->
                when (ev) {
                    is VideoRecordEvent.Start -> {
                        videoStartBootNs = TimeUtils.bootNs()
                        onRecordingStarted()
                    }
                    is VideoRecordEvent.Finalize -> {
                        currentRecording = null
                        onRecordingStopped()
                        if (ev.error == VideoRecordEvent.Finalize.ERROR_NONE) {
                            val path = ev.outputResults.outputUri.path ?: videoFile.absolutePath
                            // Queue background work so it continues if Activity closes
                            enqueueVideoPostProcess(path, videoStartBootNs, sampleFpsExpr)
                        } else {
                            Toast.makeText(this, "Record error: ${ev.error}", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            }
    }

    private fun stopRecordingVideo() {
        currentRecording?.stop()
    }

    private fun onRecordingStarted() {
        isRecording = true
        recBadge.visibility = View.VISIBLE
        Toast.makeText(this, "Recording started", Toast.LENGTH_SHORT).show()
    }

    private fun onRecordingStopped() {
        isRecording = false
        recBadge.visibility = View.GONE
        Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
    }

    // ---- SNAP (single still => GeoPose + upload)

    private fun snapStill() {
        val ic = imageCapture ?: run {
            Toast.makeText(this, "ImageCapture not ready", Toast.LENGTH_SHORT).show()
            return
        }

        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US)
        val base = "RC_${sdf.format(Date())}"
        val dir = File(getExternalFilesDir(null), "frames/$base").apply { mkdirs() }
        val photoFile = File(dir, "${base}_000001.jpg")
        val output = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        val tBootNs = TimeUtils.bootNs()

        ic.takePicture(
            output,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Toast.makeText(this@MainActivity, "Snap failed: ${exc.message}", Toast.LENGTH_LONG).show()
                }
                override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                    lifecycleScope.launch(Dispatchers.IO) {
                        val pb = poseBuf
                        val gnss = pb?.nearestGnss(tBootNs)
                        val quat = pb?.interpQuat(tBootNs) ?: pb?.nearestQuat(tBootNs)

                        if (gnss == null || quat == null ||
                            abs(gnss.tBootNs - tBootNs) > maxPoseDeltaNs) {
                            withContext(Dispatchers.Main) {
                                Toast.makeText(
                                    this@MainActivity,
                                    "No pose for snap (timestamp too far from sensors)",
                                    Toast.LENGTH_LONG
                                ).show()
                            }
                            return@launch
                        }

                        val geoFile = File(photoFile.parentFile, "${photoFile.nameWithoutExtension}.geopose.json")
                        GeoPoseWriter.writeQuaternion(
                            outFile = geoFile,
                            id = photoFile.nameWithoutExtension,
                            timestampISO = TimeUtils.iso8601UTC(gnss.tUtcMs),
                            lat = gnss.lat, lon = gnss.lon, h = gnss.h ?: 0.0,
                            quatX = quat.x, quatY = quat.y, quatZ = quat.z, quatW = quat.w,
                            posStd = gnss.accM?.toDouble(), oriStdDeg = 2.0
                        )

                        try {
                            uploader.uploadImageWithGeoPose(
                                imageFile = photoFile,
                                geoposeFile = geoFile,
                                meta = mapOf("ts" to gnss.tUtcMs, "lat" to gnss.lat, "lon" to gnss.lon)
                            )
                            withContext(Dispatchers.Main) {
                                Toast.makeText(this@MainActivity, "Snap uploaded", Toast.LENGTH_SHORT).show()
                            }
                        } catch (e: Exception) {
                            withContext(Dispatchers.Main) {
                                Toast.makeText(this@MainActivity, "Upload failed: ${e.message}", Toast.LENGTH_LONG).show()
                            }
                        }
                    }
                }
            }
        )
    }

    // ---- WorkManager: queue video → frames → GeoPose → upload

    private fun enqueueVideoPostProcess(videoPath: String, startBootNs: Long, fpsExpr: String) {
        val base = File(videoPath).nameWithoutExtension
        val data = workDataOf(
            VideoPostProcessWorker.KEY_VIDEO_PATH to videoPath,
            VideoPostProcessWorker.KEY_START_BOOT_NS to startBootNs,
            VideoPostProcessWorker.KEY_FPS_EXPR to fpsExpr
        )
        val req = OneTimeWorkRequestBuilder<VideoPostProcessWorker>()
            .setInputData(data)
            .build()

        WorkManager.getInstance(applicationContext)
            .enqueueUniqueWork("postproc_$base", ExistingWorkPolicy.REPLACE, req)

        Toast.makeText(this, "Post-processing queued: $base", Toast.LENGTH_SHORT).show()
    }

    // ---- utils

    private fun outVideoFile(): File {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US)
        val name = "RC_${sdf.format(Date())}.mp4"
        val dir = File(getExternalFilesDir(null), "videos").apply { mkdirs() }
        return File(dir, name)
    }
}
