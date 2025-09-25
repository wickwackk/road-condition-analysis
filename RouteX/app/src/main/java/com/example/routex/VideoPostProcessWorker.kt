package com.example.routex

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max

/**
 * Offline post-process:
 *  1) Extract frames from the recorded video at a requested sampling rate (fpsExpr).
 *  2) For each frame, find GNSS/orientation from PoseBuffer by boot-time (ns).
 *  3) Write OGC GeoPose (Basic / quaternion) JSON.
 *  4) Upload (image + geopose.json) to Firebase Storage and write minimal meta.
 *
 * Inputs (WorkManager):
 *  - KEY_VIDEO_PATH: String (absolute path to the recorded .mp4)
 *  - KEY_START_BOOT_NS: Long (elapsedRealtimeNanos at Recorder Start event)
 *  - KEY_FPS_EXPR: String (e.g., "1", "2", "1/3" …)
 */
class VideoPostProcessWorker(
    appContext: Context,
    params: WorkerParameters
) : CoroutineWorker(appContext, params) {

    companion object {
        const val KEY_VIDEO_PATH = "videoPath"
        const val KEY_START_BOOT_NS = "startBootNs"
        const val KEY_FPS_EXPR = "fpsExpr"

        // ± tolerance when matching poses to frame timestamps
        private const val DEFAULT_TOL_NS = 5_000_000_000L // 5 seconds
    }

    override suspend fun doWork(): Result {
        val videoPath = inputData.getString(KEY_VIDEO_PATH) ?: return Result.failure()
        val startBootNs = inputData.getLong(KEY_START_BOOT_NS, 0L)
        if (startBootNs <= 0L) return Result.failure()
        val fpsExpr = inputData.getString(KEY_FPS_EXPR) ?: "1"

        val video = File(videoPath)
        if (!video.exists()) return Result.failure()

        // 1) Extract frames using FFmpeg (to per-video subfolder)
        val framesDir = File(applicationContext.getExternalFilesDir(null), "frames/${video.nameWithoutExtension}")
        val frames: List<File> = try {
            FFmpegUtils.extractFrames(video, framesDir, fpsExpr = fpsExpr)
        } catch (e: Exception) {
            e.printStackTrace()
            return Result.failure()
        }

        // 2) Timestamp model for frames (evenly sampled from start)
        val fps = FFmpegUtils.fpsFromExpr(fpsExpr)
        val intervalNs = max(1L, (1e9 / fps).toLong())

        val poseBuf = SensorCaptureService.peekPoseBuffer() // same-process snapshot
        val uploader = FirebaseUploader()

        var idx = 0
        var written = 0

        // Use a classic for-loop (no lambda breaks/continues → compatible with Kotlin 1.9)
        val sortedFrames = frames.sortedBy { it.name }
        for (frame in sortedFrames) {
            val tFrameBootNs = startBootNs + idx * intervalNs

            // 3) Match nearest GNSS within tolerance
            val gnss = poseBuf?.nearestGnss(tFrameBootNs)
            if (gnss == null || abs(gnss.tBootNs - tFrameBootNs) > DEFAULT_TOL_NS) {
                idx++
                continue
            }

            // Orientation: prefer interp; else nearest, also within tolerance
            val qInterp = poseBuf.interpQuat(tFrameBootNs)
            val quat = qInterp ?: poseBuf.nearestQuat(tFrameBootNs)
            if (quat == null || (qInterp == null && abs(quat.tBootNs - tFrameBootNs) > DEFAULT_TOL_NS)) {
                idx++
                continue
            }

            // Stable ID & filename: <videoBase>_<000001>.jpg
            val id = "${video.nameWithoutExtension}_${String.format(Locale.US, "%06d", idx + 1)}"
            val renamed = File(frame.parentFile, "$id.jpg")
            if (frame.nameWithoutExtension != id) {
                try {
                    // copyTo keeps the original (safe); move if you prefer: frame.renameTo(renamed)
                    frame.copyTo(renamed, overwrite = true)
                } catch (e: Exception) {
                    e.printStackTrace()
                    idx++
                    continue
                }
            }

            // 4) Write GeoPose (Basic / quaternion)
            val geoFile = File(renamed.parentFile, "$id.geopose.json")
            try {
                GeoPoseWriter.writeQuaternion(
                    outFile = geoFile,
                    id = id,
                    timestampISO = TimeUtils.iso8601UTC(gnss.tUtcMs),
                    lat = gnss.lat, lon = gnss.lon, h = gnss.h ?: 0.0,
                    quatX = quat.x, quatY = quat.y, quatZ = quat.z, quatW = quat.w,
                    posStd = gnss.accM?.toDouble(), oriStdDeg = 2.0
                )
            } catch (e: Exception) {
                e.printStackTrace()
                idx++
                continue
            }

            // 5) Upload pair (best-effort)
            try {
                uploader.uploadImageWithGeoPose(
                    imageFile = renamed,
                    geoposeFile = geoFile,
                    meta = mapOf("ts" to gnss.tUtcMs, "lat" to gnss.lat, "lon" to gnss.lon)
                )
                written++
            } catch (e: Exception) {
                // Keep going; you can add a retry queue if needed
                e.printStackTrace()
            }

            idx++
        }

        return Result.success(
            workDataOf(
                "frames_total" to sortedFrames.size,
                "geopose_written" to written
            )
        )
    }
}
