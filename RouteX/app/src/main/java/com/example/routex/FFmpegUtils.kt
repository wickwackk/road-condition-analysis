package com.example.routex

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import java.io.File
import java.io.FileOutputStream
import kotlin.math.roundToLong

object FFmpegUtils {

    /** Public so MainActivity can compute frame time spacing. */
    fun fpsFromExpr(expr: String): Double {
        val s = expr.trim()
        return if ("/" in s) {
            val (n, d) = s.split("/", limit = 2)
            (n.toDoubleOrNull() ?: 1.0) / (d.toDoubleOrNull() ?: 1.0)
        } else s.toDoubleOrNull() ?: 1.0
    }

    /**
     * Extract frames at a fixed rate.
     * Tries FFmpegKit via reflection; if not present, falls back to MediaMetadataRetriever.
     * @param fpsExpr "1" (1 Hz), "2" (2 Hz), "1/3" (one frame every 3 s), etc.
     */
    fun extractFrames(video: File, outDir: File, fpsExpr: String = "1"): List<File> {
        return try {
            extractWithFFmpegKit(video, outDir, fpsExpr)
        } catch (_: Throwable) {
            val fps = fpsFromExpr(fpsExpr)
            extractWithRetriever(video, outDir, fps)
        }
    }

    // ---------- FFmpegKit (reflection; optional) ----------
    private fun extractWithFFmpegKit(video: File, outDir: File, fpsExpr: String): List<File> {
        outDir.mkdirs()
        val pattern = File(outDir, "frame_%06d.jpg").absolutePath
        val ffmpegKit = Class.forName("com.arthenica.ffmpegkit.FFmpegKit")
        val execute = ffmpegKit.getMethod("execute", String::class.java)
        val session = execute.invoke(null, "-i ${video.absolutePath} -vf fps=$fpsExpr -q:v 2 $pattern")
        val sessionClass = Class.forName("com.arthenica.ffmpegkit.Session")
        val rc = sessionClass.getMethod("getReturnCode").invoke(session)
        val rcClass = Class.forName("com.arthenica.ffmpegkit.ReturnCode")
        val ok = rcClass.getMethod("isValueSuccess", rcClass).invoke(null, rc) as Boolean
        if (!ok) throw RuntimeException("FFmpegKit failed")
        return outDir.listFiles { f -> f.extension.equals("jpg", true) }?.sortedBy { it.name } ?: emptyList()
    }

    // ---------- Fallback: Android retriever ----------
    private fun extractWithRetriever(video: File, outDir: File, fps: Double): List<File> {
        outDir.mkdirs()
        val r = MediaMetadataRetriever()
        r.setDataSource(video.absolutePath)

        val durationMs = (r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L)
        if (durationMs <= 0 || fps <= 0.0) return emptyList()

        val intervalMs = (1000.0 / fps).roundToLong()
        val out = mutableListOf<File>()
        var t = 0L
        var idx = 1
        while (t <= durationMs) {
            val bmp: Bitmap? = r.getFrameAtTime(t * 1000, MediaMetadataRetriever.OPTION_CLOSEST)
            if (bmp != null) {
                val f = File(outDir, "frame_${String.format("%06d", idx)}.jpg")
                FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                bmp.recycle()
                out.add(f)
            }
            idx++
            t += intervalMs
        }
        r.release()
        return out
    }
}
