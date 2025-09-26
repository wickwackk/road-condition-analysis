package com.example.routex

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import com.google.firebase.storage.ktx.storage
import kotlinx.coroutines.*
import kotlinx.coroutines.tasks.await
import java.time.Instant
import java.util.concurrent.atomic.AtomicBoolean

import org.json.JSONObject
import kotlin.math.*

private fun yprDegToQuat(yawDeg: Double, pitchDeg: Double, rollDeg: Double): DoubleArray {
    // Convert ZYX (yaw, pitch, roll in degrees) to quaternion (x, y, z, w)
    val cy = cos(Math.toRadians(yawDeg)   * 0.5)
    val sy = sin(Math.toRadians(yawDeg)   * 0.5)
    val cp = cos(Math.toRadians(pitchDeg) * 0.5)
    val sp = sin(Math.toRadians(pitchDeg) * 0.5)
    val cr = cos(Math.toRadians(rollDeg)  * 0.5)
    val sr = sin(Math.toRadians(rollDeg)  * 0.5)

    val w = cr*cp*cy + sr*sp*sy
    val x = sr*cp*cy - cr*sp*sy
    val y = cr*sp*cy + sr*cp*sy
    val z = cr*cp*sy - sr*sp*cy
    return doubleArrayOf(x, y, z, w)
}

private fun parseGeoPose(bytes: ByteArray): ParsedPose {
    val s = bytes.toString(Charsets.UTF_8)
    val j = JSONObject(s)

    // Basic structures you generate:
    // - BasicQuaternion: { position:{lat,lon,h}, quaternion:{x,y,z,w}, ... }
    // - BasicYpr:        { position:{lat,lon,h}, yprAngles:{yaw,pitch,roll}, ... }
    val pos = j.optJSONObject("position") ?: JSONObject()
    val lat = pos.optDouble("lat", Double.NaN)
    val lon = pos.optDouble("lon", Double.NaN)
    val h   = pos.optDouble("h",   0.0)

    val qObj = j.optJSONObject("quaternion")
    val (qx, qy, qz, qw) =
        if (qObj != null) {
            listOf(
                qObj.optDouble("x", 0.0),
                qObj.optDouble("y", 0.0),
                qObj.optDouble("z", 0.0),
                qObj.optDouble("w", 1.0)
            )
        } else {
            val ypr = j.optJSONObject("yprAngles")
            if (ypr != null) {
                val yaw = ypr.optDouble("yaw", 0.0)
                val pit = ypr.optDouble("pitch", 0.0)
                val rol = ypr.optDouble("roll", 0.0)
                val q = yprDegToQuat(yaw, pit, rol)
                listOf(q[0], q[1], q[2], q[3])
            } else listOf(0.0, 0.0, 0.0, 1.0)
        }

    return ParsedPose(lat, lon, h, qx, qy, qz, qw)
}

private data class ParsedPose(
    val lat: Double, val lon: Double, val h: Double,
    val qx: Double, val qy: Double, val qz: Double, val qw: Double
)



/**
 * ML auto-labeler (TDML-compliant):
 * Call ML.start(context, DATASET_ID) once (e.g., in Application.onCreate()).
 * Every ~1.5s it:
 *   1) lists recent TDML samples
 *   2) skips ones that already have a label doc
 *   3) downloads the image
 *   4) runs your ResnetInference
 *   5) writes a TDML label doc at tdmlDatasets/{DATASET_ID}/labels/{sampleId}
 *
 * Label doc = annotations only (strict TDML):
 * {
 *   "sampleId": "id123",
 *   "annotations": [
 *     {"type":"classification","target":"road_surface","class":"<pred>","score": <0..1>}
 *   ],
 *   "createdAt": "2025-09-25T08:13:05Z"
 * }
 */
object ML {

    private const val TAG = "ML"
    private const val INTERVAL_MS = 1500L
    private const val BATCH_LIMIT = 200           // per tick
    private const val MAX_IMAGE_BYTES = 10 * 1024 * 1024 // 10MB

    private lateinit var appContext: Context
    private var datasetId: String = "routex-2025-busan"

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var job: Job? = null
    private val running = AtomicBoolean(false)

    // Firebase handles
    private val db get() = Firebase.firestore
    private val storage get() = Firebase.storage

    //Added

    private const val RETRY_BACKOFF_MS = 5000L

    private val nextRetryAt = java.util.concurrent.ConcurrentHashMap<String, Long>()


    // Your model wrapper
    @Volatile private var infer: ResnetInference? = null
    private fun ensureInfer(): ResnetInference {
        val i = infer; if (i != null) return i
        val ni = ResnetInference(appContext)
        infer = ni
        return ni
    }

    // avoid re-trying the same sample too often in one session
    private val recentlyTried = java.util.Collections.newSetFromMap(
        java.util.concurrent.ConcurrentHashMap<String, Boolean>()
    )

    /** Start background loop */
    fun start(context: Context, datasetId: String) {
        this.appContext = context.applicationContext
        this.datasetId = datasetId

        if (running.getAndSet(true)) {
            Log.i(TAG, "Already running; start() ignored.")
            return
        }

        job = scope.launch {
            Log.i(TAG, "ML labeling started for dataset=$datasetId")
            while (isActive && running.get()) {
                try {
                    tickOnce()
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.e(TAG, "tickOnce error: ${e.message}", e)
                }
                delay(INTERVAL_MS)
            }
        }
    }

    /** Stop background loop */
    fun stop() {
        running.set(false)
        job?.cancel()
        job = null
        Log.i(TAG, "ML labeling stopped.")
    }

    /** One cycle: find unlabeled samples, run inference, write labels */
    private suspend fun tickOnce() {
        val samples = db.collection("tdmlDatasets").document(datasetId)
            .collection("samples")
            .orderBy("metadata.capturedAt", com.google.firebase.firestore.Query.Direction.DESCENDING)
            .limit(BATCH_LIMIT.toLong())
            .get().await()

        if (samples.isEmpty) return

        var labeled = 0
        val infer = ensureInfer()

        for (doc in samples.documents) {
            if (!running.get()) break

            val sampleId = doc.id

            val now = android.os.SystemClock.elapsedRealtime()
            val nr = nextRetryAt[sampleId]
            if (nr != null && now < nr) continue

            if (recentlyTried.contains(sampleId)) continue

            // skip if label already exists
            val labelRef = db.document("tdmlDatasets/$datasetId/labels/$sampleId")
            if (labelRef.get().await().exists()) {
                recentlyTried.add(sampleId)
                nextRetryAt.remove(sampleId)
                continue
            }

            val imgHrefAny = doc.get("inputs.image.href")
            if (imgHrefAny !is String) {
                //recentlyTried.add(sampleId)
                nextRetryAt[sampleId] = android.os.SystemClock.elapsedRealtime() + RETRY_BACKOFF_MS
                continue
            }
            val imgHref = imgHrefAny as String

            val poseHref = (doc.get("inputs.geopose.href") as? String) ?: ""

            val poseBytes = Firebase.storage.getReferenceFromUrl(poseHref)
                .getBytes(64 * 1024).await()
            val pose = parseGeoPose(poseBytes)

            try {
                // download image
                val bytes = storage.getReferenceFromUrl(imgHref).getBytes(MAX_IMAGE_BYTES.toLong()).await()
                val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                if (bmp == null) {
                    //recentlyTried.add(sampleId)
                    nextRetryAt[sampleId] = android.os.SystemClock.elapsedRealtime() + RETRY_BACKOFF_MS
                    continue
                }

                // run your ResnetInference (top-1)
                val top1 = ensureInfer().classifyWithProbs(bmp, topK = 1).firstOrNull()
                if (top1 == null) {
                    //recentlyTried.add(sampleId)
                    nextRetryAt[sampleId] = android.os.SystemClock.elapsedRealtime() + RETRY_BACKOFF_MS
                    continue
                }
                val predClass = top1.label
                val score = top1.prob.toDouble()

                // write TDML label doc (annotations only)
                val labelDoc = mapOf(
                    "sampleId" to sampleId,
                    "annotations" to listOf(
                        mapOf(
                            "type" to "classification",
                            "target" to "road_surface",
                            "class" to predClass,
                            "score" to score
                        )
                    ),
                    "createdAt" to com.google.firebase.Timestamp.now(),

                    "lat" to pose.lat, "lon" to pose.lon, "h" to pose.h,
                    "qx" to pose.qx, "qy" to pose.qy, "qz" to pose.qz, "qw" to pose.qw,

                    // NEW: direct hrefs to resources (optional but handy)
                    "imageHref" to imgHref,
                    "geoposeHref" to poseHref
                )

                labelRef.set(labelDoc).await()
                nextRetryAt.remove(sampleId)


                labeled++
                recentlyTried.add(sampleId)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to label $sampleId: ${e.message}")
                // don’t mark tried; we’ll retry next tick
            }
        }

        if (labeled > 0) Log.i(TAG, "ML tick: labeled $labeled sample(s).")
    }
}
