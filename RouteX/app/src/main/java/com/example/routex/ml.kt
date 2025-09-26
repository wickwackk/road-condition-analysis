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
    private const val INTERVAL_MS = 5000L
    private const val BATCH_LIMIT = 60           // per tick
    private const val MAX_IMAGE_BYTES = 10 * 1024 * 1024 // 10MB

    private lateinit var appContext: Context
    private var datasetId: String = "routex-2025-busan"

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var job: Job? = null
    private val running = AtomicBoolean(false)

    // Firebase handles
    private val db get() = Firebase.firestore
    private val storage get() = Firebase.storage

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
            .orderBy("metadata.capturedAt")
            .limit(BATCH_LIMIT.toLong())
            .get().await()

        if (samples.isEmpty) return

        var labeled = 0
        val infer = ensureInfer()

        for (doc in samples.documents) {
            if (!running.get()) break

            val sampleId = doc.id
            if (recentlyTried.contains(sampleId)) continue

            // skip if label already exists
            val labelRef = db.document("tdmlDatasets/$datasetId/labels/$sampleId")
            if (labelRef.get().await().exists()) {
                recentlyTried.add(sampleId)
                continue
            }

            val imgHrefAny = doc.get("inputs.image.href")
            if (imgHrefAny !is String) {
                recentlyTried.add(sampleId)
                continue
            }
            val imgHref = imgHrefAny as String

            try {
                // download image
                val bytes = storage.getReferenceFromUrl(imgHref).getBytes(MAX_IMAGE_BYTES.toLong()).await()
                val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                if (bmp == null) {
                    recentlyTried.add(sampleId)
                    continue
                }

                // run your ResnetInference (top-1)
                val top1 = ensureInfer().classifyWithProbs(bmp, topK = 1).firstOrNull()
                if (top1 == null) {
                    recentlyTried.add(sampleId)
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
                    "createdAt" to com.google.firebase.Timestamp.now()
                )

                labelRef.set(labelDoc).await()

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
