package com.example.routex

import android.content.Context
import android.util.Log
import com.google.firebase.FirebaseApp
import com.google.firebase.Timestamp
import com.google.firebase.firestore.SetOptions
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import com.google.firebase.storage.ktx.storage
import kotlinx.coroutines.*
import kotlinx.coroutines.tasks.await
import java.util.concurrent.atomic.AtomicBoolean

/**
 * TDML auto-generator:
 * - Call TDML.start(context, DATASET_ID) once (e.g., onCreate).
 * - It will poll Storage every ~1.5s, detect new (img + GeoPOSE) pairs,
 *   and write TDML Sample docs to Firestore.
 *
 * Firestore:
 *   tdmlDatasets/{DATASET_ID}/samples/{sampleId}
 *
 * Storage (expected from your teammates):
 *   img/{sampleId}.jpg|png
 *   GeoPOSE/{sampleId}.geopose.json  (content-type: application/geopose+json)
 */
object TDML {

    private const val TAG = "TDML"
    private const val DEFAULT_INTERVAL_MS = 1500L     // ~1.5s
    private const val LIST_LIMIT = 1000               // per page max
    private const val MAX_TOTAL_PER_TICK = 2000       // safety: limit per cycle

    private var datasetId: String = "routex-2025-busan"
    private lateinit var appContext: Context

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var job: Job? = null
    private val running = AtomicBoolean(false)

    // Firebase handles
    private val db get() = Firebase.firestore
    private val storage get() = Firebase.storage
    private val bucket: String by lazy {
        FirebaseApp.getInstance().options.storageBucket
            ?: error("No storage bucket in google-services.json")
    }

    // LRU-ish memory of processed sampleIds to avoid re-work between ticks
    // Keeps the most recent ~100k ids; adjust if needed.
    private val processed = object : LinkedHashMap<String, Boolean>(/*initialCapacity*/1024, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Boolean>?): Boolean {
            return this.size > 100_000
        }
    }

    /**
     * Start the background TDML generator.
     * Call this once (e.g., in Application or Activity onCreate).
     */
    fun start(context: Context, datasetId: String, intervalMs: Long = DEFAULT_INTERVAL_MS) {
        this.appContext = context.applicationContext
        this.datasetId = datasetId

        if (running.getAndSet(true)) {
            Log.i(TAG, "Already running; start() ignored.")
            return
        }

        job = scope.launch {
            Log.i(TAG, "TDML polling started for dataset=$datasetId, interval=${intervalMs}ms")
            while (isActive && running.get()) {
                try {
                    tickOnce()
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.e(TAG, "tickOnce error: ${e.message}", e)
                }
                delay(intervalMs)
            }
        }
    }

    /** Stop the background loop (optional). */
    fun stop() {
        running.set(false)
        job?.cancel()
        job = null
        Log.i(TAG, "TDML polling stopped.")
    }

    /** One polling cycle: list img/, find pairs, write TDML docs for new ones. */
    private suspend fun tickOnce() {
        val imgRoot = storage.reference.child("img")
        val poseRoot = storage.reference.child("GeoPOSE")

        var pageToken: String? = null
        var processedThisTick = 0

        while (processedThisTick < MAX_TOTAL_PER_TICK) {
            val page = if (pageToken == null) {
                imgRoot.list(LIST_LIMIT).await()
            } else {
                imgRoot.list(LIST_LIMIT, pageToken!!).await()
            }

            if (page.items.isEmpty()) break

            for (imgRef in page.items) {
                if (processedThisTick >= MAX_TOTAL_PER_TICK) break

                val fileName = imgRef.name // e.g., id123.jpg
                val sampleId = fileName.substringBeforeLast(".")
                val ext = fileName.substringAfterLast(".", "").lowercase()

                // quick skip if we've recently handled this id
                val already = synchronized(processed) { processed.containsKey(sampleId) }
                if (already) continue

                // ensure GeoPose exists
                val poseRef = poseRoot.child("$sampleId.geopose.json")
                val poseExists = try { poseRef.metadata.await(); true } catch (_: Exception) { false }
                if (!poseExists) continue

                // check if TDML sample doc already exists (idempotent)
                val sampleDocRef = db.document("tdmlDatasets/$datasetId/samples/$sampleId")
                val exists = sampleDocRef.get().await().exists()
                if (exists) {
                    // remember as processed so we don't keep rechecking
                    synchronized(processed) { processed[sampleId] = true }
                    continue
                }

                // compose gs:// hrefs
                val imgHref  = "gs://$bucket/img/$fileName"
                val poseHref = "gs://$bucket/GeoPOSE/$sampleId.geopose.json"
                val mediaType = if (ext == "png") "image/png" else "image/jpeg"

                // write TDML Sample doc (strict TDML; no status; no labels)
                val doc = mapOf(
                    "id" to sampleId,
                    "inputs" to mapOf(
                        "image"   to mapOf("href" to imgHref,  "mediaType" to mediaType),
                        "geopose" to mapOf("href" to poseHref, "mediaType" to "application/geopose+json", "version" to "1.0")
                    ),
                    "metadata" to mapOf(
                        "capturedAt" to Timestamp.now(),
                        "source" to "RouteX-Android"
                    ),
                    "task" to mapOf("problemType" to "classification", "target" to "road_surface"),
                    "schema" to "OGC TrainingDML-AI JSON 1.0.0"
                )

                sampleDocRef.set(doc, SetOptions.merge()).await()

                synchronized(processed) { processed[sampleId] = true }
                processedThisTick++
            }

            pageToken = page.pageToken
            if (pageToken == null) break
        }

        if (processedThisTick > 0) {
            Log.i(TAG, "TDML tick: created/confirmed $processedThisTick sample docs.")
        }
    }
}
