package com.example.routex

import android.util.Log
import com.google.firebase.firestore.DocumentSnapshot
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import com.google.firebase.storage.FirebaseStorage
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import org.json.JSONObject

class FirestoreCapturesRepository {

    private val TAG = "FirestoreCapturesRepo"

    private val samplesCol = Firebase.firestore
        .collection("tdmlDatasets")
        .document("routex-2025-busan")
        .collection("samples")

    private val labelsCol = Firebase.firestore
        .collection("tdmlDatasets")
        .document("routex-2025-busan")
        .collection("labels")

    private val storage = FirebaseStorage.getInstance().reference

    private suspend fun mapLabelWithSample(labelDoc: DocumentSnapshot): CaptureDoc? {
        val labelMap = labelDoc.data ?: run {
            Log.w(TAG, "Label document empty: ${labelDoc.id}")
            return null
        }

        val sampleId = labelMap["sampleId"] as? String ?: run {
            Log.w(TAG, "Label document missing sampleId: ${labelDoc.id}")
            return null
        }

        // --- Label info ---
        val annotations = labelMap["annotations"] as? List<Map<String, Any>> ?: emptyList()
        val firstAnnotation = annotations.firstOrNull() ?: emptyMap()

        val clazz = firstAnnotation["class"] as? String ?: ""
        val score = (firstAnnotation["score"] as? Number)?.toDouble() ?: 0.0
        val target = firstAnnotation["target"] as? String ?: ""

        // --- Sample info ---
        val sampleDoc = samplesCol.document(sampleId).get().await()
        val inputs = sampleDoc.get("inputs") as? Map<*, *> ?: emptyMap<Any, Any>()
        val image = inputs["image"] as? Map<*, *>
        val geopose = inputs["geopose"] as? Map<*, *>

        var lat = 0.0
        var lon = 0.0
        var h = 0.0
        var qx = 0.0
        var qy = 0.0
        var qz = 0.0
        var qw = 0.0

        val geoposeHref = geopose?.get("href") as? String ?: ""
        val geoposePath = geoposeHref.removePrefix("gs://routex-40302.firebasestorage.app/")

        if (geoposePath.isNotEmpty()) {
            try {
                val bytes = storage.child(geoposePath).getBytes(10_000).await()
                val json = JSONObject(bytes.toString(Charsets.UTF_8))
                json.optJSONObject("position")?.let {
                    lat = it.optDouble("lat", lat)
                    lon = it.optDouble("lon", lon)
                    h = it.optDouble("h", h)
                }
                json.optJSONObject("quaternion")?.let {
                    qx = it.optDouble("x", qx)
                    qy = it.optDouble("y", qy)
                    qz = it.optDouble("z", qz)
                    qw = it.optDouble("w", qw)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to fetch GeoPose for $sampleId", e)
            }
        }

        val imagePath = image?.get("href") as? String ?: ""

        // --- Logging ---
//        Log.d(TAG, "SampleID: $sampleId")
        Log.d(TAG, "Label: $clazz")
        Log.d(TAG, "Target: $target")
        Log.d(TAG, "Confidence Score: $score")
//        Log.d(TAG, "GeoPose Path: $geoposePath")
//        Log.d(TAG, "Position -> lat: $lat, lon: $lon, h: $h")
//        Log.d(TAG, "Quaternion -> x: $qx, y: $qy, z: $qz, w: $qw")
//        Log.d(TAG, "Image Path: $imagePath")
//        Log.d(TAG, "Device: ${sampleDoc.getString("metadata.source")}")

        return CaptureDoc(
            label = clazz,
            conf = score,
            lat = lat,
            lon = lon,
            ts = (labelMap["createdAt"] as? com.google.firebase.Timestamp)?.toDate()?.time ?: 0L,
            imagePath = imagePath,
            txtPath = null,
            deviceId = sampleDoc.getString("metadata.source"),
            modelVer = null,
            id = sampleId,
            geoposePath = geoposePath,
            h = h,
            qx = qx,
            qy = qy,
            qz = qz,
            qw = qw
        )
    }

    fun streamAll() = callbackFlow<List<CaptureDoc>> {
        val reg = labelsCol.addSnapshotListener { snap, err ->
            if (err != null) {
                Log.e(TAG, "Failed to listen to labels collection", err)
                close(err)
                return@addSnapshotListener
            }
            launch {
                val list = snap?.documents?.mapNotNull { mapLabelWithSample(it) } ?: emptyList()
                trySend(list.sortedBy { it.ts })
            }
        }
        awaitClose { reg.remove() }
    }
}
