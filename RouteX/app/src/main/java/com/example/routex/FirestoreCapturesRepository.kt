package com.example.routex

import android.util.Log
import com.google.firebase.Timestamp
import com.google.firebase.firestore.DocumentSnapshot
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch

class FirestoreCapturesRepository {

    private val TAG = "FirestoreCapturesRepo"

    // ✅ Your dataset path (updated)
    private val labelsCol = Firebase.firestore
        .collection("tdmlDatasets")
        .document("routex3-2025-busan")
        .collection("labels")

    /**
     * Convert a label document to a CaptureDoc.
     * Because all the necessary fields are already present,
     * this is now super fast — no extra Firestore or Storage calls!
     */
    private fun mapLabelDoc(labelDoc: DocumentSnapshot): CaptureDoc? {
        val data = labelDoc.data ?: return null

        // --- Annotation info ---
        val annotations = data["annotations"] as? List<Map<String, Any>> ?: emptyList()
        val firstAnnotation = annotations.firstOrNull() ?: emptyMap()

        val clazz = firstAnnotation["class"] as? String ?: ""
        val score = (firstAnnotation["score"] as? Number)?.toDouble() ?: 0.0
        val target = firstAnnotation["target"] as? String ?: ""

        // --- Direct fields from label ---
        val lat = (data["lat"] as? Number)?.toDouble() ?: Double.NaN
        val lon = (data["lon"] as? Number)?.toDouble() ?: Double.NaN
        val h = (data["h"] as? Number)?.toDouble() ?: 0.0
        val qx = (data["qx"] as? Number)?.toDouble() ?: 0.0
        val qy = (data["qy"] as? Number)?.toDouble() ?: 0.0
        val qz = (data["qz"] as? Number)?.toDouble() ?: 0.0
        val qw = (data["qw"] as? Number)?.toDouble() ?: 1.0

        val geoposeHref = data["geoposeHref"] as? String ?: ""
        val imageHref = data["imageHref"] as? String ?: ""
        val sampleId = data["sampleId"] as? String ?: labelDoc.id

        val ts = (data["createdAt"] as? Timestamp)?.toDate()?.time ?: 0L

        // Logging (optional)
        Log.d(TAG, "📍 Label: $clazz | Target: $target | Score: $score | lat=$lat lon=$lon")

        return CaptureDoc(
            label = clazz,
            conf = score,
            lat = lat,
            lon = lon,
            ts = ts,
            imagePath = imageHref,
            txtPath = null,
            deviceId = null,        // Not provided — can remove from model if unused
            modelVer = null,
            id = sampleId,
            geoposePath = geoposeHref,
            h = h,
            qx = qx,
            qy = qy,
            qz = qz,
            qw = qw
        )
    }

    /**
     * Stream all label docs in real time.
     * 🚀 This is now extremely fast because:
     * - We no longer fetch any sample docs.
     * - We don’t download any JSON files.
     * - All parsing is done from the label doc itself.
     */
    fun streamAll() = callbackFlow<List<CaptureDoc>> {
        val reg = labelsCol.addSnapshotListener { snap, err ->
            if (err != null) {
                Log.e(TAG, "❌ Failed to listen to labels collection", err)
                close(err)
                return@addSnapshotListener
            }

            launch {
                val list = snap?.documents?.mapNotNull { mapLabelDoc(it) } ?: emptyList()
                trySend(list.sortedBy { it.ts })
            }
        }
        awaitClose { reg.remove() }
    }
}
