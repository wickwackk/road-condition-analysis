package com.example.routex

import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.Query
import com.google.firebase.ktx.Firebase
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.firestore.DocumentSnapshot
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow

class FirestoreCapturesRepository {

    private val col = Firebase.firestore.collection("captures")

    // --- robust mapper: tolerates Timestamp or Long; "conf" or "confidence" ---
    private fun mapDoc(doc: DocumentSnapshot): CaptureDoc {
        val label     = doc.getString("label") ?: ""
        val conf      = doc.getDouble("conf") ?: doc.getDouble("confidence") ?: 0.0
        val lat       = doc.getDouble("lat") ?: 0.0
        val lon       = doc.getDouble("lon") ?: 0.0

        // SAFELY handle mixed types for "ts"
        val tsAny = doc.get("ts")
        val tsMillis = when (tsAny) {
            is com.google.firebase.Timestamp -> tsAny.toDate().time
            is java.util.Date               -> tsAny.time
            is Number                       -> tsAny.toLong()
            else                            -> 0L
        }

        val imagePath = doc.getString("imagePath") ?: ""
        val txtPath   = doc.getString("txtPath")
        val deviceId  = doc.getString("deviceId")
        val modelVer  = doc.getString("modelVer")

        return CaptureDoc(
            label = label,
            conf = conf,
            lat = lat,
            lon = lon,
            ts = tsMillis,
            imagePath = imagePath,
            txtPath = txtPath,
            deviceId = deviceId,
            modelVer = modelVer
        )
    }

    // Fetch once
    fun fetchAll(onResult: (Result<List<CaptureDoc>>) -> Unit) {
        col.get()
            .addOnSuccessListener { snap ->
                val list = snap.documents.map { mapDoc(it) }.sortedBy { it.ts }
                onResult(Result.success(list))
            }
            .addOnFailureListener { e -> onResult(Result.failure(e)) }
    }

    fun streamAll() = callbackFlow<List<CaptureDoc>> {
        val reg = col.addSnapshotListener { snap, err ->
            if (err != null) { close(err); return@addSnapshotListener }
            val list = (snap?.documents?.map { mapDoc(it) } ?: emptyList())
                .sortedBy { it.ts }
            trySend(list)
        }
        awaitClose { reg.remove() }
    }

    fun saveCapture(capture: CaptureDoc, onResult: (Result<Void?>) -> Unit) {
        col.add(capture)
            .addOnSuccessListener { onResult(Result.success(null)) }
            .addOnFailureListener { e -> onResult(Result.failure(e)) }
    }

}

//fun fetchLatest(limit: Long = 500, onResult: (Result<List<CaptureDoc>>) -> Unit) {
//    col.orderBy("ts", Query.Direction.DESCENDING).limit(limit)
//        .get()
//        .addOnSuccessListener { s ->
//            onResult(Result.success(s.documents.mapNotNull { it.toObject(CaptureDoc::class.java) }))
//        }
//        .addOnFailureListener { e -> onResult(Result.failure(e)) }
//}
//
//fun streamLatest(limit: Long = 500) = callbackFlow<List<CaptureDoc>> {
//    val reg = col.orderBy("ts", Query.Direction.DESCENDING).limit(limit)
//        .addSnapshotListener { snap, _ ->
//            trySend(snap?.documents?.mapNotNull { it.toObject(CaptureDoc::class.java) } ?: emptyList())
//        }
//    awaitClose { reg.remove() }
//}