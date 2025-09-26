package com.example.routex

import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.Query
import com.google.firebase.ktx.Firebase
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.firestore.DocumentSnapshot
import com.google.firebase.storage.FirebaseStorage
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import org.json.JSONObject

class FirestoreCapturesRepository {

    private val col = Firebase.firestore.collection("captures")
    private val storage = FirebaseStorage.getInstance().reference
    // --- robust mapper: tolerates Timestamp or Long; "conf" or "confidence" ---
    private suspend fun mapDoc(doc: DocumentSnapshot): CaptureDoc {
        val label     = doc.getString("label") ?: ""
        val conf      = doc.getDouble("conf") ?: doc.getDouble("confidence") ?: 0.0


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

        val id = doc.getString("id")?:""
        val geoposePath = doc.getString("geoposePath") ?: ""
        var lat       =  0.0
        var lon       =  0.0
        var h       =  0.0
        var qx       =  0.0
        var qy       =  0.0
        var qz       =  0.0
        var qw       =  0.0

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
            } catch (e: Exception) { e.printStackTrace()
                lat = 35.1796   // Busan latitude
                lon = 129.0756  // Busan longitude
                h = 0.0
                qx = 0.0
                qy = 0.0
                qz = 0.0
                qw = 0.0}
        }
        return CaptureDoc(
            label = label,
            conf = conf,
            lat = lat,
            lon = lon,
            ts = tsMillis,
            imagePath = imagePath,
            txtPath = txtPath,
            deviceId = deviceId,
            modelVer = modelVer,

            id = id,
            geoposePath = geoposePath,
            h = h,
            qx = qx,
            qy = qy,
            qz = qz,
            qw = qw

        )
    }

    // Fetch once

    suspend fun fetchAll(): List<CaptureDoc> {
        val snap = col.get().await()
        return snap.documents.map { mapDoc(it) }.sortedBy { it.ts }
    }

//    fun fetchAll(onResult: (Result<List<CaptureDoc>>) -> Unit) {
//        col.get()
//            .addOnSuccessListener { snap ->
//                val list = snap.documents.map { mapDoc(it) }.sortedBy { it.ts }
//                onResult(Result.success(list))
//            }
//            .addOnFailureListener { e -> onResult(Result.failure(e)) }
//    }
    fun streamAll() = callbackFlow<List<CaptureDoc>> {
        val reg = col.addSnapshotListener { snap, err ->
            if (err != null) { close(err); return@addSnapshotListener }
            launch {
                val list = snap?.documents?.map { mapDoc(it) } ?: emptyList()
                trySend(list.sortedBy { it.ts })
            }
        }
        awaitClose { reg.remove() }
    }
//    fun streamAll() = callbackFlow<List<CaptureDoc>> {
//        val reg = col.addSnapshotListener { snap, err ->
//            if (err != null) { close(err); return@addSnapshotListener }
//            val list = (snap?.documents?.map { mapDoc(it) } ?: emptyList())
//                .sortedBy { it.ts }
//            trySend(list)
//        }
//        awaitClose { reg.remove() }
//    }

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