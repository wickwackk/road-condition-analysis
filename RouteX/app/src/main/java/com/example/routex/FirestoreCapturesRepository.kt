package com.example.routex

import com.google.firebase.firestore.Query
import com.google.firebase.ktx.Firebase
import com.google.firebase.firestore.ktx.firestore
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow

class FirestoreCapturesRepository {
    private val col = Firebase.firestore.collection("captures")

    fun fetchAll(onResult: (Result<List<CaptureDoc>>) -> Unit) {
        col.get()
            .addOnSuccessListener { s ->
                val list = s.documents.mapNotNull { it.toObject(CaptureDoc::class.java) }
                    .sortedBy { it.ts }
                onResult(Result.success(list))
            }
            .addOnFailureListener { e -> onResult(Result.failure(e)) }
    }

    fun streamAll() = callbackFlow<List<CaptureDoc>> {
        val reg = col.addSnapshotListener { snap, _ ->
            val list = snap?.documents?.mapNotNull { it.toObject(CaptureDoc::class.java) }
                ?.sortedBy { it.ts } ?: emptyList()
            trySend(list)
        }
        awaitClose { reg.remove() }
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