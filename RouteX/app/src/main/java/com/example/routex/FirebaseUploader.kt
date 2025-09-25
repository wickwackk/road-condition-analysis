//package com.example.routex
//
//import android.content.Context
//import android.net.Uri
//import android.util.Log
//import com.google.firebase.firestore.FirebaseFirestore
//import com.google.firebase.storage.FirebaseStorage
//import kotlinx.coroutines.tasks.await
//import java.io.File
//
//
//class FirebaseUploader(context: Context) {
//    private val storage = FirebaseStorage.getInstance().reference
//    private val db = FirebaseFirestore.getInstance()
//
//
//    suspend fun uploadImageWithTxt(imageFile: File, txtContent: String, meta: Map<String, Any>) {
//        try {
//            val base = imageFile.nameWithoutExtension
//            val imgRef = storage.child("images/$base.jpg")
//            val txtRef = storage.child("images/$base.txt")
//
//
//// Upload image
//            imgRef.putFile(Uri.fromFile(imageFile)).await()
//
//
//// Upload TXT
//            val bytes = txtContent.toByteArray()
//            txtRef.putBytes(bytes).await()
//
//
//// Firestore doc
//            val doc = meta.toMutableMap().apply {
//                put("imagePath", imgRef.path)
//                put("txtPath", txtRef.path)
//            }
//            db.collection("captures").document(base).set(doc).await()
//        } catch (e: Exception) {
//            Log.e("FirebaseUploader", "Upload failed: ${e.message}")
//        }
//    }
//}

//NEWWWW
package com.example.routex

import android.net.Uri
import android.util.Log
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import kotlinx.coroutines.tasks.await
import java.io.File

class FirebaseUploader {
    private val storage = FirebaseStorage.getInstance().reference
    private val db = FirebaseFirestore.getInstance()

    suspend fun uploadImageWithGeoPose(
        imageFile: File,
        geoposeFile: File,
        meta: Map<String, Any>
    ) {
        val base = imageFile.nameWithoutExtension
        val imgRef = storage.child("img/$base.jpg")
        val gpRef = storage.child("GeoPOSE/$base.geopose.json")

        // Upload files
        imgRef.putFile(Uri.fromFile(imageFile)).await()
        gpRef.putFile(Uri.fromFile(geoposeFile)).await()

        // Firestore record
        val doc = meta.toMutableMap().apply {
            this["id"] = base
            this["imagePath"] = imgRef.path
            this["geoposePath"] = gpRef.path
        }
        db.collection("captures").document(base).set(doc).await()
        Log.i("FirebaseUploader", "Uploaded $base to ${imgRef.path} & ${gpRef.path}")
    }
}

