package com.example.routex

import com.google.firebase.Timestamp

data class CaptureDoc(

    val label: String = "",
    val conf: Double = 0.0,
    val lat: Double = 0.0,
    val lon: Double = 0.0,
    val ts: Long = 0L,
    val imagePath: String = "",
    val txtPath: String? = null,
    val deviceId: String? = null,
    val modelVer: String? = null,

    val id: String = "",
    val geoposePath: String = "",
    val h: Double = 0.0,
    val qx: Double = 0.0,
    val qy: Double = 0.0,
    val qz: Double = 0.0,
    val qw: Double = 0.0
)
