package com.example.routex

data class CaptureDoc(
    val label: String = "",
    val conf: Double = 0.0,
    val lat: Double = 0.0,
    val lon: Double = 0.0,
    val ts: Long = 0L,
    val imagePath: String = "",
    val txtPath: String? = null,
    val deviceId: String? = null,
    val modelVer: String? = null
)
