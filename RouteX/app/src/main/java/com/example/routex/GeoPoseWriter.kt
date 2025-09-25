package com.example.routex

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
data class Position(val lat: Double, val lon: Double, val h: Double)

@Serializable
data class Quaternion(val x: Double, val y: Double, val z: Double, val w: Double)

@Serializable
data class YprAngles(val yaw: Double, val pitch: Double, val roll: Double) // degrees

@Serializable
data class Accuracy(
    @SerialName("posStdDevM") val posStdDevM: Double? = null,
    @SerialName("oriStdDevDeg") val oriStdDevDeg: Double? = null
)

/** GeoPose 1.0 – Basic Quaternion */
@Serializable
data class BasicQuaternionGeoPose(
    val standard: String = "OGC.GeoPose.1.0",
    @SerialName("referenceFrame") val referenceFrame: String = "EPSG:4979",
    val id: String? = null,
    val timestamp: String? = null,   // RFC-3339 UTC
    val position: Position,
    val quaternion: Quaternion,
    val accuracy: Accuracy? = null
)

/** GeoPose 1.0 – Basic YPR (optional; useful for QA) */
@Serializable
data class BasicYprGeoPose(
    val standard: String = "OGC.GeoPose.1.0",
    @SerialName("referenceFrame") val referenceFrame: String = "EPSG:4979",
    val id: String? = null,
    val timestamp: String? = null,
    val position: Position,
    @SerialName("yprAngles") val yprAngles: YprAngles,
    val accuracy: Accuracy? = null
)

object GeoPoseWriter {
    private val json = Json { prettyPrint = true; encodeDefaults = true }

    /** Write a standards-conformant Basic-Quaternion GeoPose JSON. */
    fun writeQuaternion(
        outFile: File,
        id: String,
        timestampISO: String,
        lat: Double, lon: Double, h: Double,
        quatX: Double, quatY: Double, quatZ: Double, quatW: Double,
        posStd: Double? = null, oriStdDeg: Double? = null
    ) {
        outFile.parentFile?.mkdirs()
        val gp = BasicQuaternionGeoPose(
            id = id,
            timestamp = timestampISO,
            position = Position(lat, lon, h),
            quaternion = Quaternion(quatX, quatY, quatZ, quatW),
            accuracy = if (posStd != null || oriStdDeg != null) Accuracy(posStd, oriStdDeg) else null
        )
        outFile.writeText(json.encodeToString(BasicQuaternionGeoPose.serializer(), gp))
    }

    /** Optional helper if you also want to emit a Basic-YPR sibling for debugging. */
    fun writeYpr(
        outFile: File,
        id: String,
        timestampISO: String,
        lat: Double, lon: Double, h: Double,
        yawDeg: Double, pitchDeg: Double, rollDeg: Double,
        posStd: Double? = null, oriStdDeg: Double? = null
    ) {
        outFile.parentFile?.mkdirs()
        val gp = BasicYprGeoPose(
            id = id,
            timestamp = timestampISO,
            position = Position(lat, lon, h),
            yprAngles = YprAngles(yawDeg, pitchDeg, rollDeg),
            accuracy = if (posStd != null || oriStdDeg != null) Accuracy(posStd, oriStdDeg) else null
        )
        outFile.writeText(json.encodeToString(BasicYprGeoPose.serializer(), gp))
    }
}
