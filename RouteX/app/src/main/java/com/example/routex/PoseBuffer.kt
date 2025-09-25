package com.example.routex

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/** Thread-safe rolling buffers for GNSS and quaternion orientation (keyed by boot-time ns). */
class PoseBuffer(private val cap: Int = 4096) {

    data class Gnss(
        val tBootNs: Long,      // elapsedRealtimeNanos()
        val tUtcMs: Long,       // wall-clock ms for ISO timestamp
        val lat: Double,        // degrees
        val lon: Double,        // degrees
        val h: Double?,         // meters (WGS84 ellipsoid) or null
        val accM: Float?        // horizontal accuracy (m)
    )

    data class Quat(
        val tBootNs: Long,      // elapsedRealtimeNanos()
        val x: Double, val y: Double, val z: Double, val w: Double
    )

    private val gnss = ArrayDeque<Gnss>(cap)
    private val quat = ArrayDeque<Quat>(cap)
    private val lock = Any()

    fun addGnss(s: Gnss) = synchronized(lock) {
        if (gnss.size == cap) gnss.removeFirst()
        gnss.addLast(s)
    }

    fun addQuat(qIn: Quat) = synchronized(lock) {
        // Normalize before storing to avoid drift from sensor noise
        val n = sqrt(qIn.x*qIn.x + qIn.y*qIn.y + qIn.z*qIn.z + qIn.w*qIn.w)
        val q = if (n > 0.0) Quat(qIn.tBootNs, qIn.x/n, qIn.y/n, qIn.z/n, qIn.w/n) else qIn
        if (quat.size == cap) quat.removeFirst()
        quat.addLast(q)
    }

    fun nearestGnss(tBootNs: Long): Gnss? = synchronized(lock) {
        if (gnss.isEmpty()) return null
        var best = gnss.first()
        var bestDt = abs(best.tBootNs - tBootNs)
        for (e in gnss) {
            val dt = abs(e.tBootNs - tBootNs)
            if (dt < bestDt) { best = e; bestDt = dt }
        }
        best
    }

    fun nearestQuat(tBootNs: Long): Quat? = synchronized(lock) {
        if (quat.isEmpty()) return null
        var best = quat.first()
        var bestDt = abs(best.tBootNs - tBootNs)
        for (e in quat) {
            val dt = abs(e.tBootNs - tBootNs)
            if (dt < bestDt) { best = e; bestDt = dt }
        }
        best
    }

    /** SLERP interpolation between neighbors around tBootNs (falls back to nearest). */
    fun interpQuat(tBootNs: Long): Quat? = synchronized(lock) {
        if (quat.isEmpty()) return null
        if (quat.size == 1) return quat.first()

        // Find bracketing samples a <= t <= b
        var a = quat.first()
        var b = quat.last()
        for (e in quat) {
            if (e.tBootNs <= tBootNs) a = e else { b = e; break }
        }
        val dt = (b.tBootNs - a.tBootNs).toDouble()
        if (dt <= 0.0) return nearestQuat(tBootNs)

        val u = ((tBootNs - a.tBootNs).toDouble() / dt).coerceIn(0.0, 1.0)
        val (x,y,z,w) = slerp(a, b, u)
        Quat(tBootNs, x, y, z, w)
    }

    private fun slerp(a: Quat, b: Quat, u: Double): DoubleArray {
        var ax=a.x; var ay=a.y; var az=a.z; var aw=a.w
        var bx=b.x; var by=b.y; var bz=b.z; var bw=b.w

        // Dot product
        var cos = ax*bx + ay*by + az*bz + aw*bw
        // Take shortest path
        if (cos < 0.0) { bx=-bx; by=-by; bz=-bz; bw=-bw; cos = -cos }
        // Clamp for numerical safety
        cos = max(-1.0, min(1.0, cos))

        // Nearly colinear -> lerp + normalize
        if (cos > 0.9995) {
            val x = ax + u*(bx-ax)
            val y = ay + u*(by-ay)
            val z = az + u*(bz-az)
            val w = aw + u*(bw-aw)
            val n = sqrt(x*x + y*y + z*z + w*w)
            return doubleArrayOf(x/n, y/n, z/n, w/n)
        }

        val theta = acos(cos)
        val sinTheta = sin(theta)
        val s1 = sin((1-u)*theta)/sinTheta
        val s2 = sin(u*theta)/sinTheta

        val x = s1*ax + s2*bx
        val y = s1*ay + s2*by
        val z = s1*az + s2*bz
        val w = s1*aw + s2*bw
        return doubleArrayOf(x, y, z, w)
    }
}
