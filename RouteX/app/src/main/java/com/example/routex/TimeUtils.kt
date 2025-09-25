package com.example.routex

import android.os.SystemClock
import java.text.SimpleDateFormat
import java.util.*

object TimeUtils {
    private val isoFmtLocal = object : ThreadLocal<SimpleDateFormat>() {
        override fun initialValue(): SimpleDateFormat {
            return SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
                timeZone = TimeZone.getTimeZone("UTC")
            }
        }
    }

    fun bootNs(): Long = SystemClock.elapsedRealtimeNanos()
    fun iso8601UTC(epochMs: Long): String = isoFmtLocal.get()!!.format(Date(epochMs))
}
