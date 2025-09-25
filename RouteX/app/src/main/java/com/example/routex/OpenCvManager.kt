package com.example.routex

import android.util.Log

/**
 * Optional OpenCV loader. Safe to call even if the OpenCV AAR is not on the classpath.
 * We use reflection so there is no compile-time dependency.
 */
object OpenCvManager {
    @Volatile private var initialized = false
    @Volatile var available: Boolean = false
        private set

    @Synchronized fun ensureInit() {
        if (initialized) return
        available = try {
            val cls = Class.forName("org.opencv.android.OpenCVLoader")
            val m = cls.getMethod("initLocal")
            (m.invoke(null) as? Boolean) == true
        } catch (_: Throwable) { false }
        initialized = true
        Log.i("OpenCV", if (available) "OpenCV loaded" else "OpenCV not available (using fallback)")
    }
}
