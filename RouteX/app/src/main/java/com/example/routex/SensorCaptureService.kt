package com.example.routex

import android.Manifest
import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.*
import androidx.core.content.ContextCompat
import com.google.android.gms.location.*

class SensorCaptureService : Service(), SensorEventListener {

    companion object {
        private const val CHANNEL_ID = "routex.capture"
        private const val NOTIF_ID = 44

        fun start(ctx: Context) {
            val i = Intent(ctx, SensorCaptureService::class.java)
            ContextCompat.startForegroundService(ctx, i)
        }

        fun stop(ctx: Context) {
            ctx.stopService(Intent(ctx, SensorCaptureService::class.java))
        }

        @Volatile private var LAST_BUFFER: PoseBuffer? = null
        fun peekPoseBuffer(): PoseBuffer? = LAST_BUFFER
    }

    inner class LocalBinder : Binder() { fun getBuffer(): PoseBuffer = pose }
    private val localBinder = LocalBinder()

    private lateinit var sensorManager: SensorManager
    private lateinit var fused: FusedLocationProviderClient
    private val pose = PoseBuffer()

    private var locCallback: LocationCallback? = null

    override fun onBind(intent: Intent?): IBinder = localBinder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Keep running; WorkManager can still outlive the activity.
        return START_STICKY
    }

    override fun onCreate() {
        super.onCreate()
        LAST_BUFFER = pose

        createNotifChannel()
        startAsForeground(buildNotification())

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        fused = LocationServices.getFusedLocationProviderClient(this)

        // Rotation vector → quaternion
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)?.let { rot ->
            sensorManager.registerListener(this, rot, SensorManager.SENSOR_DELAY_GAME)
        }

        // GNSS 1 Hz
        if (hasLocationPermission()) subscribeLocationSafe()
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        locCallback?.let { fused.removeLocationUpdates(it) }
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    // ---- Foreground with TYPE (Android 14 compliant)
    private fun startAsForeground(notif: Notification) {
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(
                NOTIF_ID,
                notif,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION
            )
        } else if (Build.VERSION.SDK_INT >= 29) {
            startForeground(
                NOTIF_ID,
                notif,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION
            )
        } else {
            startForeground(NOTIF_ID, notif)
        }
    }

    // -- Sensors
    override fun onSensorChanged(e: SensorEvent) {
        if (e.sensor.type == Sensor.TYPE_ROTATION_VECTOR) {
            val q = FloatArray(4) // [w,x,y,z]
            SensorManager.getQuaternionFromVector(q, e.values)
            val tBoot = e.timestamp // ns, BOOTTIME
            pose.addQuat(
                PoseBuffer.Quat(
                    tBootNs = tBoot,
                    x = q[1].toDouble(),
                    y = q[2].toDouble(),
                    z = q[3].toDouble(),
                    w = q[0].toDouble()
                )
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* no-op */ }

    private fun hasLocationPermission(): Boolean {
        val fine = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED
        val coarse = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED
        return fine || coarse
    }

    @SuppressLint("MissingPermission")
    private fun subscribeLocationSafe() {
        try {
            val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L).build()
            val cb = object : LocationCallback() {
                override fun onLocationResult(r: LocationResult) {
                    val loc = r.lastLocation ?: return
                    val tBoot = if (Build.VERSION.SDK_INT >= 17 && loc.elapsedRealtimeNanos > 0)
                        loc.elapsedRealtimeNanos else SystemClock.elapsedRealtimeNanos()
                    pose.addGnss(
                        PoseBuffer.Gnss(
                            tBootNs = tBoot,
                            tUtcMs = System.currentTimeMillis(),
                            lat = loc.latitude,
                            lon = loc.longitude,
                            h = if (loc.hasAltitude()) loc.altitude else null,
                            accM = if (loc.hasAccuracy()) loc.accuracy else null
                        )
                    )
                }
            }
            locCallback = cb
            fused.requestLocationUpdates(req, cb, Looper.getMainLooper())
        } catch (_: SecurityException) {
            // permission revoked mid-session
        }
    }

    private fun createNotifChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "RouteX Capture", NotificationManager.IMPORTANCE_LOW)
            )
        }
    }

    private fun buildNotification(): Notification {
        return if (Build.VERSION.SDK_INT >= 26) {
            Notification.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_menu_compass)
                .setContentTitle("RouteX capture")
                .setContentText("Logging GNSS & orientation")
                .build()
        } else {
            Notification.Builder(this)
                .setSmallIcon(android.R.drawable.ic_menu_compass)
                .setContentTitle("RouteX capture")
                .setContentText("Logging GNSS & orientation")
                .build()
        }
    }
}
