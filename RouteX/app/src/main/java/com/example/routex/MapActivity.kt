package com.example.routex
import coil.load
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.ColorInt
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.*
import com.google.android.material.card.MaterialCardView
import com.google.firebase.ktx.Firebase
import com.google.firebase.storage.ktx.storage

import com.google.maps.android.SphericalUtil
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.roundToLong

class MapActivity : AppCompatActivity(), OnMapReadyCallback {

    private val repo by lazy { FirestoreCapturesRepository() }

    private var map: GoogleMap? = null
    private var streamJob: Job? = null
    private var firstCameraSet = false

    private val dotIconCache = mutableMapOf<Int, BitmapDescriptor>()

    private lateinit var btnLegendToggle: View
    private lateinit var legendCard: MaterialCardView
    private lateinit var infoCard: MaterialCardView
    private lateinit var infoLine1: TextView
    private lateinit var infoLine2: TextView
    private lateinit var infoImage: ImageView

    private var currentInfoKey: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_map)

        (supportFragmentManager.findFragmentById(R.id.map) as SupportMapFragment)
            .getMapAsync(this)

        findViewById<com.google.android.material.floatingactionbutton.FloatingActionButton>(R.id.fabBack)
            .setOnClickListener { finish() }

        btnLegendToggle = findViewById(R.id.btnLegendToggle)
        legendCard = findViewById(R.id.legendCard)
        btnLegendToggle.setOnClickListener {
            legendCard.visibility = if (legendCard.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        infoCard = findViewById(R.id.infoCard)
        infoLine1 = findViewById(R.id.infoLine1)
        infoLine2 = findViewById(R.id.infoLine2)
        infoImage = findViewById(R.id.infoImage)

        infoCard.setOnClickListener { hideInfoCard() }
    }

    override fun onMapReady(googleMap: GoogleMap) {
        map = googleMap.apply {
            uiSettings.apply {
                isZoomControlsEnabled = true
                isMapToolbarEnabled = false
            }
        }
        enableMyLocationIfGranted()

        map?.setOnMapClickListener { hideInfoCard() }
        map?.setOnCameraMoveStartedListener { hideInfoCard() }

        map?.setOnPolylineClickListener { poly ->
            val meta = poly.tag as? PolyMeta ?: return@setOnPolylineClickListener
            val key = "poly:${meta.label}:${"%.2f".format(meta.avgConf)}:${poly.id}"
            if (currentInfoKey == key && infoCard.visibility == View.VISIBLE) hideInfoCard()
            else {
                showInfoCard(prettyLabel(meta.label), meta.avgConf, null)
                currentInfoKey = key
            }
        }

        map?.setOnMarkerClickListener { mk ->
            val meta = mk.tag as? DotMeta ?: return@setOnMarkerClickListener false
            val key = "dot:${meta.label}:${"%.2f".format(meta.conf)}:${mk.position.latitude}:${mk.position.longitude}"
            if (currentInfoKey == key && infoCard.visibility == View.VISIBLE) hideInfoCard()
            else {
                showInfoCard(prettyLabel(meta.label), meta.conf, meta.imagePath)
                currentInfoKey = key
            }
            true
        }

        startRealtimeStream()
    }

    private fun startRealtimeStream() {
        streamJob?.cancel()
        streamJob = lifecycleScope.launch {
            repo.streamAll().collectLatest { items ->
                drawRoadOverlay(items)
            }
        }
    }

    private fun drawRoadOverlay(items: List<CaptureDoc>) {
        val m = map ?: return
        hideInfoCard()
        m.clear()

        val valid = items.filter { !it.lat.isNaN() && !it.lon.isNaN() }
        if (valid.isEmpty()) return

        var firstPoint: LatLng? = null
        val railSeen = hashSetOf<String>()
        val byLabel = valid.groupBy { canonicalLabel(it.label) }

        for ((label, docs) in byLabel) {
            if (docs.isEmpty()) continue
            val color = colorForLabel(label)
            val icon = dotIconForColor(color)

            val pts = docs.mapIndexed { idx, d ->
                Node(idx, LatLng(d.lat, d.lon), d.conf, d.h, d.qx, d.qy, d.qz, d.qw, d.imagePath).also {
                    if (firstPoint == null) firstPoint = it.ll
                }
            }

            pts.forEach { n ->
                val marker = m.addMarker(
                    MarkerOptions()
                        .position(n.ll)
                        .icon(icon)
                        .anchor(0.5f, 0.5f)
                )
                marker?.tag = DotMeta(label, n.conf, n.imagePath)

                val euler = quaternionToEuler(n.x, n.y, n.z, n.w)
                val arrowEnd = SphericalUtil.computeOffset(n.ll, 10.0, euler.yaw)
                m.addPolyline(
                    PolylineOptions()
                        .add(n.ll, arrowEnd)
                        .color(color)
                        .width(4f)
                        .zIndex(LINE_Z + 1)
                )
            }

            val addedPairs = hashSetOf<Pair<Int, Int>>()
            for (i in pts.indices) {
                val a = pts[i]
                val cands = pts.indices.filter { it != i }.map { j ->
                    val b = pts[j]
                    val dist = SphericalUtil.computeDistanceBetween(a.ll, b.ll)
                    val heading = SphericalUtil.computeHeading(a.ll, b.ll)
                    Nbor(j, dist, heading)
                }.filter { it.dist <= CONNECT_THRESHOLD_M }.sortedBy { it.dist }

                if (cands.isEmpty()) continue

                addEdgeOnce(m, a, pts[cands.first().idx], label, color, addedPairs, railSeen)

                val second = cands.drop(1).firstOrNull { cand ->
                    angleSepDeg(cands.first().headingDeg, cand.headingDeg) >= MIN_ANGLE_DEG
                }
                if (second != null) addEdgeOnce(m, a, pts[second.idx], label, color, addedPairs, railSeen)
            }
        }

        firstPoint?.let {
            if (!firstCameraSet) {
                firstCameraSet = true
                map?.animateCamera(CameraUpdateFactory.newLatLngZoom(it, 15f))
            }
        }
    }

    private fun addEdgeOnce(
        m: GoogleMap,
        a: Node,
        b: Node,
        label: String,
        @ColorInt color: Int,
        addedPairs: HashSet<Pair<Int, Int>>,
        railSeen: HashSet<String>
    ) {
        val pairKey = if (a.idx < b.idx) a.idx to b.idx else b.idx to a.idx
        if (!addedPairs.add(pairKey)) return

        val key = railKey(a.ll, b.ll, label)
        if (!railSeen.add(key)) return

        val poly = m.addPolyline(
            PolylineOptions()
                .add(a.ll, b.ll)
                .color(color)
                .width(LINE_WIDTH)
                .startCap(RoundCap())
                .endCap(RoundCap())
                .jointType(JointType.ROUND)
                .clickable(true)
                .zIndex(LINE_Z)
        )
        poly.tag = PolyMeta(label, (a.conf + b.conf) / 2.0)
    }

    private fun angleSepDeg(a: Double, b: Double): Double {
        val d = abs(a - b)
        return if (d > 180.0) 360.0 - d else d
    }

    private fun railKey(a: LatLng, b: LatLng, label: String): String {
        val midLat = (a.latitude + b.latitude) / 2.0
        val midLng = (a.longitude + b.longitude) / 2.0

        val metersPerDegLat = 111_320.0
        val metersPerDegLng = metersPerDegLat * cos(Math.toRadians(midLat))

        val cellDegLat = RAIL_CELL_M / metersPerDegLat
        val cellDegLng = RAIL_CELL_M / metersPerDegLng

        val gy = (midLat / cellDegLat).roundToLong()
        val gx = (midLng / cellDegLng).roundToLong()

        val h = abs(SphericalUtil.computeHeading(a, b))
        val heading180 = if (h > 180.0) 360.0 - h else h
        val hb = (heading180 / BEARING_BIN_DEG).roundToInt()

        return "$label|$hb|$gx|$gy"
    }

    private fun canonicalLabel(raw: String?): String {
        val s = raw?.trim()?.lowercase() ?: ""
        return when {
            s.startsWith("asphalt_good") -> "asphalt_good"
            s.startsWith("asphalt_regular") -> "asphalt_regular"
            s.startsWith("asphalt_bad") -> "asphalt_bad"
            s.startsWith("paved_regular") -> "paved_regular"
            s.startsWith("paved_bad") -> "paved_bad"
            s.startsWith("unpaved_regular") -> "unpaved_regular"
            s.startsWith("unpaved_bad") -> "unpaved_bad"
            s.startsWith("asphalt") -> "asphalt_regular"
            s.startsWith("paved") -> "paved_regular"
            s.startsWith("unpaved") -> "unpaved_regular"
            else -> "asphalt_regular"
        }
    }

    @ColorInt
    private fun colorForLabel(canonical: String): Int = when (canonical) {
        "asphalt_good" -> ContextCompat.getColor(this, R.color.asphalt_good)
        "asphalt_regular" -> ContextCompat.getColor(this, R.color.asphalt_regular)
        "asphalt_bad" -> ContextCompat.getColor(this, R.color.asphalt_bad)
        "paved_regular" -> ContextCompat.getColor(this, R.color.paved_regular)
        "paved_bad" -> ContextCompat.getColor(this, R.color.paved_bad)
        "unpaved_regular" -> ContextCompat.getColor(this, R.color.unpaved_regular)
        "unpaved_bad" -> ContextCompat.getColor(this, R.color.unpaved_bad)
        else -> ContextCompat.getColor(this, R.color.asphalt_regular)
    }

    private fun withAlpha(@ColorInt color: Int, alpha: Int = DOT_FILL_ALPHA): Int =
        (color and 0x00FFFFFF) or ((alpha and 0xFF) shl 24)

    private fun dotIconForColor(@ColorInt base: Int): BitmapDescriptor =
        dotIconCache.getOrPut(base) {
            val size = dpToPx(DOT_DP)
            val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
            val c = Canvas(bmp)
            val r = size / 2f

            val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = withAlpha(base)
                style = Paint.Style.FILL
            }
            val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = base
                style = Paint.Style.STROKE
                strokeWidth = dpToPx(DOT_STROKE_DP).toFloat()
            }
            c.drawCircle(r, r, r - stroke.strokeWidth / 2f, fill)
            c.drawCircle(r, r, r - stroke.strokeWidth / 2f, stroke)

            BitmapDescriptorFactory.fromBitmap(bmp)
        }

    private fun dpToPx(dp: Float): Int =
        max(1, (dp * resources.displayMetrics.density).roundToInt())

    private fun enableMyLocationIfGranted() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED
        ) map?.isMyLocationEnabled = true
    }

    override fun onDestroy() {
        streamJob?.cancel()
        super.onDestroy()
    }

    private data class EulerAngles(val yaw: Double, val pitch: Double, val roll: Double)
    private fun quaternionToEuler(x: Double, y: Double, z: Double, w: Double): EulerAngles {
        val t0 = 2.0 * (w * x + y * z)
        val t1 = 1.0 - 2.0 * (x * x + y * y)
        val roll = Math.toDegrees(Math.atan2(t0, t1))

        var t2 = 2.0 * (w * y - z * x)
        t2 = t2.coerceIn(-1.0, 1.0)
        val pitch = Math.toDegrees(Math.asin(t2))

        val t3 = 2.0 * (w * z + x * y)
        val t4 = 1.0 - 2.0 * (y * y + z * z)
        val yaw = Math.toDegrees(Math.atan2(t3, t4))

        return EulerAngles(yaw, pitch, roll)
    }

    private data class Node(
        val idx: Int,
        val ll: LatLng,
        val conf: Double,
        val h: Double,
        val x: Double,
        val y: Double,
        val z: Double,
        val w: Double,
        val imagePath: String?
    )

    private data class Nbor(val idx: Int, val dist: Double, val headingDeg: Double)
    private data class PolyMeta(val label: String, val avgConf: Double)
    private data class DotMeta(val label: String, val conf: Double, val imagePath: String?)

    companion object {
        private const val CONNECT_THRESHOLD_M = 200.0
        private const val MIN_ANGLE_DEG = 100.0
        private const val RAIL_CELL_M = 8.0
        private const val BEARING_BIN_DEG = 12.0
        private const val DOT_DP = 16f
        private const val DOT_STROKE_DP = 2.5f
        private const val DOT_FILL_ALPHA = 0x66
        private const val DOT_Z = 1000f
        private const val LINE_Z = 10f
        private const val LINE_WIDTH = 16f
    }

    private fun showInfoCard(classification: String, confidence: Double, imagePath: String?) {
        infoLine1.text = "Classification: $classification"
        infoLine2.text = "Confidence: ${"%.2f".format(confidence)}"

        if (!imagePath.isNullOrEmpty()) {
            infoImage.visibility = View.VISIBLE

            // Check if path is a gs:// URL
            if (imagePath.startsWith("gs://")) {
                val storageRef = Firebase.storage.getReferenceFromUrl(imagePath)
                storageRef.downloadUrl
                    .addOnSuccessListener { uri ->
                        // Load image via Coil once we have the https URL
                        infoImage.load(uri) {
                            placeholder(R.drawable.ic_placeholder)
                            error(R.drawable.ic_error)
                        }
                    }
                    .addOnFailureListener { e ->
                        Log.e("MapActivity", "Failed to get download URL", e)
                        infoImage.setImageResource(R.drawable.ic_error)
                    }
            } else {
                // If it's already an https URL
                infoImage.load(imagePath) {
                    placeholder(R.drawable.ic_placeholder)
                    error(R.drawable.ic_error)
                }
            }

        } else {
            infoImage.visibility = View.GONE
        }

        infoCard.visibility = View.VISIBLE
    }

    private fun hideInfoCard() {
        infoCard.visibility = View.GONE
        currentInfoKey = null
    }

    private fun prettyLabel(canonical: String): String = when (canonicalLabel(canonical)) {
        "asphalt_good" -> "Asphalt - Good"
        "asphalt_regular" -> "Asphalt - Regular"
        "asphalt_bad" -> "Asphalt - Bad"
        "paved_regular" -> "Paved - Regular"
        "paved_bad" -> "Paved - Bad"
        "unpaved_regular" -> "Unpaved - Regular"
        "unpaved_bad" -> "Unpaved - Bad"
        else -> "Unclassified"
    }
}
