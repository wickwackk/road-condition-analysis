package com.example.routex

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Bundle
import android.widget.Toast
import androidx.annotation.ColorInt
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.*
import com.google.maps.android.SphericalUtil
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.util.Locale
import kotlin.math.*

class MapActivity : AppCompatActivity(), OnMapReadyCallback {

    private val repo by lazy { FirestoreCapturesRepository() }

    private var map: GoogleMap? = null
    private var streamJob: Job? = null
    private var firstCameraSet = false

    // cache a tiny colored-dot bitmap per color
    private val dotIconCache = mutableMapOf<Int, BitmapDescriptor>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_map)

        (supportFragmentManager.findFragmentById(R.id.map) as SupportMapFragment)
            .getMapAsync(this)

        findViewById<com.google.android.material.floatingactionbutton.FloatingActionButton>(R.id.fabBack)
            .setOnClickListener { finish() }
    }

    override fun onMapReady(googleMap: GoogleMap) {
        map = googleMap.apply {
            uiSettings.isZoomControlsEnabled = true
            uiSettings.isMapToolbarEnabled = false
        }
        enableMyLocationIfGranted()

        map?.setOnPolylineClickListener { poly ->
            val meta = poly.tag as? PolyMeta
            Toast.makeText(
                this,
                meta?.let { "${it.label} • avg conf=${"%.2f".format(it.avgConf)}" } ?: "segment",
                Toast.LENGTH_SHORT
            ).show()
        }

        map?.setOnMarkerClickListener { mk ->
            val meta = mk.tag as? DotMeta
            if (meta != null) {
                Toast.makeText(
                    this,
                    "${meta.label} • conf=${"%.2f".format(meta.conf)}",
                    Toast.LENGTH_SHORT
                ).show()
                true
            } else false
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

    // ---------------- drawing ----------------

    private fun drawRoadOverlay(items: List<CaptureDoc>) {
        val m = map ?: return
        m.clear()

        val valid = items.filter { !it.lat.isNaN() && !it.lon.isNaN() }
        if (valid.isEmpty()) return

        val byLabel = valid.groupBy { normalizeLabel(it.label) }

        var firstPoint: LatLng? = null

        // rail de-dup set for this draw pass
        val railSeen = hashSetOf<String>()

        for ((label, docs) in byLabel) {
            if (docs.isEmpty()) continue

            val pts = docs.mapIndexed { idx, d ->
                Node(
                    idx = idx,
                    ll = LatLng(d.lat, d.lon),
                    conf = d.conf
                )
            }

            // Draw clickable dots
            val color = colorForLabel(label)
            val icon = dotIconForColor(color)
            pts.forEach { n ->
                val mk = m.addMarker(
                    MarkerOptions()
                        .position(n.ll)
                        .icon(icon)
                        .anchor(0.5f, 0.5f)
                        .zIndex(DOT_Z)
                        .flat(true)
                )
                mk?.tag = DotMeta(label, n.conf)
                if (firstPoint == null) firstPoint = n.ll
            }

            // Keep one undirected edge between any pair
            val addedPairs = hashSetOf<Pair<Int, Int>>()

            // For each node, choose the nearest candidate, then (optionally) one in opposite direction
            for (i in pts.indices) {
                val a = pts[i]

                val cands = buildList {
                    for (j in pts.indices) if (j != i) {
                        val b = pts[j]
                        val dist = SphericalUtil.computeDistanceBetween(a.ll, b.ll)
                        if (dist <= CONNECT_THRESHOLD_M) {
                            val heading = SphericalUtil.computeHeading(a.ll, b.ll) // [-180,180]
                            add(Nbor(j, dist, heading))
                        }
                    }
                }.sortedBy { it.dist }

                if (cands.isEmpty()) continue

                // 1) nearest
                val first = cands.first()
                addEdgeOnce(
                    m = m,
                    a = a,
                    b = pts[first.idx],
                    label = label,
                    color = color,
                    addedPairs = addedPairs,
                    railSeen = railSeen
                )

                // 2) a second neighbor that is far enough in angle from the first (to avoid same-direction twin rail)
                val second = cands.drop(1).firstOrNull { cand ->
                    angleSepDeg(first.headingDeg, cand.headingDeg) >= MIN_ANGLE_DEG
                }

                if (second != null) {
                    addEdgeOnce(
                        m = m,
                        a = a,
                        b = pts[second.idx],
                        label = label,
                        color = color,
                        addedPairs = addedPairs,
                        railSeen = railSeen
                    )
                }
            }
        }

        // focus once
        val fp = firstPoint
        if (!firstCameraSet && fp != null) {
            firstCameraSet = true
            map?.animateCamera(CameraUpdateFactory.newLatLngZoom(fp, 15f))
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

        // ---- rail dedupe: skip if a nearly-parallel edge already occupies this cell ----
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

    // ---------------- utils ----------------

    private fun angleSepDeg(a: Double, b: Double): Double {
        val d = abs(a - b)
        return if (d > 180.0) 360.0 - d else d
    }

    /** Make a coarse key for an edge’s “rail”: quantize midpoint + heading. */
    private fun railKey(a: LatLng, b: LatLng, label: String): String {
        val midLat = (a.latitude + b.latitude) / 2.0
        val midLng = (a.longitude + b.longitude) / 2.0

        // meters per degree
        val metersPerDegLat = 111_320.0
        val metersPerDegLng = metersPerDegLat * cos(Math.toRadians(midLat))

        // convert quantization from meters -> degrees at this latitude
        val cellDegLat = RAIL_CELL_M / metersPerDegLat
        val cellDegLng = RAIL_CELL_M / metersPerDegLng

        val gy = (midLat / cellDegLat).roundToLong()
        val gx = (midLng / cellDegLng).roundToLong()

        // heading bucket (ignore direction: 0..180)
        val h = abs(SphericalUtil.computeHeading(a, b))
        val heading180 = if (h > 180.0) 360.0 - h else h
        val hb = (heading180 / BEARING_BIN_DEG).roundToInt()

        return "$label|$hb|$gx|$gy"
    }

    private fun normalizeLabel(raw: String): String {
        val s = raw.lowercase(Locale.US)
        return when {
            s.startsWith("asphalt") -> "asphalt"
            s.startsWith("unpaved") -> "unpaved"
            s.startsWith("paved") -> "paved"
            else -> "other"
        }
    }

    private fun colorForLabel(normalized: String): Int = when (normalized) {
        "asphalt" -> 0xFFFF4444.toInt() // red
        "unpaved" -> 0xFFFF8800.toInt() // orange
        "paved"   -> 0xFF2962FF.toInt() // blue
        else      -> 0xFF555555.toInt() // gray
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
        val granted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        if (granted) map?.isMyLocationEnabled = true
    }

    override fun onDestroy() {
        streamJob?.cancel()
        super.onDestroy()
    }

    // ---------------- data & tunables ----------------

    private data class Node(val idx: Int, val ll: LatLng, val conf: Double)
    private data class Nbor(val idx: Int, val dist: Double, val headingDeg: Double)
    private data class PolyMeta(val label: String, val avgConf: Double)
    private data class DotMeta(val label: String, val conf: Double)

    companion object {
        // Distance within which points can connect (meters)
        private const val CONNECT_THRESHOLD_M = 200.0

        // Require the optional second neighbor to differ in heading by at least this (deg)
        private const val MIN_ANGLE_DEG = 100.0

        // Rail dedupe: spatial cell size and heading bucket (tune to be stricter/looser)
        private const val RAIL_CELL_M = 8.0            // merge rails within ~8 m
        private const val BEARING_BIN_DEG = 12.0       // merge rails if heading within ~12°

        // Dot appearance
        private const val DOT_DP = 16f
        private const val DOT_STROKE_DP = 2.5f
        private const val DOT_FILL_ALPHA = 0x66 // 0..255

        // Z and width
        private const val DOT_Z = 1000f
        private const val LINE_Z = 10f
        private const val LINE_WIDTH = 16f
    }
}
