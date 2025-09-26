package com.example.routex

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Bundle
import android.view.View
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.annotation.ColorInt
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.*
import com.google.android.material.card.MaterialCardView
import com.google.maps.android.SphericalUtil
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.util.Locale
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

    // cache a tiny colored-dot bitmap per color
    private val dotIconCache = mutableMapOf<Int, BitmapDescriptor>()

    // UI refs
    private lateinit var btnLegendToggle: View
    private lateinit var legendCard: MaterialCardView

    private lateinit var infoCard: MaterialCardView
    private lateinit var infoLine1: TextView
    private lateinit var infoLine2: TextView

    // track current info target so a second tap hides it
    private var currentInfoKey: String? = null

    

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_map)

        (supportFragmentManager.findFragmentById(R.id.map) as SupportMapFragment)
            .getMapAsync(this)

        // Back arrow (FAB) -> return to previous screen
        findViewById<com.google.android.material.floatingactionbutton.FloatingActionButton>(R.id.fabBack)
            .setOnClickListener { finish() }

        // Bind legend toggle UI
        btnLegendToggle = findViewById(R.id.btnLegendToggle)
        btnLegendToggle.setOnClickListener {
            legendCard.visibility = if (legendCard.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
        legendCard = findViewById(R.id.legendCard)

        btnLegendToggle.setOnClickListener {
            legendCard.visibility =
                if (legendCard.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        // Bind info card
        infoCard = findViewById(R.id.infoCard)
        infoLine1 = findViewById(R.id.infoLine1)
        infoLine2 = findViewById(R.id.infoLine2)

        // Tapping the info card itself also hides it
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

        

        // Hide info card on map taps or when user starts moving the camera
        map?.setOnMapClickListener { hideInfoCard() }
        map?.setOnCameraMoveStartedListener { _ -> hideInfoCard() }

        // Polyline tap -> show average confidence in the bottom info card
        map?.setOnPolylineClickListener { poly ->
            val meta = poly.tag as? PolyMeta ?: return@setOnPolylineClickListener
            val key = "poly:${meta.label}:${"%.2f".format(meta.avgConf)}:${poly.id}"
            if (currentInfoKey == key && infoCard.visibility == View.VISIBLE) {
                hideInfoCard()
            } else {
                showInfoCard(
                    classification = prettyLabel(meta.label),
                    confidence = meta.avgConf
                )
                currentInfoKey = key
            }
        }

        // Marker tap -> show point confidence in the bottom info card
        map?.setOnMarkerClickListener { mk ->
            val meta = mk.tag as? DotMeta ?: return@setOnMarkerClickListener false
            val key = "dot:${meta.label}:${"%.2f".format(meta.conf)}:${mk.position.latitude}:${mk.position.longitude}"
            if (currentInfoKey == key && infoCard.visibility == View.VISIBLE) {
                hideInfoCard()
            } else {
                showInfoCard(
                    classification = prettyLabel(meta.label),
                    confidence = meta.conf
                )
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

    // ---------------- drawing ----------------

    private fun drawRoadOverlay(items: List<CaptureDoc>) {
        val m = map ?: return
        hideInfoCard() // ensure no stale info visible when overlays refresh
        m.clear()

        val valid = items.filter { !it.lat.isNaN() && !it.lon.isNaN() }
        if (valid.isEmpty()) return

        // Group by the precise, canonical label (e.g., "asphalt_good")
        val byLabel = valid.groupBy { canonicalLabel(it.label) }

        var firstPoint: LatLng? = null
        val railSeen = hashSetOf<String>()  // per-frame rail de-dupe

        for ((label, docs) in byLabel) {
            if (docs.isEmpty()) continue

            val pts = docs.mapIndexed { idx, d ->
                Node(
                    idx = idx,
                    ll = LatLng(d.lat, d.lon),
                    conf = d.conf,
                    h = d.h,
                    x = d.qx,
                    y = d.qy,
                    z = d.qz,
                    w = d.qw
                )
            }

            val color = colorForLabel(label)
            val icon = dotIconForColor(color)

            // Dots (clickable)
            pts.forEach { n ->
                val euler = quaternionToEuler(n.x, n.y, n.z, n.w)
                val heading = euler.yaw
                val tiltPitch = euler.pitch
                val tiltRoll = euler.roll

                val arrowLengthM = 10.0

                // Apply yaw for map heading
                val arrowEnd = SphericalUtil.computeOffset(n.ll, arrowLengthM, heading)
                m.addPolyline(
                    PolylineOptions()
                        .add(n.ll, arrowEnd)
                        .color(color)
                        .width(4f)
                        .zIndex(LINE_Z + 1)
                )

                // Optional: adjust FOV cone points using pitch/roll
                val fovAngle = 30.0
                val fovLength = 15.0

                // simple approximation: tilt the left/right points by pitch (north/south)
                val left = SphericalUtil.computeOffset(n.ll, fovLength, heading - fovAngle / 2 + tiltPitch)
                val right = SphericalUtil.computeOffset(n.ll, fovLength, heading + fovAngle / 2 + tiltPitch)
                m.addPolygon(
                    PolygonOptions()
                        .add(n.ll, left, right)
                        .fillColor(withAlpha(color, 50))
                        .strokeColor(color)
                        .strokeWidth(2f)
                        .zIndex(LINE_Z)
                )
            }




            // Keep one undirected edge between any pair
            val addedPairs = hashSetOf<Pair<Int, Int>>()

            // For each node, connect to nearest neighbor and (optionally) a second sufficiently different angle
            for (i in pts.indices) {
                val a = pts[i]

                val cands = buildList {
                    for (j in pts.indices) if (j != i) {
                        val b = pts[j]
                        val dist = SphericalUtil.computeDistanceBetween(a.ll, b.ll)
                        if (dist <= CONNECT_THRESHOLD_M) {
                            val heading = SphericalUtil.computeHeading(a.ll, b.ll) // [-180, 180]
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

                // 2) optional second with angle separation
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

        // Focus once on the first point we drew
        firstPoint?.let { fp ->
            if (!firstCameraSet) {
                firstCameraSet = true
                map?.animateCamera(CameraUpdateFactory.newLatLngZoom(fp, 15f))
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

        // Rail de-dupe: skip if a nearly-parallel edge already occupies this cell
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

    /**
     * Canonicalize labels to the *exact* seven classes you care about.
     * Any unknowns fall back to "asphalt_regular" (neutral gray).
     */
    private fun canonicalLabel(raw: String?): String {
        val s = raw?.trim()?.lowercase(Locale.US) ?: ""
        return when {
            s.startsWith("asphalt_good")    -> "asphalt_good"
            s.startsWith("asphalt_regular") -> "asphalt_regular"
            s.startsWith("asphalt_bad")     -> "asphalt_bad"

            s.startsWith("paved_regular")   -> "paved_regular"
            s.startsWith("paved_bad")       -> "paved_bad"

            s.startsWith("unpaved_regular") -> "unpaved_regular"
            s.startsWith("unpaved_bad")     -> "unpaved_bad"

            // If only category comes through (e.g., "asphalt"), treat as "regular"
            s.startsWith("asphalt")         -> "asphalt_regular"
            s.startsWith("paved")           -> "paved_regular"
            s.startsWith("unpaved")         -> "unpaved_regular"

            else                            -> "asphalt_regular"
        }
    }

    /** Map each canonical label to the exact color from colors.xml. */
    @ColorInt
    private fun colorForLabel(canonical: String): Int = when (canonical) {
        "asphalt_good"    -> ContextCompat.getColor(this, R.color.asphalt_good)      // dark gray
        "asphalt_regular" -> ContextCompat.getColor(this, R.color.asphalt_regular)   // gray
        "asphalt_bad"     -> ContextCompat.getColor(this, R.color.asphalt_bad)       // light gray

        "paved_regular"   -> ContextCompat.getColor(this, R.color.paved_regular)     // yellow
        "paved_bad"       -> ContextCompat.getColor(this, R.color.paved_bad)         // yellowish orange

        "unpaved_regular" -> ContextCompat.getColor(this, R.color.unpaved_regular)   // dark orange
        "unpaved_bad"     -> ContextCompat.getColor(this, R.color.unpaved_bad)       // reddish orange

        else              -> ContextCompat.getColor(this, R.color.asphalt_regular)
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

    private data class EulerAngles(val yaw: Double, val pitch: Double, val roll: Double)

    private fun quaternionToEuler(x: Double, y: Double, z: Double, w: Double): EulerAngles {
        // roll (x-axis rotation)
        val t0 = +2.0 * (w * x + y * z)
        val t1 = +1.0 - 2.0 * (x * x + y * y)
        val roll = Math.toDegrees(Math.atan2(t0, t1))

        // pitch (y-axis rotation)
        var t2 = +2.0 * (w * y - z * x)
        t2 = t2.coerceIn(-1.0, 1.0)
        val pitch = Math.toDegrees(Math.asin(t2))

        // yaw (z-axis rotation)
        val t3 = +2.0 * (w * z + x * y)
        val t4 = +1.0 - 2.0 * (y * y + z * z)
        val yaw = Math.toDegrees(Math.atan2(t3, t4))

        return EulerAngles(yaw, pitch, roll)
    }

    private data class Node(val idx: Int, val ll: LatLng, val conf: Double, val h: Double, val x: Double, val y: Double, val z: Double , val w: Double)
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

    // --------- Info card helpers ---------

    private fun showInfoCard(classification: String, confidence: Double) {
        infoLine1.text = "Classification: $classification"
        infoLine2.text = "Confidence: ${"%.2f".format(confidence)}"
        infoCard.visibility = View.VISIBLE
    }

    private fun hideInfoCard() {
        infoCard.visibility = View.GONE
        currentInfoKey = null
    }

    private fun prettyLabel(canonical: String): String = when (canonicalLabel(canonical)) {
        "asphalt_good"    -> "Asphalt - Good"
        "asphalt_regular" -> "Asphalt - Regular"
        "asphalt_bad"     -> "Asphalt - Bad"
        "paved_regular"   -> "Paved - Regular"
        "paved_bad"       -> "Paved - Bad"
        "unpaved_regular" -> "Unpaved - Regular"
        "unpaved_bad"     -> "Unpaved - Bad"
        else              -> "Unclassified"
    }
}