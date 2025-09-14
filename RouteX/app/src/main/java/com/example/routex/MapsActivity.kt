package com.example.routex

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.routex.databinding.ActivityMainBinding
import com.example.routex.databinding.ActivityMapBinding
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.BitmapDescriptorFactory
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.LatLngBounds
import com.google.android.gms.maps.model.MarkerOptions
import com.google.android.gms.maps.model.PolylineOptions
import com.google.firebase.firestore.FirebaseFirestore


class MapsActivity : AppCompatActivity(), OnMapReadyCallback {

    private lateinit var mMap: GoogleMap
    private val db = FirebaseFirestore.getInstance()

    private lateinit var binding: ActivityMapBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize binding
        binding = ActivityMapBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val mapFragment = supportFragmentManager
            .findFragmentById(R.id.map) as SupportMapFragment
        mapFragment.getMapAsync(this)

        // Set back button click listener
        binding.btnBack.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            finish() // optional, close this activity
        }
    }

    override fun onMapReady(googleMap: GoogleMap) {
        mMap = googleMap

        // Enable zoom controls and gestures
        mMap.uiSettings.isZoomControlsEnabled = true
        mMap.uiSettings.isZoomGesturesEnabled = true
        mMap.uiSettings.isScrollGesturesEnabled = true
        mMap.uiSettings.isRotateGesturesEnabled = true
        mMap.uiSettings.isTiltGesturesEnabled = true

        // Enable my-location button (user must grant location permission)
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.ACCESS_FINE_LOCATION
            )
            == PackageManager.PERMISSION_GRANTED
        ) {
            mMap.isMyLocationEnabled = true
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.ACCESS_FINE_LOCATION),
                1234
            )
        }

        // Center map (Seoul fallback)
        val defaultLocation = LatLng(35.2340, 129.0807)
        mMap.moveCamera(CameraUpdateFactory.newLatLngZoom(defaultLocation, 15f))

        loadRoadData()
    }

    private fun loadRoadData() {
        db.collection("captures")
            .get()
            .addOnSuccessListener { snapshot ->
                val boundsBuilder = LatLngBounds.Builder()
                for (doc in snapshot) {
                    val lat = doc.getDouble("lat")
                    val lon = doc.getDouble("lon")
                    val label = doc.getString("label") ?: "unknown"

                    if (lat != null && lon != null && !lat.isNaN() && !lon.isNaN()) {
                        val point = LatLng(lat, lon)
                        boundsBuilder.include(point)

                        val marker = mMap.addMarker(
                            MarkerOptions()
                                .position(point)
                                .title(label)
                                .icon(BitmapDescriptorFactory.defaultMarker(getHueForLabel(label)))
                        )

                        // Attach full Firestore info as snippet
                        marker?.snippet = doc.data.toString()
                    }
                }

                // Move camera to include all markers
                val bounds = boundsBuilder.build()
                val padding = 100 // px
                mMap.moveCamera(CameraUpdateFactory.newLatLngBounds(bounds, padding))

                // Marker click listener to show info
                mMap.setOnMarkerClickListener { marker ->
                    marker.showInfoWindow()
                    true
                }
            }
            .addOnFailureListener { e ->
                android.util.Log.e("MapsActivity", "Error loading Firestore: ${e.message}")
            }
    }

    private fun loadRoaData() {
        db.collection("captures")
            .get()
            .addOnSuccessListener { snapshot ->
                val pointsByLabel = mutableMapOf<String, MutableList<LatLng>>()

                for (doc in snapshot) {
                    val lat = doc.getDouble("lat")
                    val lon = doc.getDouble("lon")
                    val label = doc.getString("label") ?: "unknown"

                    // ✅ Log the raw document data
                    android.util.Log.d(
                        "MapsActivity",
                        "Doc id=${doc.id}, lat=$lat, lon=$lon, label=$label, full=${doc.data}"
                    )

                    if (lat != null && lon != null && !lat.isNaN() && !lon.isNaN()) {
                        val point = LatLng(lat, lon)

                        val list = pointsByLabel.getOrPut(label) { mutableListOf() }
                        list.add(point)

                        mMap.addMarker(MarkerOptions().position(point).title(label))
                    }
                }

                for ((label, points) in pointsByLabel) {
                    if (points.size >= 2) {
                        for ((label, points) in pointsByLabel) {
                            for (point in points) {
                                mMap.addMarker(
                                    MarkerOptions()
                                        .position(point)
                                        .title(label)
                                        .icon(
                                            BitmapDescriptorFactory.defaultMarker(
                                                getHueForLabel(
                                                    label
                                                )
                                            )
                                        )
                                )
                            }
                        }

                    }
                }
            }
            .addOnFailureListener { e ->
                android.util.Log.e("MapsActivity", "Error loading Firestore: ${e.message}")
            }
    }


    private fun getHueForLabel(label: String): Float {
        return when (label) {
            "asphalt_good" -> BitmapDescriptorFactory.HUE_GREEN
            "asphalt_regular" -> BitmapDescriptorFactory.HUE_YELLOW
            "asphalt_bad" -> BitmapDescriptorFactory.HUE_YELLOW
            "paved_regular" -> BitmapDescriptorFactory.HUE_ORANGE
            "paved_bad" -> BitmapDescriptorFactory.HUE_ORANGE
            "unpaved_regular" -> BitmapDescriptorFactory.HUE_BLUE
            "unpaved_bad" -> BitmapDescriptorFactory.HUE_VIOLET
            else -> BitmapDescriptorFactory.HUE_ROSE
        }
    }
}
