
package com.example.routex

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.CancellationTokenSource
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

class RoadLocation(private val context: Context) {

    private val client = LocationServices.getFusedLocationProviderClient(context)

    /**
     * Returns a fresh location fix if possible, otherwise falls back to last known.
     * Make sure location permissions are granted before calling.
     */
    @SuppressLint("MissingPermission")
    suspend fun getCurrentLocation(): Location? {
        // 1) Try to get a fresh fix
        val fresh = getFreshLocation()
        if (fresh != null) return fresh

        // 2) Fall back to last known (may be null on first run)
        return suspendCancellableCoroutine { cont ->
            client.lastLocation
                .addOnSuccessListener { cont.resume(it) }
                .addOnFailureListener { cont.resume(null) }
        }
    }

    @SuppressLint("MissingPermission")
    private suspend fun getFreshLocation(): Location? = suspendCancellableCoroutine { cont ->
        val cts = CancellationTokenSource()
        client.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, cts.token)
            .addOnSuccessListener { cont.resume(it) }
            .addOnFailureListener { cont.resume(null) }

        cont.invokeOnCancellation { cts.cancel() }
    }
}
