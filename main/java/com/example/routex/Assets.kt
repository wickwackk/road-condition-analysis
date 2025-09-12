package com.example.routex

import android.content.Context
import org.json.JSONArray
import java.io.File
import java.io.IOException

/**
 * Copy an asset into the app's internal files dir and return its absolute path.
 * Set [overwrite] = true (default) during development to avoid stale cached files.
 */
fun assetFilePath(context: Context, assetName: String, overwrite: Boolean = true): String {
    val outFile = File(context.filesDir, assetName)
    if (overwrite || !outFile.exists() || outFile.length() == 0L) {
        try {
            context.assets.open(assetName).use { input ->
                context.openFileOutput(assetName, Context.MODE_PRIVATE).use { output ->
                    input.copyTo(output)
                }
            }
        } catch (e: Exception) {
            throw IOException("Failed to copy asset '$assetName' → ${outFile.absolutePath}: ${e.message}", e)
        }
    }
    return outFile.absolutePath
}

/** Delete a previously cached copy of an asset from filesDir (useful if you swapped models). */
fun deleteCachedAsset(context: Context, assetName: String): Boolean {
    return try {
        File(context.filesDir, assetName).delete()
    } catch (_: Exception) {
        false
    }
}

/**
 * Load class labels from assets, supporting either:
 *  - JSON array file (e.g., class_names.json: ["asphalt_bad", ...])
 *  - Plain text file (e.g., class_names.txt: one label per line)
 */
fun loadLabelsFlexible(context: Context, assetName: String): List<String> {
    val text = context.assets.open(assetName).bufferedReader().use { it.readText() }.trim()
    return if (assetName.endsWith(".json", ignoreCase = true) || text.startsWith("[")) {
        val arr = JSONArray(text)
        List(arr.length()) { i -> arr.getString(i) }
    } else {
        text.lines().map { it.trim() }.filter { it.isNotEmpty() }
    }
}

/** Legacy helper (kept for back-compat with older code using a .txt file). */
fun loadLabels(context: Context, assetName: String): List<String> =
    context.assets.open(assetName).bufferedReader().readLines()
