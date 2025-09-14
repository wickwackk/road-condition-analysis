package com.example.routex

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import kotlin.math.exp

data class Pred(val index: Int, val label: String, val prob: Float)

class ResnetInference(ctx: Context) {
    private val app = ctx.applicationContext
    private val labels: List<String> = runCatching {
        loadLabelsFlexible(app, "class_names.json")
    }.getOrElse {
        Log.e("ResnetInference", "Failed to load labels: ${it.message}")
        listOf("unknown")
    }

    @Volatile private var module: Module? = null

    private fun loadModuleIfNeeded(): Module? {
        val m = module
        if (m != null) return m
        return runCatching {
            val path = assetFilePath(app, "resnet_scripted.pt") // or .ptl if you used Lite
            val f = File(path)
            Log.d("ResnetInference", "Loading model from: $path (exists=${f.exists()}, size=${f.length()})")
            Module.load(path)
        }.onSuccess { module = it }
            .onFailure { Log.e("ResnetInference", "Module load failed: ${it.message}") }
            .getOrNull()
    }

    /** Top-K predictions with softmax probabilities (default K=3). */
    // ResnetInference.kt
    fun classifyWithProbs(bmp: Bitmap, topK: Int = 3): List<Pred> {
        val m = loadModuleIfNeeded() ?: return listOf(Pred(0, "unknown", 1f))

        // 1) Resize to the model's expected size
        val resized = Bitmap.createScaledBitmap(bmp, 224, 224, true)

        // 2) Choose the normalization that matches training
        val mean   = floatArrayOf(0.485f, 0.456f, 0.406f)  // try also 0.5f,0.5f,0.5f if needed
        val std    = floatArrayOf(0.229f, 0.224f, 0.225f)

        val input  = TensorImageUtils.bitmapToFloat32Tensor(resized, mean, std)
        val out: Tensor = m.forward(IValue.from(input)).toTensor()
        val logits = out.dataAsFloatArray
        val probs  = softmax(logits)
        val idxs   = probs.indices.sortedByDescending { probs[it] }.take(topK)
        Log.d("ML", "logits=${logits.size} classes=${labels.size} top=${idxs.joinToString()}")
        return idxs.map { i -> Pred(i, labels.getOrElse(i) { "unknown" }, probs[i]) }
    }


    /** Back-compat single label. */
    fun classify(bmp: Bitmap): String = classifyWithProbs(bmp, 1).first().label

    private fun softmax(x: FloatArray): FloatArray {
        // stable softmax
        val max = x.maxOrNull() ?: 0f
        var sum = 0.0
        val exps = DoubleArray(x.size)
        for (i in x.indices) { exps[i] = exp((x[i] - max).toDouble()); sum += exps[i] }
        val out = FloatArray(x.size)
        val inv = 1.0 / sum
        for (i in x.indices) out[i] = (exps[i] * inv).toFloat()
        return out
    }
}
