package com.example.routex

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File


object ImageUtils {
    fun decodeDownsampled(f: File, reqW: Int, reqH: Int): Bitmap {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(f.absolutePath, opts)
        opts.inSampleSize = calculateInSampleSize(opts, reqW, reqH)
        opts.inJustDecodeBounds = false
        return BitmapFactory.decodeFile(f.absolutePath, opts)
    }


    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        val (height: Int, width: Int) = options.run { outHeight to outWidth }
        var inSampleSize = 1
        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }
        return inSampleSize
    }
}