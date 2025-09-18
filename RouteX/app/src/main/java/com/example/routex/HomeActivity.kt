package com.example.routex

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity

class HomeActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        fun tap(v: View, go: () -> Unit) {
            v.animate().scaleX(0.98f).scaleY(0.98f).setDuration(80).withEndAction {
                v.animate().scaleX(1f).scaleY(1f).setDuration(80).start()
                go()
            }.start()
        }

        findViewById<View>(R.id.cardCamera).setOnClickListener {
            tap(it) { startActivity(Intent(this, MainActivity::class.java)) }  // camera screen
        }
        findViewById<View>(R.id.cardMap).setOnClickListener {
            tap(it) { startActivity(Intent(this, MapActivity::class.java)) }   // map screen
        }
    }
}
