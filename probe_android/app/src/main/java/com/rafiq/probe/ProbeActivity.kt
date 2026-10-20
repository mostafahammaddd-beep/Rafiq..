package com.rafiq.probe

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.app.Activity
import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.security.MessageDigest
import java.util.Random

/**
 * أداة قياس (ليست التطبيق الرئيسي): هل يمكن تشغيل نموذج المعلّم int8 على هذا الهاتف؟
 * 1) تنزيل النموذج المنشور علناً (MIT) والتحقق من بصمته  2) تحميله بـ ONNX Runtime
 * 3) طباعة مدخلاته ومخرجاته الحقيقية  4) قياس السرعة (RTF = زمن المعالجة / مدة الصوت).
 */
class ProbeActivity : Activity() {

    private val main = Handler(Looper.getMainLooper())
    private val log = StringBuilder()
    private lateinit var logView: TextView
    private lateinit var statusView: TextView
    @Volatile private var busy = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this)
        root.orientation = LinearLayout.VERTICAL
        root.setPadding(24, 24, 24, 24)

        val title = TextView(this)
        title.text = "Rafiq Probe — قياس تشغيل النموذج على هذا الهاتف"
        title.textSize = 18f
        title.setTypeface(null, Typeface.BOLD)
        root.addView(title)

        val hint = TextView(this)
        hint.text = "1) اتصل بالواي فاي (التنزيل 876MB)  2) نزّل النموذج  3) شغّل القياس  4) انسخ النتيجة وأرسلها"
        hint.textSize = 13f
        root.addView(hint)

        val b1 = Button(this)
        b1.text = "1) تنزيل النموذج (876MB)"
        b1.setOnClickListener { download() }
        root.addView(b1)

        val b2 = Button(this)
        b2.text = "2) تشغيل القياس"
        b2.setOnClickListener { runTest() }
        root.addView(b2)

        val b3 = Button(this)
        b3.text = "3) نسخ النتيجة"
        b3.setOnClickListener { copyLog() }
        root.addView(b3)

        statusView = TextView(this)
        statusView.textSize = 14f
        root.addView(statusView)

        logView = TextView(this)
        logView.textSize = 12f
        logView.typeface = Typeface.MONOSPACE
        logView.setTextIsSelectable(true)
        root.addView(logView)

        val scroll = ScrollView(this)
        scroll.addView(root)
        setContentView(scroll)
    }

    // ------------------------------------------------------------------ مساعدات الواجهة

    private fun say(line: String) {
        main.post {
            log.append(line).append('\n')
            logView.text = log.toString()
        }
    }

    private fun setStatus(text: String) {
        main.post { statusView.text = text }
    }

    private fun copyLog() {
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("rafiq-probe", log.toString()))
        setStatus("تم النسخ ✓ الصقه في المحادثة")
    }

    // ------------------------------------------------------------------ تنزيل النموذج

    private fun download() {
        if (busy) return
        busy = true
        Thread {
            try {
                val dest = File(filesDir, MODEL_FILE)
                if (dest.exists() && dest.length() == MODEL_BYTES) {
                    say("النموذج موجود مسبقاً (${dest.length() / 1048576} MB).")
                } else {
                    val part = File(filesDir, "$MODEL_FILE.part")
                    val conn = URL(MODEL_URL).openConnection() as HttpURLConnection
                    conn.connectTimeout = 30000
                    conn.readTimeout = 60000
                    conn.instanceFollowRedirects = true
                    conn.setRequestProperty("User-Agent", "RafiqProbe/1.0")
                    conn.connect()
                    if (conn.responseCode != 200) throw RuntimeException("HTTP " + conn.responseCode)
                    val total = conn.contentLengthLong
                    val md = MessageDigest.getInstance("SHA-256")
                    conn.inputStream.use { input ->
                        FileOutputStream(part).use { out ->
                            val buf = ByteArray(256 * 1024)
                            var done = 0L
                            var lastPct = -1
                            while (true) {
                                val n = input.read(buf)
                                if (n < 0) break
                                out.write(buf, 0, n)
                                md.update(buf, 0, n)
                                done += n
                                val pct = if (total > 0) (done * 100 / total).toInt() else 0
                                if (pct != lastPct) {
                                    lastPct = pct
                                    setStatus("تنزيل: $pct%  (${done / 1048576} MB)")
                                }
                            }
                        }
                    }
                    val sha = md.digest().joinToString("") { b -> "%02x".format(b) }
                    if (sha == MODEL_SHA) {
                        say("✓ التنزيل اكتمل والبصمة SHA-256 مطابقة.")
                    } else {
                        say("⚠ البصمة لا تطابق المتوقع: $sha")
                    }
                    part.renameTo(dest)
                    setStatus("اكتمل التنزيل.")
                }
            } catch (t: Throwable) {
                say("خطأ في التنزيل: " + t.javaClass.simpleName + " " + (t.message ?: ""))
            } finally {
                busy = false
            }
        }.start()
    }

    // ------------------------------------------------------------------ القياس

    private fun runTest() {
        if (busy) return
        busy = true
        Thread {
            try {
                val f = File(filesDir, MODEL_FILE)
                if (!f.exists() || f.length() != MODEL_BYTES) {
                    say("نزّل النموذج أولاً (الخطوة 1).")
                } else {
                    benchmark(f)
                }
            } catch (t: Throwable) {
                say("خطأ أثناء القياس: " + t.javaClass.simpleName + " " + (t.message ?: ""))
            } finally {
                busy = false
            }
        }.start()
    }

    private fun benchmark(modelFile: File) {
        val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        val cores = Runtime.getRuntime().availableProcessors()
        say("الجهاز: ${Build.MANUFACTURER} ${Build.MODEL} | Android API ${Build.VERSION.SDK_INT} | أنوية: $cores")
        say("الذاكرة: ${mi.totalMem / 1048576} MB (متاح ${mi.availMem / 1048576} MB)")
        setStatus("تحميل النموذج في الذاكرة… (قد يستغرق دقيقة)")

        val env = OrtEnvironment.getEnvironment()
        val threads = Math.max(2, Math.min(4, cores))
        val so = OrtSession.SessionOptions()
        so.setIntraOpNumThreads(threads)
        val t0 = System.nanoTime()
        val session = env.createSession(modelFile.absolutePath, so)
        say("تحميل الجلسة: ${fmt((System.nanoTime() - t0) / 1e9)} ث | خيوط المعالجة: $threads")

        say("— المدخلات —")
        for ((name, info) in session.inputInfo) {
            val ti = info.info as? TensorInfo
            say("  $name: " + (ti?.type?.toString() ?: "?") + " " + (ti?.shape?.joinToString(",", "[", "]") ?: ""))
        }
        say("— المخرجات —")
        for ((name, info) in session.outputInfo) {
            val ti = info.info as? TensorInfo
            say("  $name: " + (ti?.type?.toString() ?: "?") + " " + (ti?.shape?.joinToString(",", "[", "]") ?: ""))
        }

        for (frames in intArrayOf(150, 300)) {
            setStatus("قياس السرعة ($frames إطاراً)…")
            val built = makeInputs(env, session, frames)
            val inputs = built.first
            val seconds = built.second
            val times = ArrayList<Double>()
            for (i in 0 until 3) {
                val a = System.nanoTime()
                val res = session.run(inputs)
                val dt = (System.nanoTime() - a) / 1e9
                if (i == 0 && frames == 150) {
                    for (e in res) {
                        val tv = e.value as? OnnxTensor
                        say("  ناتج ${e.key}: " + (tv?.info?.shape?.joinToString(",", "[", "]") ?: "?"))
                    }
                }
                res.close()
                times.add(dt)
            }
            for (t in inputs.values) t.close()
            val best = Math.min(times[1], times[2])
            say(
                "إطارات=$frames (~${fmt(seconds)} ث صوت): " +
                    "تشغيل1=${fmt(times[0])} تشغيل2=${fmt(times[1])} تشغيل3=${fmt(times[2])} ث  ⇒  RTF ≈ ${fmt(best / seconds)}"
            )
        }
        session.close()
        say("انتهى القياس. (RTF أقل من 0.5 = ممتاز للتلاوة الحية؛ حتى 1.5 يصلح لوضع «سجّل ثم حلّل»)")
        setStatus("اكتمل القياس ✓ — اضغط «نسخ النتيجة»")
    }

    /** يبني مدخلات عشوائية بالأشكال التي يعلنها النموذج نفسه. */
    private fun makeInputs(env: OrtEnvironment, session: OrtSession, frames: Int): Pair<Map<String, OnnxTensor>, Double> {
        val inputs = HashMap<String, OnnxTensor>()
        var seconds = frames / 50.0
        val rnd = Random(1)
        for ((name, info) in session.inputInfo) {
            val ti = info.info as TensorInfo
            val rank = ti.shape.size
            val isRawAudio = ti.type == OnnxJavaType.FLOAT && rank == 2
            if (isRawAudio) seconds = frames * 320 / 16000.0
            val shape = LongArray(rank) { i ->
                val d = ti.shape[i]
                if (d > 0) {
                    d
                } else if (i == 0) {
                    1L
                } else if (i == 1) {
                    if (isRawAudio) (frames * 320).toLong() else frames.toLong()
                } else {
                    1L
                }
            }
            var count = 1L
            for (d in shape) count *= d
            val n = count.toInt()
            val tensor: OnnxTensor = when (ti.type) {
                OnnxJavaType.FLOAT -> {
                    val arr = FloatArray(n) { (rnd.nextGaussian() * 0.5).toFloat() }
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), shape)
                }
                OnnxJavaType.INT64 -> {
                    val arr = LongArray(n) { 1L }
                    OnnxTensor.createTensor(env, LongBuffer.wrap(arr), shape)
                }
                OnnxJavaType.INT32 -> {
                    val arr = IntArray(n) { 1 }
                    OnnxTensor.createTensor(env, IntBuffer.wrap(arr), shape)
                }
                else -> throw RuntimeException("نوع مدخل غير مدعوم في الأداة: " + ti.type + " للمدخل " + name)
            }
            inputs[name] = tensor
        }
        return Pair(inputs, seconds)
    }

    private fun fmt(x: Double): String = String.format("%.2f", x)

    companion object {
        private const val MODEL_URL =
            "https://huggingface.co/ahmady36/shaikhy-models/resolve/main/muaalem_tajweed_int8_fastattn.onnx"
        private const val MODEL_FILE = "muaalem_int8.onnx"
        private const val MODEL_BYTES = 876441104L
        private const val MODEL_SHA = "2106429568d55dc05bf2cda7a97cf1dbcb13bf85a0d258d2cbf4d397e220226a"
    }
}
