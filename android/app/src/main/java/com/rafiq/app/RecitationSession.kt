package com.rafiq.app

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.sqrt

/** نتيجة كلمة واحدة: status = ok | tashkeel | tajweed | letters | unchecked */
data class WordIssue(val type: String, val kind: String, val note: String, val focus: String)

data class WordResult(val index: Int, val status: String, val issues: List<WordIssue>)

/** عميل شبكة مشترك (يُعاد استخدامه لتفادي إنشاء خيوط جديدة في كل جلسة). */
object Net {
    val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(90, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()
}

/**
 * جلسة تلاوة واحدة:
 *  - تلتقط الصوت (16kHz / Mono / PCM16) على خيط مستقل وتقسمه إلى قطع مدة كل منها 100ms.
 *  - تبث القطع للخادم عبر WebSocket، وتحتفظ بكل القطع في الذاكرة؛ فإن انقطع الاتصال
 *    أعادت الاتصال تلقائياً وأعادت إرسال الصوت من البداية فلا تضيع التلاوة.
 *  - كل الاستدعاءات للـ Listener تتم على الخيط الرئيسي.
 */
class RecitationSession(
    private val appContext: Context,
    private val strictness: String,
    private val batch: Boolean,
    private val listener: Listener
) {
    interface Listener {
        /** 0 = اتصال، 1 = متصل، 2 = إعادة اتصال */
        fun onConnection(state: Int)
        fun onReady(words: List<String>)
        fun onProgress(tracked: Int, committed: Int, audioMs: Long, updates: List<WordResult>)
        fun onFinal(words: List<WordResult>, complete: Boolean, debug: String)
        fun onLevel(rms: Float, voiced: Boolean, audioMs: Long)
        fun onAutoStop()
        fun onError(message: String)
        fun onFatal(message: String)
    }

    private val main = Handler(Looper.getMainLooper())
    private val lock = Any()
    private val chunks = ArrayList<ByteArray>()
    private val voicedCum = ArrayList<Long>()
    private var sentIdx = 0
    private var stopSent = false

    @Volatile private var socket: WebSocket? = null
    @Volatile private var open = false
    @Volatile private var capturing = false
    @Volatile private var stopRequested = false
    @Volatile private var captureDone = false
    @Volatile private var finished = false
    @Volatile private var captureStarted = false
    @Volatile private var attempts = 0
    private var record: AudioRecord? = null

    // ------------------------------------------------------------------ بدء / إيقاف

    @SuppressLint("MissingPermission")
    fun start(): Boolean {
        val sr = RecitationConfig.SAMPLE_RATE
        val minBuf = AudioRecord.getMinBufferSize(
            sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) return false
        val rec: AudioRecord = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sr,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                max(minBuf, sr * 2)
            )
        } catch (e: Exception) {
            return false
        }
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            return false
        }
        record = rec
        capturing = true
        if (batch) {
            beginCapture()            // التسجيل محلي بالكامل؛ يُرسل مرة واحدة عند الإيقاف
        } else {
            connect()                 // الالتقاط يبدأ عند وصول «ready» من الخادم (انظر beginCapture)
        }
        return true
    }

    /** يبدأ التقاط الصوت فقط حين يصبح الخادم جاهزاً، فلا يتراكم تسجيل طويل أثناء إيقاظه. */
    private fun beginCapture() {
        if (captureStarted || finished) return
        val rec = record ?: return
        captureStarted = true
        val t = Thread { captureLoop(rec) }
        t.name = "rafiq-capture"
        t.start()
    }

    private fun releaseUnstartedRecorder() {
        if (!captureStarted) {
            try {
                record?.release()
            } catch (_: Exception) {
            }
        }
    }

    /** المستخدم ضغط «إيقاف»: نكمل إرسال ما التُقط ثم نطلب التقرير النهائي. */
    fun stop() {
        stopRequested = true
        pump()
    }

    /** إنهاء فوري دون انتظار التقرير. */
    fun abort() {
        finished = true
        capturing = false
        stopRequested = true
        try {
            socket?.cancel()
        } catch (_: Exception) {
        }
        socket = null
        open = false
        releaseUnstartedRecorder()
    }

    /** عدد ملّيثواني الصوت المنطوق (بعد حد الصمت) منذ لحظة audioMs من التسجيل. */
    fun voicedSinceMs(audioMs: Long): Long {
        return synchronized(lock) {
            val total = if (voicedCum.isEmpty()) 0L else voicedCum[voicedCum.size - 1]
            val idx = (audioMs / 100L).toInt() - 1
            val atMark = when {
                idx < 0 -> 0L
                idx < voicedCum.size -> voicedCum[idx]
                else -> total
            }
            total - atMark
        }
    }

    // ------------------------------------------------------------------ التقاط الصوت

    private fun captureLoop(rec: AudioRecord) {
        val chunk = RecitationConfig.SAMPLE_RATE / 10   // 100ms
        val shorts = ShortArray(chunk)
        var noise = 0.004f
        var voicedMs = 0L
        var elapsedMs = 0L
        var silentRun = 0
        val preRoll = ArrayDeque<ByteArray>()
        try {
            rec.startRecording()
            while (capturing && !stopRequested && !finished) {
                var filled = 0
                while (filled < chunk && capturing && !stopRequested) {
                    val r = rec.read(shorts, filled, chunk - filled)
                    if (r < 0) {
                        fatal("تعذّر قراءة الميكروفون. تأكد أن تطبيقاً آخر لا يستخدمه.")
                        return
                    }
                    filled += r
                }
                if (filled < chunk) break

                var sum = 0.0
                val bytes = ByteArray(chunk * 2)
                for (i in 0 until chunk) {
                    val s = shorts[i].toInt()
                    sum += (s * s).toDouble()
                    bytes[2 * i] = (s and 0xFF).toByte()
                    bytes[2 * i + 1] = ((s shr 8) and 0xFF).toByte()
                }
                val rms = (sqrt(sum / chunk) / 32768.0).toFloat()
                if (rms < noise * 1.6f + 0.002f) noise = 0.95f * noise + 0.05f * rms
                val thr = (noise * 3f).coerceIn(0.006f, 0.02f)
                val voiced = rms >= thr
                elapsedMs += 100L
                val cumBefore = voicedMs
                if (voiced) voicedMs += 100L

                // بوابة الصمت: نحتفظ بالكلام + حتى 500ms من الصمت الطبيعي بعده، ونتجاوز الصمت الأطول،
                // مع إعادة آخر 200ms قبل بدء الكلام التالي (حتى لا تُقطع بدايات الكلمات ولا مدّات الحروف).
                val toSend = ArrayList<ByteArray>()
                val cums = ArrayList<Long>()
                if (voiced) {
                    for (p in preRoll) {
                        toSend.add(p)
                        cums.add(cumBefore)
                    }
                    preRoll.clear()
                    toSend.add(bytes)
                    cums.add(voicedMs)
                    silentRun = 0
                } else {
                    silentRun += 1
                    if (silentRun <= HANGOVER_CHUNKS) {
                        toSend.add(bytes)
                        cums.add(cumBefore)
                    } else {
                        preRoll.addLast(bytes)
                        if (preRoll.size > PREROLL_CHUNKS) preRoll.removeFirst()
                    }
                }
                val audioMs = synchronized(lock) {
                    for (i in toSend.indices) {
                        chunks.add(toSend[i])
                        voicedCum.add(cums[i])
                    }
                    chunks.size * 100L
                }
                if (toSend.isNotEmpty()) pump()
                val shownMs = elapsedMs
                main.post { if (!finished) listener.onLevel(rms, voiced, shownMs) }
                if (audioMs >= MAX_SESSION_MS && !stopRequested) {
                    stopRequested = true
                    main.post { if (!finished) listener.onAutoStop() }
                }
            }
        } catch (e: Exception) {
            fatal("تعذّر تشغيل الميكروفون.")
            return
        } finally {
            try {
                rec.stop()
            } catch (_: Exception) {
            }
            try {
                rec.release()
            } catch (_: Exception) {
            }
            captureDone = true
            if (batch) uploadBatch() else pump()
        }
    }

    // ------------------------------------------------------------------ وضع «سجّل ثم حلّل»

    private fun buildWav(pcm: ByteArray, sampleRate: Int): ByteArray {
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(36 + pcm.size)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16)
        header.putShort(1)                 // PCM
        header.putShort(1)                 // mono
        header.putInt(sampleRate)
        header.putInt(sampleRate * 2)
        header.putShort(2)
        header.putShort(16)
        header.put("data".toByteArray())
        header.putInt(pcm.size)
        return header.array() + pcm
    }

    /** يرسل التسجيل كاملاً (بعد بوابة الصمت) إلى /analyze ويقرأ النتيجة النهائية نفسها التي يرسلها البث. */
    private fun uploadBatch() {
        if (finished) return
        val pcm = synchronized(lock) {
            val bos = ByteArrayOutputStream()
            for (c in chunks) bos.write(c)
            bos.toByteArray()
        }
        if (pcm.size < RecitationConfig.SAMPLE_RATE * 2) {      // أقل من ثانية من الصوت
            finished = true
            main.post { listener.onFinal(ArrayList<WordResult>(), false, "") }
            return
        }
        val wav = buildWav(pcm, RecitationConfig.SAMPLE_RATE)
        val url = RecitationConfig.analyzeUrl(appContext)
        val client = Net.client.newBuilder()
            .readTimeout(180, TimeUnit.SECONDS)
            .writeTimeout(120, TimeUnit.SECONDS)
            .build()
        var attempt = 0
        while (attempt < 3 && !finished) {
            attempt += 1
            try {
                val body = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("sura_idx", RecitationConfig.SURA_IDX.toString())
                    .addFormDataPart("aya_idx", RecitationConfig.AYA_IDX.toString())
                    .addFormDataPart("strictness", strictness)
                    .addFormDataPart("file", "recitation.wav", wav.toRequestBody("audio/wav".toMediaType()))
                    .build()
                val req = Request.Builder().url(url).post(body).build()
                client.newCall(req).execute().use { r ->
                    val text = r.body?.string() ?: ""
                    if (r.isSuccessful) {
                        val o = JSONObject(text)
                        val words = parseWords(o.optJSONArray("words"))
                        val complete = o.optJSONObject("summary")?.optBoolean("complete", false) ?: false
                        val debugText = o.optJSONObject("debug")?.toString() ?: ""
                        finished = true
                        main.post { listener.onFinal(words, complete, debugText) }
                        return
                    }
                }
            } catch (e: Exception) {
                // نعيد المحاولة أدناه
            }
            if (!finished && attempt < 3) {
                try {
                    Thread.sleep(3000L * attempt)
                } catch (_: InterruptedException) {
                }
            }
        }
        if (!finished) fatal("تعذّر إرسال التسجيل للتحليل. تأكد من الإنترنت ثم أعد المحاولة.")
    }

    // ------------------------------------------------------------------ الشبكة

    private fun connect() {
        if (finished) return
        val url = RecitationConfig.webSocketUrl(appContext, strictness)
        val req = Request.Builder().url(url).build()
        val state = if (attempts == 0) 0 else 2
        main.post { if (!finished) listener.onConnection(state) }
        socket = Net.client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                synchronized(lock) {
                    open = true
                    sentIdx = 0          // أعد إرسال كل الصوت للخادم الجديد
                    stopSent = false
                }
                main.post { if (!finished) listener.onConnection(1) }
                pump()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleMessage(text)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                onSocketGone(webSocket)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                onSocketGone(webSocket)
            }
        })
    }

    private fun onSocketGone(ws: WebSocket) {
        if (ws !== socket) return
        open = false
        if (finished) return
        attempts++
        if (attempts > 8) {
            fatal("تعذّر الاتصال بالخادم. تأكد من الإنترنت ثم أعد المحاولة.")
            return
        }
        main.post { if (!finished) listener.onConnection(2) }
        main.postDelayed({ if (!finished) connect() }, minOf(5000L, 700L * attempts))
    }

    /** يرسل كل قطع الصوت المتبقية بالترتيب، ثم رسالة الإيقاف إن طُلبت. */
    private fun pump() {
        synchronized(lock) {
            val ws = socket
            if (ws == null || !open) return
            while (sentIdx < chunks.size) {
                if (!ws.send(chunks[sentIdx].toByteString())) return
                sentIdx++
            }
            if (stopRequested && captureDone && !stopSent) {
                if (ws.send("{\"type\":\"stop\"}")) stopSent = true
            }
        }
    }

    private fun handleMessage(text: String) {
        val o = try {
            JSONObject(text)
        } catch (e: Exception) {
            return
        }
        when (o.optString("type")) {
            "ready" -> {
                attempts = 0
                beginCapture()
                val arr = o.optJSONArray("words")
                val ws = ArrayList<String>()
                if (arr != null) for (i in 0 until arr.length()) ws.add(arr.optString(i, ""))
                main.post { if (!finished) listener.onReady(ws) }
            }
            "progress" -> {
                val tracked = o.optInt("tracked", -1)
                val committed = o.optInt("committed", 0)
                val audioMs = o.optLong("audio_ms", 0L)
                val updates = parseWords(o.optJSONArray("updates"))
                main.post { if (!finished) listener.onProgress(tracked, committed, audioMs, updates) }
            }
            "final" -> {
                finished = true
                capturing = false
                val words = parseWords(o.optJSONArray("words"))
                val summary = o.optJSONObject("summary")
                val complete = summary?.optBoolean("complete", false) ?: false
                val debugText = o.optJSONObject("debug")?.toString() ?: ""
                main.post { listener.onFinal(words, complete, debugText) }
                try {
                    socket?.close(1000, "done")
                } catch (_: Exception) {
                }
            }
            "error" -> {
                val msg = o.optString("message", "")
                main.post { if (!finished) listener.onError(msg) }
            }
        }
    }

    private fun parseWords(arr: JSONArray?): List<WordResult> {
        val out = ArrayList<WordResult>()
        if (arr == null) return out
        for (i in 0 until arr.length()) {
            val w = arr.optJSONObject(i) ?: continue
            val issues = ArrayList<WordIssue>()
            val ia = w.optJSONArray("issues")
            if (ia != null) {
                for (k in 0 until ia.length()) {
                    val io = ia.optJSONObject(k) ?: continue
                    issues.add(
                        WordIssue(
                            io.optString("t", "letters"),
                            io.optString("k", ""),
                            io.optString("n", ""),
                            io.optString("f", "")
                        )
                    )
                }
            }
            out.add(WordResult(w.optInt("i", -1), w.optString("s", "ok"), issues))
        }
        return out
    }

    private fun fatal(message: String) {
        finished = true
        capturing = false
        releaseUnstartedRecorder()
        main.post { listener.onFatal(message) }
    }

    companion object {
        private const val MAX_SESSION_MS = 300_000L   // 5 دقائق كحد أقصى
        private const val HANGOVER_CHUNKS = 5          // 500ms صمت طبيعي بعد الكلام
        private const val PREROLL_CHUNKS = 2           // 200ms قبل بدء الكلام التالي
    }
}
