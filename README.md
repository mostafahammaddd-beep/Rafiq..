# رَفِيق — Rafiq v14 (Quran recitation corrector)

Recite Ayat al-Kursi → every word turns **green** (correct), **yellow** (tashkeel / vowel),
**orange** (tajweed) or **red** (letters / skipped word). Tap a coloured word to read a short Arabic
note that explains the mistake and how to say it correctly.

## 1. Deploy the server (Windows CMD)
```
cd Rafiq\backend
py -m pip install modal numpy
py -m modal deploy modal_app.py
```
The URL stays the same as before (same app name), so the phone app needs no setup.
The first deploy after this update rebuilds the image (a few minutes). Cold start of the server
takes about 30–60 s; the app wakes the server the moment you open it and shows "الخادم جاهز" when ready.

## 2. Build the phone app
Push the whole `Rafiq` folder to GitHub → *Actions* → **Build APK** → download the `Rafiq-APK`
artifact (or the release) and install it. It is signed with the same debug key, so it updates your
current app in place.

## 3. Using it
Open the app, wait for **● الخادم جاهز**, press 🎤, recite from the start of the verse, press ⏹ when done
(it also finishes by itself when you complete the verse). 🔊 plays a reference recitation.
⚙ → *Accuracy*: **متساهل** (forgiving) · **متوازن** (default) · **صارم** (strict).

## 4. If a word is marked wrongly
⚙ → **نسخ تقرير فني** copies a technical report of the last session. Paste it to your developer/assistant:
it shows exactly what the model heard and why each word was flagged, so the rules can be tuned.
You can also open `<your-url>/health` and `<your-url>/debug/ref` in a browser.

## 5. Switches at the top of `backend/modal_app.py`
| Setting | Meaning |
|---|---|
| `PRECISION = "auto"` | Measures the CPU at start-up. Uses faster int8 maths only if full precision is too slow. `"fp32"` forces full precision. |
| `USE_GPU = False` | Set `True` only after Modal accepts a card (T4 GPU ≈ 10× faster, same code). |

## 6. Honest limits
* No model can be "100 % reliable". Wrong-sounding detection of subtle tajweed (madd length, tafkheem,
  qalqalah) depends on the muaalem model's confidence; the app only reports what the model is sure about,
  and "متساهل/متوازن/صارم" controls how sure it must be.
* Works for Ayat al-Kursi (the engine itself accepts any ayah: `?sura_idx=&aya_idx=`).
* `backend/dev/run_all_tests.py` is an automatic self-test (needs only numpy); you never need to run it.
