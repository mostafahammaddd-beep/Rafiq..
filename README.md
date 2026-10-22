# رَفِيق — Rafiq v14.4

Recite Ayat al-Kursi → every word turns **green** (correct), **yellow** (tashkeel), **orange** (tajweed)
or **red** (letters / skipped word). Tap a coloured word for a short Arabic note.

## Update
1. `cd backend` → `py -m modal deploy modal_app.py` (same URL).
2. Push this folder to GitHub (the *Build APK* workflow builds the app). Delete any other file in
   `.github/workflows` besides `build.yml` (e.g. `main.yml`).

## Two ways to analyse (⚙ → طريقة التحليل)
* **بعد التسجيل (default, cheapest):** the app records on the phone (long silences are dropped, 500 ms after
  speech and 200 ms before the next word are always kept), then uploads once. The server is awake only while it
  analyses (about 25 s), not while you recite.
* **مباشر:** words are coloured while you recite; the server stays awake for the whole recitation (about 2x the cost).

## Cost (4 cores + 4 GB; Modal rates $0.0000131/core-s, $0.00000222/GB-s; RTF 0.352 measured)
| one Ayat al-Kursi session | cost |
|---|---|
| v14.1 live (8 cores, 5 min idle) | ≈ $0.050 |
| v14.2 live | ≈ $0.0074 |
| v14.4 live, snapshot ready | ≈ $0.0051 |
| **v14.4 record-then-analyze, snapshot ready** | **≈ $0.0023** |
| same, before the first snapshot exists (≈35 s cold start) | ≈ $0.0042 |

## Hafs 'an 'Asim policy used by the judge (see `madd_bounds` in `backend/modal_app.py`)
madd munfasil / silah kubra: 2 (qasr) to 5 all valid · madd muttasil 4–6 · lazim 6 · 'arid and lin at a stop 2/4/6 ·
tabii, silah sughra, badal, 'iwad exactly 2. A final vowel dropped at a stop, ta marbuta → ha, tanween → alif at a
stop, a word started fresh after a stop (hamzat al-wasl pronounced, no idgham with the previous word) are all valid
*when the audio really contains a stop*. The expected pronunciations themselves come from `quran-transcript`.

## Telling me what is wrong
Tap a green word → "⚠ فيها خطأ لم يُكتشف"; tap a coloured word → "✔ نطقتها صحيحة"; then ⚙ → "نسخ تقرير فني"
and paste it. (The report is now compact: one line per word.)  Free fallback: `backend/run_local.py` (see the file).
`backend/dev/run_all_tests.py` and `mutation_test.py` are automatic self-tests.
