# رَفِيق — Rafiq v14.3

Recite Ayat al-Kursi → every word turns **green** (correct), **yellow** (tashkeel), **orange** (tajweed)
or **red** (letters / skipped word). Tap a coloured word for a short Arabic note.

## 1. Update (3 steps)
1. **Server:** `cd backend` → `py -m modal deploy modal_app.py`  (same URL as before).
2. **App:** push this whole folder to GitHub; the *Build APK* workflow builds it (Actions → Rafiq-APK).
3. **GitHub cleanup (once):** open `.github/workflows` in your repo and delete any file other than `build.yml`
   (for example `main.yml`). Having two workflows builds the app twice.

## 2. What changed in v14.3
* **Fixed a real bug seen in your report:** a recitation recorded while the server was waking arrived as one
  big backlog and only 18 of 50 words were judged. Backlogs are now judged in order and completely, and the app
  no longer starts recording until the server says it is ready (so there is no backlog in the first place).
* **Tajweed false-alarm guards:** neighbouring letters with swapped ghunna/qalqalah labels (a model label
  slip, e.g. م loses ghunna while the next ل gains it) and ghunna on letters that cannot carry it are ignored.
* **Clearer vowel notes** when a reciter keeps a final vowel (e.g. تَأْخُذُهَا instead of تَأْخُذُهُ).
* **Silence gate in the app:** long silences are dropped, but 500 ms of natural silence after speech and 200 ms
  before the next word are always kept, so pauses (waqf), word starts, final vowels and madd lengths are untouched.
* **Faster cold starts (needs one deploy to take effect):** `USE_MEMORY_SNAPSHOT = True` in `modal_app.py`.
  Modal creates the snapshot after the function has been deployed and run a few times; the first starts after a
  deploy are still slow. If a deploy ever fails because of it, set it to `False` and deploy again.
* One awake server now serves two reciters before another is started (`target_inputs=2`).

## 3. Cost (Modal bills the whole time the server is awake)
Rates used: $0.0000131 per core-second and $0.00000222 per GB-second. One session (~70 s of recitation) costs
about **$0.005**; the floor for one live user is the recitation time itself, so a live session cannot get below
about $0.004 at 4 cores. Two users at the same time split the cost.

## 4. Free fallback: run the server on your own PC
`cd backend` → `py -m pip install -r requirements-local.txt` → `py run_local.py`, then in a second window
`cloudflared tunnel --url http://localhost:8000` and paste the printed https address in the app (⚙ → رابط الخادم).

## 5. Telling me what is wrong
Tap a **green** word → "⚠ فيها خطأ لم يُكتشف"; tap a **coloured** word → "✔ نطقتها صحيحة".
Then ⚙ → "نسخ تقرير فني" and paste it in the chat. `backend/dev/run_all_tests.py` is an automatic self-test.
