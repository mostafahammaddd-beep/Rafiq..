# رَفِيق — Rafiq v14.2

Recite Ayat al-Kursi → every word turns **green** (correct), **yellow** (tashkeel), **orange** (tajweed)
or **red** (letters / skipped word). Tap a coloured word for a short Arabic note.

## 1. Deploy / update the server (Windows CMD)
```
cd backend
py -m pip install modal numpy
py -m modal deploy modal_app.py
```
Same URL as before. Push the whole folder to GitHub to rebuild the APK (it updates the installed app).

## 2. Stop burning Modal credit (what changed)
Modal bills about **$0.000123 per second** for the old setup (8 cores + 8 GB). The old settings kept the
server alive 5 minutes after every use **and** woke it every time the app opened, i.e. about $0.037 of pure
idle per app opening. Now: 4 cores, 4 GB, sleeps after 10 s, and **the app no longer wakes the server by
itself**. The chip says "⏸ الخادم نائم": tap it (or press the mic) when you actually want to recite; the first
start takes up to ~30 s. One whole session costs about **$0.006** (roughly 6× less, and $0 for just opening the app).

## 3. Free plan B: run the same server on your own PC
```
cd backend
py -m pip install -r requirements-local.txt
py run_local.py                      (window 1)
cloudflared tunnel --url http://localhost:8000      (window 2; download cloudflared from GitHub, no account)
```
Paste the printed `https://….trycloudflare.com` address in the app: ⚙ → رابط الخادم. Free, supports one or a few
people at a time, the address changes each run, the PC must stay on. (For addresses containing `modal.run`
the app uses the sleeping/economy behaviour; for any other address it connects immediately.)

## 4. Telling me what is wrong (this is how accuracy gets fixed)
* Tap a **green** word → **⚠ فيها خطأ لم يُكتشف** if you know you made a mistake there.
* Tap a **coloured** word → **✔ نطقتها صحيحة** if you said it right.
* ⚙ → **نسخ تقرير فني** copies those notes plus what the model heard, word by word. Paste it in the chat.

## 5. On-device feasibility probe (the only route to $0 for any number of users)
Run workflow **Build Probe APK** on GitHub (Actions tab → Run workflow), install `Rafiq-Probe-APK`, follow the
3 buttons (Wi-Fi recommended, 876 MB download), and send me the copied result. It does not touch the main app.

## 6. Honest limits
No system can be 100 % accurate on real human audio, and free hosting cannot serve thousands of simultaneous
users with this model. Compute cost per reciter is roughly 1–2 CPU cores for as long as they speak; only
running the model on each user's own phone makes cost independent of the number of users.
`backend/dev/run_all_tests.py` and `mutation_test.py` are automatic self-tests (numpy only).
