"""
Self-checking simulation suite for the Rafiq engine (no model, no GPU, no internet needed).
Run:  py dev/run_all_tests.py      (needs only numpy)
A fake QPS phonemizer + a fake recitation model (which returns the phonemes inside whatever audio
window the engine hands it, including cut-off words at the edges) exercise the real engine code.
"""
import asyncio, json, os, sys, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
import modal_app as M

FAILS = []
def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (("   -> " + str(detail)) if (detail and not cond) else ""))
    if not cond: FAILS.append(name)

ref = sim.build_ref()
PAUSES = {6, 11, 18, 25, 31, 39, 43, 46}
BASE = [ref.waqf[k] if (k in PAUSES and ref.waqf[k]) else ref.wasl[k] for k in range(ref.n)]
WASL = list(ref.wasl)

def flagged(final): return {w["i"]: w for w in final["words"] if w["s"] != "ok"}
def kinds(w): return [i["k"] for i in w.get("issues", [])]

# ---------------------------------------------------------------- reference building
check("reference: 50 words, stop sign ignored, exact segmentation", ref.n == 50 and not ref.approx)
check("reference: waqf forms computed", sum(1 for w in ref.waqf if w) >= 45)

# ---------------------------------------------------------------- clean recitation
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, BASE, pauses=PAUSES, seed=3))
check("clean: all 50 judged correct", final["summary"]["ok"] == 50 and final["summary"]["complete"], final["summary"])
lags = sim.lag_stats(log, sp)
check("clean: tracking lag <= 3 words (before client smoothing)", max(lags) <= 3, max(lags))
check("clean: model never receives more than 6 s (old design: whole recitation)",
      max(s["wave_s"] for s in eng.debug["steps"]) <= 6.0)

# ---------------------------------------------------------------- injected mistakes
def inject(spoken):
    s = list(spoken); exp = {}
    s[3] = s[3].replace("ل", "ن", 1); exp[3] = ("letters", "replace")
    s[14] = s[14].replace(M.KASRA, M.FATHA); exp[14] = ("tashkeel", "vowel_swap")
    i = s[15].index("اا"); s[15] = s[15][:i] + s[15][i + 2:]; exp[15] = ("tajweed", "madd_dropped")
    s[21] = s[21].replace("لل", "ل", 1); exp[21] = ("tajweed", "shaddah_dropped")
    s[27] = ""; exp[27] = ("letters", "skipped")
    i = s[33].index("ح"); s[33] = s[33][:i] + s[33][i + 1:]; exp[33] = ("letters", "missing")
    return s, exp
for preset in ("balanced", "strict"):
    spoken, exp = inject(BASE)
    eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=PAUSES, seed=5), preset=preset)
    got = flagged(final)
    ok = all(w in got and got[w]["s"] == e[0] and e[1] in kinds(got[w]) for w, e in exp.items())
    check("errors/%s: all 6 injected mistakes found with the right colour and kind" % preset, ok,
          {w: (got[w]["s"], kinds(got[w])) if w in got else None for w in exp})
    check("errors/%s: zero false positives" % preset, set(got) == set(exp), set(got) - set(exp))
spoken, exp = inject(BASE)
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=PAUSES, seed=5), preset="relaxed")
check("errors/relaxed: tolerates the dropped madd that balanced reports", 15 not in flagged(final))

# ---------------------------------------------------------------- robustness (all must be clean)
cases = {
    "fast reciter": dict(scale=0.6, pauses=PAUSES, seed=11),
    "slow reciter": dict(scale=1.7, pauses=PAUSES, seed=12),
    "no pauses (all wasl)": dict(pauses=set(), seed=13, spoken=WASL),
    "waqf form on every word": dict(pauses=set(range(50)), seed=14, spoken=[ref.waqf[k] or ref.wasl[k] for k in range(50)]),
    "opening du'a before the verse": dict(pauses=PAUSES, seed=15, junk=sim.fake_ph("أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ")),
    "model noise (3 %, low confidence)": dict(pauses=PAUSES, seed=17, noise_p=0.03),
    "3-second pause mid-verse": dict(pauses=PAUSES | {29}, seed=19),
}
for name, kw in cases.items():
    spoken = kw.pop("spoken", BASE)
    eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, spoken, **kw))
    check("robust: " + name, final["summary"]["complete"] and not flagged(final), (final["summary"], list(flagged(final))[:5]))
st = list(BASE); st[20] = st[19] + st[20]
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, st, pauses=PAUSES, seed=18))
check("robust: stutter (previous word repeated)", final["summary"]["complete"] and not flagged(final))
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, BASE, pauses=PAUSES, seed=16, stop_after=24, trail=0.8))
check("stop halfway: ~25 words judged, none flagged, not marked complete",
      20 <= final["summary"]["judged"] <= 26 and not flagged(final) and not final["summary"]["complete"], final["summary"])
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, BASE, pauses=PAUSES, seed=20), rtf=0.6)
check("slow server (rtf 0.6): still complete and clean", final["summary"]["complete"] and not flagged(final))
eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, BASE, pauses=PAUSES, seed=20), rtf=1.3)
check("overloaded server (rtf 1.3): completes, window capped at 10 s",
      final["summary"]["complete"] and max(s["wave_s"] for s in eng.debug["steps"]) <= 10.01)

# ---------------------------------------------------------------- burst replay (reconnect)
spoken, _ = inject(BASE)
sp = sim.Speech(ref, spoken, pauses=PAUSES, seed=5); holder = {}
eng = M.SessionEngine(ref, sp.recognizer(holder), M.get_preset("balanced")); holder["engine"] = eng
eng.add_audio(sp.audio)
while eng.ready_to_run() and not eng.complete: eng.step()
final = eng.flush()
check("burst replay (whole recording at once): same 6 mistakes, nothing else", set(flagged(final)) == {3, 14, 15, 21, 27, 33}, list(flagged(final)))

# ---------------------------------------------------------------- silence
eng = M.SessionEngine(ref, lambda w: M.Recognition("", None, None), M.get_preset("balanced"))
rng = np.random.default_rng(0)
for _ in range(100):
    eng.add_audio((rng.standard_normal(1600) * 0.0008).astype(np.float32))
    if eng.ready_to_run(): eng.step()
check("silence only: nothing is judged", eng.seq == 0 and eng.tracked == -1)

# ---------------------------------------------------------------- sifat (qalqalah / tafkheem)
def sifat_run(preset, flip):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); holder = {}; plain = sp.recognizer(holder)
    def rec(wave):
        r = plain(wave); items = []
        for g in sim.fake_groups(r.text):
            letter = next((c for c in g.phonemes if not M.is_mark(c)), ""); d = {"phonemes_group": g.phonemes}
            for lv in ("qalqla", "tafkheem_or_taqeeq", "ghonna"):
                d[lv] = types.SimpleNamespace(text=flip(lv, letter, getattr(g, lv)), prob=0.97)
            items.append(d)
        return M.Recognition(r.text, r.probs, M.convert_sifat(items))
    eng = M.SessionEngine(ref, rec, M.get_preset(preset)); holder["engine"] = eng
    t = 0.0
    while t < sp.total and not eng.complete:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.1) * M.SR)]); t += 0.1
        if eng.ready_to_run(): eng.step()
    return flagged(eng.flush())
g = sifat_run("balanced", lambda lv, l, lab: lab)
check("sifat: faithful sifat -> no flags", not g)
g = sifat_run("balanced", lambda lv, l, lab: "not_moqalqal" if (lv == "qalqla" and l == "ط") else lab)
check("sifat: missing qalqalah on ط is reported (orange)", list(g) == [33] and g[33]["s"] == "tajweed" and kinds(g[33]) == ["qalqla"], g)
g = sifat_run("balanced", lambda lv, l, lab: "moraqaq" if (lv == "tafkheem_or_taqeeq" and l == "ر") else lab)
check("sifat: light ر reported as tafkheem error", set(g) == {18, 41, 43}, list(g))
g = sifat_run("relaxed", lambda lv, l, lab: "moraqaq" if (lv == "tafkheem_or_taqeeq" and l == "ر") else lab)
check("sifat: relaxed ignores tafkheem", not g)

# ---------------------------------------------------------------- real WebSocket handler with a fake socket
fa = types.ModuleType("fastapi")
class FastAPI:
    def __init__(self, **kw): self.routes = {}
    def add_middleware(self, *a, **k): pass
    def __getattr__(self, kind):
        if kind in ("get", "post", "websocket"):
            return lambda path: (lambda f: self.routes.__setitem__((kind, path), f) or f)
        raise AttributeError(kind)
fa.FastAPI = FastAPI; fa.File = fa.Form = lambda *a, **k: None; fa.HTTPException = Exception; fa.UploadFile = object; fa.WebSocket = object
mw = types.ModuleType("fastapi.middleware"); cors = types.ModuleType("fastapi.middleware.cors"); cors.CORSMiddleware = object
resp = types.ModuleType("fastapi.responses"); resp.JSONResponse = lambda x: x
sys.modules.update({"fastapi": fa, "fastapi.middleware": mw, "fastapi.middleware.cors": cors, "fastapi.responses": resp})

class FakeRuntime:
    int8 = False; device = "cpu"
    def __init__(self, speech):
        self.speech = speech
        self.ab = ((speech.audio * 32767).astype("<i2").astype(np.float32) / 32768.0).astype(np.float32).tobytes()
    def ensure_loaded(self): pass
    def get_ref(self, s, a): return ref
    def status(self): return {"status": "ok"}
    def recognize(self, wave, ref_):
        off = self.ab.find(wave[:800].tobytes()) // 4
        sel = [ch for (tm, ch, _w) in self.speech.chars if off / M.SR <= tm < (off + len(wave)) / M.SR]
        return M.Recognition("".join(sel), [0.97] * len(sel), None)

class FakeWS:
    def __init__(self, qp): self.query_params = qp; self.q = asyncio.Queue(); self.sent = []; self.closed = False
    async def accept(self): pass
    async def send_text(self, t): self.sent.append(json.loads(t))
    async def receive(self): return await self.q.get()
    async def close(self):
        if not self.closed: self.closed = True; await self.q.put({"type": "websocket.disconnect"})

async def ws_case(stop_at=None):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    handler = M.build_fastapi_app(FakeRuntime(sp)).routes[("websocket", "/ws/live-recitation")]
    ws = FakeWS({"sura_idx": "2", "aya_idx": "255", "strictness": "balanced"})
    task = asyncio.create_task(handler(ws)); end = sp.total if stop_at is None else stop_at; t = 0.0
    while t < end and not ws.closed:
        await ws.q.put({"type": "websocket.receive", "bytes": (sp.audio[int(t*M.SR): int((t+0.1)*M.SR)] * 32767).astype("<i2").tobytes()})
        t += 0.1; await asyncio.sleep(0.004)
    if not ws.closed: await ws.q.put({"type": "websocket.receive", "text": json.dumps({"type": "stop"})})
    await asyncio.wait_for(task, 30)
    return ws, [m for m in ws.sent if m["type"] == "final"]
async def ws_abrupt():
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    handler = M.build_fastapi_app(FakeRuntime(sp)).routes[("websocket", "/ws/live-recitation")]
    ws = FakeWS({}); task = asyncio.create_task(handler(ws))
    for i in range(30):
        await ws.q.put({"type": "websocket.receive", "bytes": (sp.audio[i*1600:(i+1)*1600] * 32767).astype("<i2").tobytes()}); await asyncio.sleep(0.003)
    await ws.q.put({"type": "websocket.disconnect"}); await asyncio.wait_for(task, 10)
async def ws_all():
    ws, fin = await ws_case(None)
    kinds_ = [m["type"] for m in ws.sent]
    check("websocket: ready -> progress... -> final(50 ok, complete) -> closed, no errors",
          kinds_[0] == "ready" and len(fin) == 1 and fin[0]["summary"]["ok"] == 50 and ws.closed and "error" not in kinds_, fin[:1])
    ws, fin = await ws_case(14.0)
    check("websocket: stop button -> final with judged words, then closed", len(fin) == 1 and 15 <= fin[0]["summary"]["judged"] <= 24 and ws.closed, fin[:1])
    await ws_abrupt(); check("websocket: abrupt network drop ends the handler cleanly", True)
asyncio.run(ws_all())

print("\n%d check(s) failed" % len(FAILS) if FAILS else "\nALL CHECKS PASSED")
sys.exit(1 if FAILS else 0)
