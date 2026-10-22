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
check("reference: stop (waqf) forms computed for words that end in a vowel/tanween",
      ref.info["waqf_forms"] >= 20 and ref.waqf[9] and ref.waqf[11], ref.info.get("waqf_forms"))

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
bal_, rel_, str_ = M.get_preset("balanced"), M.get_preset("relaxed"), M.get_preset("strict")
def mk(exp, seg, p): return [i.k for i in M.judge_word([exp], seg, None, p)[0]]
nat = "سَمَااوَاتِ"
check("madd natural: exact 2 ok everywhere", all(mk(nat, nat, p) == [] for p in (bal_, rel_, str_)))
check("madd natural shortened to 1: balanced+strict flag, relaxed tolerates",
      mk(nat, nat.replace("اا", "ا", 1), bal_) == ["madd_short"] and mk(nat, nat.replace("اا", "ا", 1), str_) == ["madd_short"]
      and mk(nat, nat.replace("اا", "ا", 1), rel_) == [])
check("madd natural stretched to 3: only strict flags; to 4: balanced flags too; relaxed tolerates 4",
      mk(nat, nat.replace("اا", "ااا", 1), bal_) == [] and mk(nat, nat.replace("اا", "ااا", 1), str_) == ["madd_long"]
      and mk(nat, nat.replace("اا", "اااا", 1), bal_) == ["madd_long"] and mk(nat, nat.replace("اا", "اااا", 1), rel_) == [])
check("madd dropped entirely is flagged in every preset", all(mk(nat, nat.replace("اا", "", 1), p) == ["madd_dropped"] for p in (bal_, rel_, str_)))
mut = "شَااااءَ"
check("madd muttasil: 4 ok, 2 flagged (4 harakat needed)", mk(mut, mut, bal_) == [] and mk(mut, "شَااءَ", bal_) == ["madd_short"])

# ---- the two real mistakes seen in the live log (QPS uses ۦ for kasra-madd): must read as ONE tashkeel issue
iss = M.judge_word(["لمِهِۦۦۦۦ"], "لمِهَاا", None, str_)[0]
check("live log w36 (هِۦ said as هَا): a single yellow tashkeel issue, no confusing tajweed/letters note",
      [i.t for i in iss] == ["tashkeel"] and iss[0].k == "vowel_swap", [(i.t, i.k) for i in iss])
iss = M.judge_word(["اشَااااءَ"], "اشِۦۦۦۦءَ", None, str_)[0]
check("live log w39 (شَاءَ said as شِيء): single tashkeel issue", [i.t for i in iss] == ["tashkeel"], [(i.t, i.k) for i in iss])
iss = M.judge_word(["بِمَاا"], "بِمَۦۦ", None, str_)[0]
check("madd letter swapped alone (ا -> ۦ) is reported as tashkeel 'madd_vowel_swap' with Arabic names",
      [(i.t, i.k) for i in iss] == [("tashkeel", "madd_vowel_swap")] and "الياء" in iss[0].n and "الألف" in iss[0].n, [(i.t, i.k) for i in iss])
iss = M.judge_word(["بِمَاا"], "بِمَيي", None, str_)[0]
check("madd alif said as a real ya is a letters error", [(i.t, i.k) for i in iss] == [("letters", "replace")], [(i.t, i.k) for i in iss])

# ---- segmentation ladder: the library's spaced output may misbehave in several ways
def bounds(r): return r.starts
for mode in ("none", "waqf_all", "merge"):
    r2 = M.build_ref_aya(2, 255, sim.FakeLib(spaced=mode))
    drift = max(abs(a - b) for a, b in zip(bounds(r2), bounds(ref)))
    want = ("spaced+aligned",) if mode == "merge" else ("aligned",)
    check("ladder/%s: spaced output unusable/merged -> %s segmentation, boundaries within 2 chars" % (mode, want[0]),
          r2.info["segmentation"] in want and drift <= 2 and r2.n == 50, (r2.info.get("segmentation"), drift))
    check("ladder/%s: no word starts with a vowel mark or silah letter (leaked-vowel artefact)" % mode,
          all(not M.is_mark(f[0]) and f[0] not in "ۥۦ" for f in r2.wasl), [f[:3] for f in r2.wasl if M.is_mark(f[0]) or f[0] in "ۥۦ"])
    check("ladder/%s: stop forms still found" % mode, r2.info.get("waqf_forms", 0) >= 15, r2.info.get("waqf_forms"))
    eng, log, final, sp = sim.run_stream(r2, sim.Speech(ref, BASE, pauses=PAUSES, seed=3), preset="strict")
    check("ladder/%s: clean recitation -> 50 ok, none flagged (strict)" % mode,
          final["summary"]["ok"] == 50 and not flagged(final), (final["summary"], list(flagged(final))[:6]))
    spoken, exp = inject(BASE)
    eng, log, final, sp = sim.run_stream(r2, sim.Speech(ref, spoken, pauses=PAUSES, seed=5), preset="strict")
    got = flagged(final)
    check("ladder/%s: the 6 injected mistakes are still found (strict), no extras" % mode,
          set(got) == set(exp) and all(got[w]["s"] == e[0] for w, e in exp.items()),
          {w: got[w]["s"] if w in got else None for w in exp} | {"extra": sorted(set(got) - set(exp))})
r_adopt = M.build_ref_aya(2, 255, sim.FakeLib(spaced="ok"))
check("ladder/ok: exact spaced segmentation is preferred", r_adopt.info["segmentation"] == "spaced-exact")

# ---- stop-form without a stop: silence in the audio decides
def waqf_case(preset, pause):
    spoken = list(BASE); spoken[9] = ref.waqf[9]            # سِنَةٌ said in stop form
    p = set(PAUSES) | ({9} if pause else set()); p = p - ({9} if not pause else set())
    eng, log, final, sp = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=p, seed=7), preset=preset)
    return flagged(final)
g = waqf_case("strict", pause=False)
check("waqf without a pause: strict flags the dropped final vowel (yellow)",
      list(g) == [9] and g[9]["s"] == "tashkeel" and kinds(g[9]) == ["waqf_no_pause"], {w: (g[w]["s"], kinds(g[w])) for w in g})
check("waqf WITH a real pause: strict does not flag", waqf_case("strict", pause=True) == {})
check("waqf without a pause: balanced does not flag (default stays conservative)", waqf_case("balanced", pause=False) == {})

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


# ---- regression tests built from the REAL strings in the live report
real_groups = ["ءَللَااهُ", "لَاااا", "ءِلَااهَ", "ءِللَاا", "هُوَ", "لحَييُ", "لقَييُۥۥمُ", "لَاا", "تَءخُذُهُۥۥ", "سِنَتُوووَلَاا"]
real_F = "".join(real_groups)
gs, acc = [], 0
for gr in real_groups[:-1]:
    acc += len(gr); gs.append(acc)
est = [0] + [g + (1 if i % 2 == 0 else -1) for i, g in enumerate(gs)] + [len(real_F)]    # +-1 estimation noise
snapped, exact = M._snap_to_library_groups(est, gs)
check("real log: estimated boundaries snap to the library's exact group edges",
      snapped[1:-1] == gs and all(exact[1:-1]), (snapped, gs))
leak = "كُرسِييُهُ" + "لسسَمَااوَاتِ"
b = [0, len("كُرسِييُه"), len(leak)]                    # boundary placed BEFORE the final damma (real artefact)
check("real log: a final vowel mark never leaks into the next word", M.refine_bounds(leak, b)[1] == len("كُرسِييُهُ"), M.refine_bounds(leak, b))
leak2 = "هُوَ" + "لعَلِييُ"
check("real log: silah/madd letters stay with the previous word",
      M.refine_bounds("هُو" + "ۥۥ" + "لحَييُ", [0, 3, 9])[1] == 5)

# ---- shaddah / ghunna length strictness (junction 'ووو' = idgham with ghunna)
junction = "سِنَتُوووَلَاا"
def jk(seg, p): return [i.k for i in M.judge_word([junction], seg, None, M.get_preset(p))[0]]
check("ghunna/shaddah run: exact length ok everywhere", all(jk(junction, p) == [] for p in ("relaxed", "balanced", "strict")))
check("ghunna/shaddah run: one unit short -> strict flags, balanced+relaxed tolerate",
      jk("سِنَتُووَلَاا", "strict") == ["shaddah_dropped"] and jk("سِنَتُووَلَاا", "balanced") == [] and jk("سِنَتُووَلَاا", "relaxed") == [])
check("ghunna/shaddah run: nearly gone -> balanced and strict flag",
      jk("سِنَتُوَلَاا", "balanced") == ["shaddah_dropped"] and jk("سِنَتُوَلَاا", "strict") == ["shaddah_dropped"])

# ---- tolerant sifat mapping (22 % of real windows had pred_mapped=false)
grp = [{"g": "ءَ"}, {"g": "لل"}, {"g": "َا"}]
m = M.tolerant_group_map(grp, "ءَللَِا")
check("sifat mapping: survives a doubled vowel mark at a window edge", m == [0, 0, 1, 1, 2, 2, 2], m)
check("sifat mapping: refuses unrelated text", M.tolerant_group_map(grp, "بتثجحخدذرزسشص") is None)

# ---- [PAD] labels (prob 0.0, seen in the live report) must never produce a flag
def sifat_pad(preset):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); holder = {}; plain = sp.recognizer(holder)
    def rec(wave):
        r = plain(wave); items = []
        for g in sim.fake_groups(r.text):
            items.append({"phonemes_group": g.phonemes,
                          "tafkheem_or_taqeeq": types.SimpleNamespace(text="[PAD]", prob=0.0),
                          "qalqla": types.SimpleNamespace(text="[PAD]", prob=0.0),
                          "ghonna": types.SimpleNamespace(text="[PAD]", prob=0.0)})
        return M.Recognition(r.text, r.probs, M.convert_sifat(items))
    eng = M.SessionEngine(ref, rec, M.get_preset(preset)); holder["engine"] = eng
    t = 0.0
    while t < sp.total and not eng.complete:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.1) * M.SR)]); t += 0.1
        if eng.ready_to_run(): eng.step()
    return flagged(eng.flush())
check("sifat: [PAD] / prob 0.0 predictions never flag a word", sifat_pad("strict") == {})


# ====================================================================== v14.3: regression tests from the live report
import re as _re
# ---- the live failure: a recitation recorded while the server was cold-starting arrives as one big backlog
def backlog_run(preset="strict", burst_s=50.0, chunk_s=10.0, spoken=None):
    """~50 s of audio recorded while the server was cold-starting arrives at once, then the rest trickles in."""
    sp = sim.Speech(ref, spoken or BASE, scale=2.0, pauses=PAUSES, seed=21, lead=10.0, pause_s=1.2)
    holder = {}; eng = M.SessionEngine(ref, sp.recognizer(holder), M.get_preset(preset)); holder["engine"] = eng
    eng.add_audio(sp.audio[: int(burst_s * M.SR)]); t = burst_s
    if eng.ready_to_run(): eng.step()
    while t < sp.total:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + chunk_s) * M.SR)]); t += chunk_s
        if eng.ready_to_run(): eng.step()                 # ONE window per arrival -> the backlog stays large
    return eng, eng.flush(), sp
eng, fin, sp = backlog_run()
check("live failure: a %.0f s recitation delivered as a growing backlog is judged completely (50/50)" % sp.total,
      fin["summary"]["judged"] == 50 and fin["summary"]["complete"] and not flagged(fin), fin["summary"])
saved = M.MAX_BACKLOG_S
M.MAX_BACKLOG_S = 45.0                                    # the v14.2 limit: this must FAIL (proves the test is meaningful)
eng_old, fin_old, _sp = backlog_run()
M.MAX_BACKLOG_S = saved
check("live failure: the old 45 s limit really did strand words (test is meaningful)", fin_old["summary"]["judged"] < 40, fin_old["summary"]["judged"])

# ---- adjacent swapped sifat labels (the live report: meem loses ghunna, next lam gains it) are not mistakes
def sifat_custom(mutate, preset="strict"):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); holder = {}; plain = sp.recognizer(holder)
    def rec(wave):
        r = plain(wave); groups = sim.fake_groups(r.text); items = []
        for g in groups:
            items.append({"phonemes_group": g.phonemes,
                          "qalqla": types.SimpleNamespace(text=g.qalqla, prob=0.97),
                          "tafkheem_or_taqeeq": types.SimpleNamespace(text=g.tafkheem_or_taqeeq, prob=0.97),
                          "ghonna": types.SimpleNamespace(text=g.ghonna, prob=0.97)})
        mutate(items)
        return M.Recognition(r.text, r.probs, M.convert_sifat(items))
    eng = M.SessionEngine(ref, rec, M.get_preset(preset)); holder["engine"] = eng
    t = 0.0
    while t < sp.total and not eng.complete:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.1) * M.SR)]); t += 0.1
        if eng.ready_to_run(): eng.step()
    return flagged(eng.flush())
def lbl(items, i, key, text): items[i][key] = types.SimpleNamespace(text=text, prob=0.97)
APPLIED = {"n": 0}
def first_letter(g): return next((c for c in g if not M.is_mark(c)), "")
def candidates(items):
    """(i, i+1): a noon/meem group followed by a group of a letter that can never carry ghunna"""
    return [i for i in range(len(items) - 1)
            if first_letter(items[i]["phonemes_group"]) in "نم"
            and first_letter(items[i + 1]["phonemes_group"]) not in "نمويى" and first_letter(items[i + 1]["phonemes_group"])]
def swap_pair(items):
    for i in candidates(items):
        lbl(items, i, "ghonna", "not_maghnoon"); lbl(items, i + 1, "ghonna", "maghnoon"); APPLIED["n"] += 1
def only_first(items):
    for i in candidates(items):
        lbl(items, i, "ghonna", "not_maghnoon"); APPLIED["n"] += 1
def extra_on_plain(items):
    for i in range(len(items)):
        if first_letter(items[i]["phonemes_group"]) in "لتبفكس":
            lbl(items, i, "ghonna", "maghnoon"); APPLIED["n"] += 1
APPLIED["n"] = 0
g_swap = sifat_custom(swap_pair); n_swap = APPLIED["n"]
check("sifat: swapped neighbouring ghunna labels (live report w6/w7) are not flagged (mutation applied %d times)" % n_swap,
      n_swap > 10 and g_swap == {}, (n_swap, list(g_swap)))
APPLIED["n"] = 0
g_one = sifat_custom(only_first); n_one = APPLIED["n"]
check("sifat: a genuinely missing ghunna (no neighbour swap) is still flagged (applied %d times)" % n_one,
      n_one > 10 and any("ghonna" in kinds(w) for w in g_one.values()), (n_one, list(g_one)))
APPLIED["n"] = 0
g_extra = sifat_custom(extra_on_plain); n_extra = APPLIED["n"]
check("sifat: 'ghunna' on ل/ت/ب/ف/ك/س is impossible and never flagged (applied %d times)" % n_extra,
      n_extra > 10 and g_extra == {}, (n_extra, list(g_extra)))
# prove the guards are what make those pass: with the guards disabled the same mutations DO flag words
_sg, _sh = M.GHUNNA_LETTERS, M.SessionEngine._label_shift
M.GHUNNA_LETTERS = set("ءابتثجحخدذرزسشصضطظعغفقكلمنهوي"); M.SessionEngine._label_shift = lambda *a, **k: False
g_nog = sifat_custom(swap_pair); g_nog2 = sifat_custom(extra_on_plain)
M.GHUNNA_LETTERS, M.SessionEngine._label_shift = _sg, _sh
check("sifat: with the guards disabled those same mutations are flagged (the tests have teeth)", len(g_nog) > 0 and len(g_nog2) > 0, (len(g_nog), len(g_nog2)))

# ---- a final vowel kept while the stop form is cheaper: report the VOWEL mistake, not 'vowel on a sukun'
iss = M.judge_word(["تَءخُذُهُۥۥ", "تَءخُذُه"], "تَءخُذُهَاا", None, M.get_preset("strict"))[0]
check("live log w8 (تَأْخُذُهُ said ـهَا): one clear vowel_swap on ه (not 'vowel on a sukun')",
      [(i.t, i.k) for i in iss] == [("tashkeel", "vowel_swap")] and "«ه»" in iss[0].n, [(i.t, i.k) for i in iss])

# ---- client silence gate: dead air is dropped, but pauses/word edges/madd are untouched
for lbl_, kw in (("long 3 s pauses + 4 s lead/trail", dict(pause_s=3.0, lead=4.0, trail=4.0)),
                 ("normal pauses", dict(pause_s=0.6, lead=1.0, trail=1.5))):
    sp0 = sim.Speech(ref, BASE, pauses=PAUSES, seed=3, **kw)
    sp1 = sim.client_trim(sp0)
    saved_pct = 100.0 * (1 - len(sp1.audio) / len(sp0.audio))
    eng, log, fin, _ = sim.run_stream(ref, sp1, preset="strict")
    check("silence gate (%s): saves %.0f%% of the audio and the verdict is unchanged (50 ok)" % (lbl_, saved_pct),
          fin["summary"]["ok"] == 50 and not flagged(fin) and (saved_pct > 25 if "long" in lbl_ else True), (fin["summary"], saved_pct))
spoken, exp = inject(BASE)
sp1 = sim.client_trim(sim.Speech(ref, spoken, pauses=PAUSES, seed=5, pause_s=3.0, lead=4.0, trail=4.0))
eng, log, fin, _ = sim.run_stream(ref, sp1, preset="strict"); got = flagged(fin)
check("silence gate: all 6 injected mistakes still found, nothing extra", set(got) == set(exp), (sorted(got), sorted(exp)))
for pause, expect in ((True, set()), (False, {9})):
    spoken9 = list(BASE); spoken9[9] = ref.waqf[9]
    p = (set(PAUSES) | {9}) if pause else (set(PAUSES) - {9})
    sp1 = sim.client_trim(sim.Speech(ref, spoken9, pauses=p, seed=7, pause_s=3.0))
    eng, log, fin, _ = sim.run_stream(ref, sp1, preset="strict")
    check("silence gate keeps the stop-form logic exact (%s a real pause)" % ("with" if pause else "without"),
          set(flagged(fin)) == expect, list(flagged(fin)))

# ---- Android <-> backend contract audit
HERE = os.path.dirname(os.path.abspath(__file__))
kt_dir = os.path.join(HERE, "..", "..", "android", "app", "src", "main", "java", "com", "rafiq", "app")
kt = open(os.path.join(kt_dir, "RecitationSession.kt"), encoding="utf-8").read()
cfg_kt = open(os.path.join(kt_dir, "RecitationConfig.kt"), encoding="utf-8").read()
kotlin_keys = set(_re.findall(r'opt(?:String|Int|Long|Boolean|JSONArray|JSONObject)\("([A-Za-z_]+)"', kt))
spoken, exp = inject(BASE)
eng, log, fin, _ = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=PAUSES, seed=5), preset="strict")
msgs = [m for _t, m in log] + [fin]
fin_ws = None
def all_keys(o, acc):
    if isinstance(o, dict):
        for k, v in o.items(): acc.add(k); all_keys(v, acc)
    elif isinstance(o, list):
        for v in o: all_keys(v, acc)
    return acc
server_keys = set()
for m_ in msgs: all_keys(m_, server_keys)
server_keys |= {"words", "type"}        # 'ready' carries the word list (checked by the websocket test above)
final_msg = dict(eng.final_message()); final_msg["debug"] = M.debug_snapshot(eng, sim_runtime := types.SimpleNamespace(status=lambda: {}))
all_keys(final_msg, server_keys)
missing = sorted(k for k in kotlin_keys if k not in server_keys and k != "message")
check("contract: every JSON field the Android app reads exists in the server messages", not missing, missing)
check("contract: server sends the 'message' field of error messages", '"message"' in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read())
check("contract: app sends {\"type\":\"stop\"} and the server understands it",
      '{\\"type\\":\\"stop\\"}' in kt and 'kind == "stop"' in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read())
check("contract: websocket query parameters match", all(k in cfg_kt for k in ("sura_idx=", "aya_idx=", "strictness="))
      and all(('qp.get("%s"' % k) in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read() for k in ("sura_idx", "aya_idx", "strictness")))
check("contract: audio is 16 kHz mono PCM16 little-endian on both sides",
      "SAMPLE_RATE: Int = 16000" in cfg_kt and "SR = 16000" in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read()
      and "(s shr 8) and 0xFF" in kt and 'dtype="<i2"' in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read())


# ====================================================================== v14.3b: Hafs policy, ghunna symbol, ibtida', sub-phonemic
S_, B_, R_ = M.get_preset("strict"), M.get_preset("balanced"), M.get_preset("relaxed")
def mkk(exp, seg, p): return [i.k for i in M.judge_word([exp], seg, None, p)[0]]
# madd munfasil / silah kubra: 2 (qasr) .. 5 are all valid in Hafs 'an 'Asim; < 2 or > 5 is an error
munf = "لَاااا"
check("hafs: madd munfasil with 2 harakat (qasr) is valid in every preset",
      all(mkk(munf, "لَاا", p) == [] for p in (S_, B_, R_)))
check("hafs: madd munfasil 4 and 5 harakat are valid", all(mkk(munf, "لَ" + "ا" * n, S_) == [] for n in (4, 5)))
check("hafs: madd munfasil dropped to 1 harakah is an error", mkk(munf, "لَا", S_) == ["madd_short"])
check("hafs: madd munfasil stretched past 5: strict flags 6, balanced tolerates 6 but flags 7",
      mkk(munf, "لَ" + "ا" * 6, S_) == ["madd_long"] and mkk(munf, "لَ" + "ا" * 6, B_) == []
      and mkk(munf, "لَ" + "ا" * 7, B_) == ["madd_long"])
silah = "عِںںںدَهُۥۥۥۥ"
check("hafs: silah kubra with 2 harakat (live report w23) is valid", all(mkk(silah, "عِںںںدَهُۥۥ", p) == [] for p in (S_, B_, R_)))
check("hafs: madd muttasil still needs 4 (2 is an error)", mkk("شَااااءَ", "شَااءَ", S_) == ["madd_short"] and mkk("شَااااءَ", "شَااااءَ", S_) == [])
arid = "لعَظِۦۦۦۦم"
check("hafs: madd 'arid at a stop is valid at 2, 4 and 6 (7 flagged)",
      all(mkk(arid, "لعَظِ" + "ۦ" * n + "م", S_) == [] for n in (2, 4, 6)) and mkk(arid, "لعَظِ" + "ۦ" * 7 + "م", S_) == ["madd_long"])
# ghunna symbol runs (ikhfaa): ںںں in the live output
ikh = "مَںںںذَ"
check("ghunna ikhfaa: exact length ok; one short -> strict flags; two short -> balanced+strict flag; relaxed tolerates",
      all(mkk(ikh, ikh, p) == [] for p in (S_, B_, R_))
      and mkk(ikh, "مَںںذَ", S_) == ["ghunna_short"] and mkk(ikh, "مَںںذَ", B_) == []
      and mkk(ikh, "مَںذَ", B_) == ["ghunna_short"] and mkk(ikh, "مَںذَ", S_) == ["ghunna_short"] and mkk(ikh, "مَںذَ", R_) == [])
check("ghunna ikhfaa: ikhfaa not applied at all (noon instead of the ghunna symbol) is a tajweed 'rule' error",
      mkk(ikh, "مَنذَ", S_) == ["rule"])
# liin letter stretched like a madd (ya of بَيْنَ)
check("liin letter: بَيْنَ with an elongated ya -> strict flags, balanced tolerates 2 but flags 3",
      mkk("بَينَ", "بَينَ", S_) == [] and mkk("بَينَ", "بَيينَ", S_) == ["lin_extended"]
      and mkk("بَينَ", "بَيينَ", B_) == [] and mkk("بَينَ", "بَييينَ", B_) == ["lin_extended"])

# ---- vowel on a different letter is alignment drift (stutter after a stop), a real swap on the same letter is not
check("drift: a heard vowel that belongs to another letter is not a vowel_swap",
      M.judge_word(["لءَرضِ"], "لءَرضمَن", None, S_, lead=0, trail=3)[0] == []
      and [i.k for i in M.judge_word(["لءَرضِ"], "لءَرضَ", None, S_)[0]] == ["vowel_swap"])
# the same alignment-drift case at window level: reciter stops, then repeats the next word
st_ = list(BASE); st_[20] = st_[19] + st_[20]
eng_, log_, fin_, sp_ = sim.run_stream(ref, sim.Speech(ref, st_, pauses=PAUSES, seed=18), preset="strict")
check("drift: stop + repeating the next word is judged clean (50 ok, strict)", fin_["summary"]["ok"] == 50 and not flagged(fin_), {w: kinds(x) for w, x in flagged(fin_).items()})

# ---- ibtida' (start) forms: after a pause the word starts fresh (live report w11->w12: tanween idgham not applied)
check("ibtida': start forms were computed (hamzat al-wasl / no assimilation)", ref.info.get("start_forms", 0) >= 6, ref.info.get("start_forms"))
check("ibtida': لَهُ after a paused نَوْمٌ has a distinct start form (single lam, not the idgham 'لل')",
      ref.start_forms[12] and ref.start_forms[12] != ref.wasl[12] and ref.wasl[12].startswith("لل") and not ref.start_forms[12].startswith("لل"),
      (ref.wasl[12], ref.start_forms[12]))
def with_ibtida(r, pauses):
    out = []
    for k in range(r.n):
        after, at = (k - 1) in pauses, k in pauses
        if after and at: f = r.start_waqf[k] or r.start_forms[k] or r.waqf[k] or r.wasl[k]
        elif after: f = r.start_forms[k] or r.wasl[k]
        elif at: f = r.waqf[k] or r.wasl[k]
        else: f = r.wasl[k]
        out.append(f)
    return out
IB_PAUSES = PAUSES | {9, 11}
spoken_ib = with_ibtida(ref, IB_PAUSES)
eng, log, fin, sp = sim.run_stream(ref, sim.Speech(ref, spoken_ib, pauses=IB_PAUSES, seed=31), preset="strict")
check("ibtida': a recitation that restarts words fresh after every pause is 50/50 correct (strict)",
      fin["summary"]["ok"] == 50 and not flagged(fin), (fin["summary"], {w: kinds(x) for w, x in flagged(fin).items()}))
saved_sf, saved_sw = list(ref.start_forms), list(ref.start_waqf)
ref.start_forms, ref.start_waqf = [None] * ref.n, [None] * ref.n
eng, log, fin_no, sp = sim.run_stream(ref, sim.Speech(ref, spoken_ib, pauses=IB_PAUSES, seed=31), preset="strict")
ref.start_forms, ref.start_waqf = saved_sf, saved_sw
check("ibtida': without start forms the same correct recitation IS falsely flagged (the test has teeth)", len(flagged(fin_no)) > 0, len(flagged(fin_no)))


# ---- a fresh start without a stop: hamzat al-wasl pronounced / idgham not applied in continuous reading
def fresh_start_case(preset, k, pause_before):
    spoken = list(BASE); spoken[k] = ref.start_forms[k]
    p = set(PAUSES); p = (p | {k - 1}) if pause_before else (p - {k - 1})
    if not pause_before:
        spoken[k - 1] = ref.wasl[k - 1]          # no stop: the previous word is spoken in its continuing form
    eng, log, fin, sp = sim.run_stream(ref, sim.Speech(ref, spoken, pauses=p, seed=41), preset=preset)
    return flagged(fin)
g1 = fresh_start_case("strict", 5, pause_before=False)                      # الْحَيُّ said as ءَلحَيّ right after هُوَ
check("hamzat al-wasl pronounced in continuous reading (strict): flagged as a tajweed 'start_no_pause'",
      list(g1) == [5] and g1[5]["s"] == "tajweed" and kinds(g1[5]) == ["start_no_pause"], {w: (x["s"], kinds(x)) for w, x in g1.items()})
check("the same fresh start AFTER a real stop is correct (no flag)", fresh_start_case("strict", 5, pause_before=True) == {})
check("balanced does not apply the start-without-pause check (conservative default)", fresh_start_case("balanced", 5, pause_before=False) == {})
g2 = fresh_start_case("strict", 12, pause_before=False)                     # لَهُ said with a single lam although نَوْمٌ was not stopped on
check("idgham not applied after نَوْمٌ in continuous reading (strict): flagged", 12 in g2 and "start_no_pause" in kinds(g2[12]), {w: kinds(x) for w, x in g2.items()})

# ---- strict: sub-phonemic mistakes (right phoneme, wrong articulation) via the other sifat levels
def sifat_generic(preset, letter, key, new_label):
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5); holder = {}; plain = sp.recognizer(holder)
    def rec(wave):
        r = plain(wave); items = []
        for g in sim.fake_groups(r.text):
            L = next((c for c in g.phonemes if not M.is_mark(c)), "")
            d = {"phonemes_group": g.phonemes}
            for lv in ("qalqla", "tafkheem_or_taqeeq", "ghonna", "hams_or_jahr"):
                lab = getattr(g, lv)
                if lv == key and L == letter: lab = new_label
                d[lv] = types.SimpleNamespace(text=lab, prob=0.95)
            items.append(d)
        return M.Recognition(r.text, r.probs, M.convert_sifat(items))
    eng = M.SessionEngine(ref, rec, M.get_preset(preset)); holder["engine"] = eng
    t = 0.0
    while t < sp.total and not eng.complete:
        eng.add_audio(sp.audio[int(t * M.SR): int((t + 0.1) * M.SR)]); t += 0.1
        if eng.ready_to_run(): eng.step()
    return flagged(eng.flush())
g_s = sifat_generic("strict", "ب", "hams_or_jahr", "hams")
g_b = sifat_generic("balanced", "ب", "hams_or_jahr", "hams")
check("strict: ب articulated whispered (hams instead of jahr) is flagged as a sifa mistake; balanced ignores it",
      len(g_s) > 0 and all(any(k.startswith("sifa:") for k in kinds(w)) for w in g_s.values()) and g_b == {}, (len(g_s), len(g_b)))


check("contract: the app uploads to /analyze with the same form fields the server declares",
      all(('addFormDataPart("%s"' % k) in kt for k in ("file", "sura_idx", "aya_idx", "strictness"))
      and '@api.post("/analyze")' in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read()
      and all(("%s: " % k) in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read().split("async def analyze(")[1].split(") ->")[0]
              for k in ("file", "sura_idx", "aya_idx", "strictness"))
      and 'fun analyzeUrl' in cfg_kt and '"/analyze"' in cfg_kt)
check("contract: the uploaded WAV header matches what the server decodes (RIFF/WAVE, 16 kHz, mono, PCM16)",
      all(s_ in kt for s_ in ('"RIFF"', '"WAVE"', "putInt(sampleRate)", "header.putShort(1)", "header.putShort(16)"))
      and "decode_audio_bytes" in open(os.path.join(HERE, "..", "modal_app.py"), encoding="utf-8").read())

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
async def ws_burst():
    sp = sim.Speech(ref, BASE, pauses=PAUSES, seed=5)
    handler = M.build_fastapi_app(FakeRuntime(sp)).routes[("websocket", "/ws/live-recitation")]
    ws = FakeWS({"sura_idx": "2", "aya_idx": "255", "strictness": "strict"}); task = asyncio.create_task(handler(ws))
    pcm = (sp.audio * 32767).astype("<i2").tobytes()
    for i in range(0, len(pcm), 3200 * 10):
        await ws.q.put({"type": "websocket.receive", "bytes": pcm[i:i + 3200 * 10]})
    await ws.q.put({"type": "websocket.receive", "text": json.dumps({"type": "stop"})})
    await asyncio.wait_for(task, 60)
    return ws, [m for m in ws.sent if m["type"] == "final"]
async def ws_all():
    ws, fin = await ws_case(None)
    kinds_ = [m["type"] for m in ws.sent]
    check("websocket: ready -> progress... -> final(50 ok, complete) -> closed, no errors",
          kinds_[0] == "ready" and len(fin) == 1 and fin[0]["summary"]["ok"] == 50 and ws.closed and "error" not in kinds_, fin[:1])
    ws, fin = await ws_case(14.0)
    check("websocket: the final message carries the debug trace (server may scale to zero afterwards)",
          "debug" in fin[0] and "ref_info" in fin[0]["debug"] and len(fin[0]["debug"]["steps"]) > 0)
    ws, fin = await ws_case(14.0)
    check("websocket: stop button -> final with judged words, then closed", len(fin) == 1 and 15 <= fin[0]["summary"]["judged"] <= 24 and ws.closed, fin[:1])
    ws, fin = await ws_burst()
    prog = [m for m in ws.sent if m["type"] == "progress"]
    check("websocket: whole recording sent at once then stop -> progress keeps streaming while catching up, final is complete",
          len(prog) >= 4 and len(fin) == 1 and fin[0]["summary"]["judged"] == 50, (len(prog), fin[:1]))
    await ws_abrupt(); check("websocket: abrupt network drop ends the handler cleanly", True)
asyncio.run(ws_all())

# ---- /analyze (record-then-upload) and the compact technical report
import wave as _wave, io as _io
class FakeUpload:
    def __init__(self, data): self.data = data
    async def read(self): return self.data
def wav_bytes(x):
    b = _io.BytesIO()
    with _wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(M.SR); w.writeframes((x * 32767).astype("<i2").tobytes())
    return b.getvalue()
async def analyze_case():
    spoken, exp = inject(BASE)
    sp = sim.Speech(ref, spoken, pauses=PAUSES, seed=5, pause_s=2.0, lead=3.0, trail=3.0)
    sp = sim.client_trim(sp)
    handler = M.build_fastapi_app(FakeRuntime(sp)).routes[("post", "/analyze")]
    res = await handler(file=FakeUpload(wav_bytes(sp.audio)), sura_idx=2, aya_idx=255, strictness="strict")
    return res, exp
res, exp_ = asyncio.run(analyze_case())
got_ = {w["i"] for w in res["words"] if w["s"] != "ok"}
check("/analyze: a trimmed WAV upload is judged completely (50 words) and finds the 6 mistakes, nothing extra",
      res["summary"]["judged"] == 50 and got_ == set(exp_), (res["summary"], sorted(got_)))
size_kb = len(json.dumps(res["debug"], ensure_ascii=False).encode("utf-8")) / 1024
check("/analyze: the technical report has all 50 words and fits in a chat message (%.0f KB)" % size_kb,
      len(res["debug"]["words"]) == 50 and size_kb < 60, size_kb)
check("/analyze: response carries every field the app reads", all(k in res for k in ("type", "words", "summary", "debug")) and "complete" in res["summary"])


print("\n%d check(s) failed" % len(FAILS) if FAILS else "\nALL CHECKS PASSED")
sys.exit(1 if FAILS else 0)
