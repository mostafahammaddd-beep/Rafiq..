"""Dev-only simulation of the Rafiq engine (no model, no network needed)."""
import random
import sys
import types
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modal_app as M  # noqa: E402

WORDS = ["اللَّهُ", "لَا", "إِلَٰهَ", "إِلَّا", "هُوَ", "الْحَيُّ", "الْقَيُّومُ",
         "لَا", "تَأْخُذُهُ", "سِنَةٌ", "وَلَا", "نَوْمٌ",
         "لَهُ", "مَا", "فِي", "السَّمَاوَاتِ", "وَمَا", "فِي", "الْأَرْضِ",
         "مَنْ", "ذَا", "الَّذِي", "يَشْفَعُ", "عِنْدَهُ", "إِلَّا", "بِإِذْنِهِ",
         "يَعْلَمُ", "مَا", "بَيْنَ", "أَيْدِيهِمْ", "وَمَا", "خَلْفَهُمْ",
         "وَلَا", "يُحِيطُونَ", "بِشَيْءٍ", "مِنْ", "عِلْمِهِ", "إِلَّا", "بِمَا", "شَاءَ",
         "وَسِعَ", "كُرْسِيُّهُ", "السَّمَاوَاتِ", "وَالْأَرْضَ",
         "وَلَا", "يَئُودُهُ", "حِفْظُهُمَا", "وَهُوَ", "الْعَلِيُّ", "الْعَظِيمُ"]

FATHA, DAMMA, KASRA, SHADDA = "\u064e", "\u064f", "\u0650", "\u0651"
TANWEEN = {"\u064b": FATHA, "\u064c": DAMMA, "\u064d": KASRA}
SUKUN = {"\u0652", "\u06e1"}
HAMZAS = set("ٱأإآؤئء")


def fake_ph(word: str, final: bool = False) -> str:
    out: List[str] = []
    prev_mark = None
    tanween_tail = False
    for idx, ch in enumerate(word):
        nxt = word[idx + 1] if idx + 1 < len(word) else ""
        if ch in HAMZAS:
            out.append("ء"); prev_mark = None
        elif ch in M.BASIC_LETTERS or ch == "ٰ":
            if ch == "ٰ":
                out.extend(["ا", "ا"]); continue
            c = {"ى": "ا", "ة": "ت"}.get(ch, ch)
            out.append(c)
            if c in "او" or c == "ي":
                matching = {"ا": FATHA, "و": DAMMA, "ي": KASRA}[c]
                if prev_mark == matching and nxt not in (FATHA, DAMMA, KASRA, SHADDA):
                    out.append(c)                      # natural madd = 2 chars
            prev_mark = None
        elif ch in (FATHA, DAMMA, KASRA):
            out.append(ch); prev_mark = ch
        elif ch == SHADDA:
            last = next((x for x in reversed(out) if not M.is_mark(x)), "")
            if out and out[-1] in (FATHA, DAMMA, KASRA):
                out.insert(len(out) - 1, last)      # geminate stays contiguous: لل + َ
            else:
                out.append(last)
        elif ch in TANWEEN:
            out.append(TANWEEN[ch]); out.append("ن"); tanween_tail = True; prev_mark = None
    s = "".join(out)
    if final:
        if tanween_tail:
            s = s[:-2]
        while s and s[-1] in (FATHA, DAMMA, KASRA):
            s = s[:-1]
    return s


def lib_word(w: str, final: bool, initial: bool) -> str:
    """Like the real library: the hamza of 'ال' is pronounced only when the text starts there
    (isolated word / first word); in the middle of a verse it disappears (hamzat al-wasl)."""
    if w.startswith("ال"):
        body = fake_ph(w[1:], final)
        return ("ءَ" + body) if initial else body
    return fake_ph(w, final)


class FakeLib:
    """spaced: how the fake library's remove_spaces=False output misbehaves
       ok | none (no spaces) | waqf_all (every word in stop form) | merge (two words glued)"""
    def __init__(self, with_stop_token=True, spaced="ok"):
        self.with_stop = with_stop_token
        self.spaced = spaced

    def get_uthmani(self, s, a):
        toks = list(WORDS)
        if self.with_stop:
            toks.insert(7, "\u06da")          # standalone stop sign must be ignored
        return " ".join(toks)

    def phonetize(self, text, remove_spaces):
        toks = [t for t in text.split() if M._BASE_RE.search(t)]
        words = [lib_word(w, i == len(toks) - 1, i == 0) for i, w in enumerate(toks)]
        if remove_spaces:
            ph = "".join(words)
        elif self.spaced == "none":
            ph = "".join(words)
        elif self.spaced == "waqf_all":
            ph = " ".join(lib_word(w, True, i == 0) for i, w in enumerate(toks))
        elif self.spaced == "merge" and len(words) > 7:
            ws = list(words); ws[5:7] = [ws[5] + ws[6]]; ph = " ".join(ws)
        else:
            ph = " ".join(words)
        groups = fake_groups("".join(words)) if remove_spaces else None
        return types.SimpleNamespace(phonemes=ph, sifat=groups)


def fake_groups(ph: str):
    groups, cur = [], ""
    for ch in ph:
        if not M.is_mark(ch) and cur and cur[-1] != ch and not M.is_mark(cur[-1]) or \
           (not M.is_mark(ch) and cur and M.is_mark(cur[-1])):
            groups.append(cur); cur = ""
        cur += ch
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        letter = next((c for c in g if not M.is_mark(c)), "")
        out.append(types.SimpleNamespace(
            phonemes=g,
            qalqla="moqalqal" if letter in "قطبجد" else "not_moqalqal",
            tafkheem_or_taqeeq="mofakham" if letter in "صضطظقغخر" else "moraqaq",
            ghonna="maghnoon" if letter in "نم" else "not_maghnoon",
        ))
    return out


# ------------------------------------------------------------------ synthetic speaker
class Speech:
    def __init__(self, ref, spoken: List[str], scale=1.0, pauses: Optional[Set[int]] = None,
                 lead=1.0, junk: str = "", seed=1, noise_p=0.0, trail=1.5, stop_after: Optional[int] = None,
                 pause_s: float = 0.6):
        rnd = random.Random(seed)
        self.pauses = pauses or set()
        t = lead
        self.chars: List[Tuple[float, str, int]] = []
        self.spans: List[Tuple[float, float]] = []
        segs = []
        if junk:
            d = 0.075 * len(junk) * scale
            for i, ch in enumerate(junk):
                self.chars.append((t + (i + 0.5) / len(junk) * d, ch, -1))
            segs.append((t, t + d)); t += d + 0.4
        n = len(spoken) if stop_after is None else stop_after + 1
        for k in range(n):
            s = spoken[k]
            d = max(0.25, 0.085 * len(s) * scale)
            for i, ch in enumerate(s):
                self.chars.append((t + (i + 0.5) / max(1, len(s)) * d, ch, k))
            self.spans.append((t, t + d)); segs.append((t, t + d))
            t += d
            t += (pause_s if k in self.pauses else rnd.uniform(0.01, 0.05))
        self.total = t + trail
        x = np.zeros(int(self.total * M.SR), dtype=np.float32)
        x += (np.random.default_rng(seed).standard_normal(x.size) * 0.0008).astype(np.float32)
        rng = np.random.default_rng(seed + 1)
        for (a, b) in segs:
            i0, i1 = int(a * M.SR), int(b * M.SR)
            tt = np.arange(i1 - i0) / M.SR
            env = 0.10 * (0.7 + 0.3 * np.sin(2 * np.pi * 4.0 * tt))
            x[i0:i1] += (rng.standard_normal(i1 - i0) * env).astype(np.float32)
        self.audio = x
        self.noise_p = noise_p
        self.rnd = random.Random(seed + 7)

    def true_word_at(self, t: float) -> int:
        k = -1
        for i, (a, b) in enumerate(self.spans):
            if t >= a:
                k = i
        return k

    def recognizer(self, engine_ref):
        def rec(wave):
            a, b = engine_ref["engine"].last_window
            sel = [(ch) for (tm, ch, _w) in self.chars if a / M.SR <= tm < b / M.SR]
            text, probs = [], []
            for ch in sel:
                if self.noise_p and self.rnd.random() < self.noise_p:
                    text.append(self.rnd.choice("بتثجحخدذرزسشصضطظعغفقكلمنهوي")); probs.append(0.45)
                else:
                    text.append(ch); probs.append(0.97)
            return M.Recognition("".join(text), probs, None)
        return rec


def run_stream(ref, speech: Speech, preset="balanced", rtf=0.35, stop_at: Optional[float] = None, verbose=False):
    holder: Dict = {}
    rec = speech.recognizer(holder)
    eng = M.SessionEngine(ref, rec, M.get_preset(preset))
    holder["engine"] = eng
    t, hop = 0.0, 0.1
    busy_until = 0.0
    pending: List[Tuple[float, dict]] = []
    log = []
    end = speech.total if stop_at is None else stop_at
    while t < end and not eng.complete:
        t2 = min(end, t + hop)
        eng.add_audio(speech.audio[int(t * M.SR): int(t2 * M.SR)])
        t = t2
        while pending and pending[0][0] <= t:
            tm, msg = pending.pop(0)
            log.append((tm, msg))
        if t >= busy_until and eng.ready_to_run():
            msg = eng.step()
            a, b = eng.last_window
            compute = rtf * (b - a) / M.SR + 0.05
            busy_until = t + compute
            if msg:
                pending.append((busy_until, msg))
    for tm, msg in pending:
        log.append((tm, msg))
    final = eng.flush()
    return eng, log, final, speech


def lag_stats(log, speech):
    lags = []
    for tm, msg in log:
        tw = speech.true_word_at(tm)
        if msg["tracked"] >= 0 and tw >= 0:
            lags.append(tw - msg["tracked"])
    return lags


def build_ref():
    return M.build_ref_aya(2, 255, FakeLib())


# ------------------------------------------------------------------ mirror of the Android silence gate
def client_trim(speech: "Speech", hangover=5, preroll=2):
    """Same algorithm as RecitationSession.captureLoop: speech + up to 500 ms of silence are kept, longer
    silence is dropped except the last 200 ms before the next word (pre-roll). Returns a remapped Speech."""
    import copy
    chunk = M.SR // 10
    x = speech.audio
    n = len(x) // chunk
    noise, silent, pre, kept = 0.004, 0, [], []
    for i in range(n):
        seg = x[i * chunk:(i + 1) * chunk].astype("float64")
        rms = float((seg ** 2).mean() ** 0.5)
        if rms < noise * 1.6 + 0.002:
            noise = 0.95 * noise + 0.05 * rms
        voiced = rms >= min(max(noise * 3, 0.006), 0.02)
        if voiced:
            kept.extend(pre); pre = []; kept.append(i); silent = 0
        else:
            silent += 1
            if silent <= hangover: kept.append(i)
            else:
                pre.append(i)
                if len(pre) > preroll: pre.pop(0)
    kept = sorted(set(kept))
    new_index = {old: new for new, old in enumerate(kept)}
    def remap(t):
        c = int(t * 10)
        while c not in new_index and c < n: c += 1          # snap into the next kept chunk
        return (new_index.get(c, len(kept)) + (t * 10 - int(t * 10))) / 10.0
    out = copy.copy(speech)
    out.audio = np.concatenate([x[i * chunk:(i + 1) * chunk] for i in kept]) if kept else x[:0]
    out.chars = [(remap(tm), ch, w) for (tm, ch, w) in speech.chars]
    out.spans = [(remap(a), remap(b)) for (a, b) in speech.spans]
    out.total = len(out.audio) / M.SR
    return out
