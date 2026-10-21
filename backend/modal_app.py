"""
Rafiq v14 — Real-time Quran recitation corrector (Modal.com backend, CPU-friendly).

Engine: `quran-muaalem` (obadx/muaalem-model-v3_2) + `quran-transcript` (QPS phonemes).

WHY v14 IS DIFFERENT (root-cause fixes)
---------------------------------------
v13 re-ran the 600M-parameter model on ALL the audio since the start every second (cost grows
with the recitation length -> lag, then freeze) and blocked the WebSocket loop while doing so.
v14 uses a BOUNDED SLIDING WINDOW:

  * Only the not-yet-judged audio (plus ~1 word of context) is sent to the model, so the cost per
    update is constant (~2-5 seconds of audio) no matter how long the recitation is.
  * Inference runs in a separate worker task; audio keeps flowing while the model works.
  * Words are judged ONCE, when the reciter has clearly moved past them (or paused), never on
    half-heard words -> no flicker and far fewer false alarms.
  * Phoneme classes are derived structurally (Unicode marks / repeated letters) instead of from
    guessed QPS symbols; unknown symbols can only produce a soft "tajweed" note, never a red error.
  * Every word is checked against BOTH its continuation (wasl) and stop (waqf) form.

WebSocket protocol  (`/ws/live-recitation?sura_idx=2&aya_idx=255&strictness=balanced`)
  client -> server : binary frames = 16 kHz mono int16 LE PCM;  text {"type":"stop"} / {"type":"reset"}
  server -> client : {"type":"ready"|"progress"|"final"|"error", ...}   (see SessionEngine)

Other endpoints: GET /health, GET /debug/last, GET /debug/ref, POST /analyze (offline test).
"""

import asyncio
import copy
import gc
import io
import json
import math
import os
import re
import threading
import time
import traceback
import unicodedata
import wave
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Thread pools must be pinned BEFORE numpy / torch are imported.
NUM_THREADS = int(os.environ.get("RAFIQ_THREADS", "4"))   # Modal cores / local CPU threads
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

try:  # allows importing the pure-python engine in local tests without Modal installed
    import modal
except ImportError:  # pragma: no cover
    modal = None

# ==============================================================================
# Deployment switches (the only things you may want to change)
# ==============================================================================
USE_MEMORY_SNAPSHOT = True   # fast cold starts (Modal restores a snapshot of the loaded model); False = old behaviour
USE_GPU = False          # set True ONLY after you add a card to Modal (T4 GPU = ~10x faster)
PRECISION = "auto"       # "auto": int8 only if fp32 is far too slow (rtf > AUTO_INT8_ABOVE_RTF) | "fp32" | "int8"
AUTO_INT8_ABOVE_RTF = 0.60
MODEL_ID = "obadx/muaalem-model-v3_2"
APP_NAME = "rafiq-quran-muaalem-backend"
DEFAULT_SURA, DEFAULT_AYA = 2, 255

# ==============================================================================
# Streaming-engine constants
# ==============================================================================
SR = 16000
FRAME = 320                       # 20 ms analysis frame
MIN_NEW_AUDIO_S = 0.45            # run the model when at least this much new audio arrived
MIN_VOICED_S = 0.35               # ...and the window contains at least this much speech
MAX_WINDOW_S = 10.0               # hard cap on audio sent to the model per update
MAX_BACKLOG_S = 240.0             # backlog (cold-start replay, bad network) is judged in order; skipped only beyond this
TAIL_SILENCE_S = 0.35             # silence at the end of the audio = the reciter paused (waqf)
CTX_WORDS = 2                     # already-judged words kept in the reference as free context
CTX_MARGIN_S = 0.20               # the next window starts this much before its context word
REGION_AHEAD_WORDS = 26           # how far ahead of the cursor the aligner may look
MAX_SESSION_S = 360               # memory guard
QUALITY_MAX_COST = 0.62           # window rejected when alignment cost per reference char > this
MIN_MATCHED_LETTERS = 3           # ...or when fewer letters than this were matched
WORD_REACH_COV = 0.40             # share of a word's letters that must match to count as "reached"
WORD_SKIP_COV = 0.25              # below this a word that was passed over counts as skipped
LAST_WORD_SETTLE_COV = 0.60       # last word is judged on a pause only if this much was matched
UNCLEAR_ISSUES = 3                # >= this many letter issues in one word -> one "unclear word" note
DEBUG_KEEP_STEPS = 80

SIFAT_LEVELS: Tuple[str, ...] = (
    "ghonna", "qalqla", "tafkheem_or_taqeeq", "hams_or_jahr", "shidda_or_rakhawa",
    "itbaq", "safeer", "tikraar", "tafashie", "istitala",
)


@dataclass(frozen=True)
class Preset:
    """Strictness profile. 'balanced' is the default.
    NOTE: CTC posteriors of emitted tokens are ~0.99-1.0 for almost everything (see live log), so the
    confidence gates only remove genuinely uncertain emissions; the tolerance rules do the real work."""
    name: str
    conf_letter: float          # min model confidence to report a letter / tajweed issue
    conf_vowel: float           # min model confidence to report a vowel (tashkeel) issue
    conf_sifat: float           # min confidence to report a qalqala / tafkheem / ghunna mismatch
    short_slack: int            # madd may be this many harakat shorter than the lower bound
    long_slack: int             # madd may be this many harakat longer than the upper bound
    flag_missing_vowel: bool
    flag_extra_vowel: bool
    flag_extra_letter: bool
    flag_extra_shaddah: bool
    flag_waqf_no_pause: bool    # word-final vowel dropped although the reciter did not stop
    sifat_levels: Tuple[str, ...]


PRESETS: Dict[str, Preset] = {
    "relaxed": Preset("relaxed", conf_letter=0.85, conf_vowel=0.85, conf_sifat=0.92,
                      short_slack=1, long_slack=2, flag_missing_vowel=False, flag_extra_vowel=False,
                      flag_extra_letter=False, flag_extra_shaddah=False, flag_waqf_no_pause=False,
                      sifat_levels=("qalqla", "ghonna")),
    "balanced": Preset("balanced", conf_letter=0.70, conf_vowel=0.70, conf_sifat=0.82,
                       short_slack=0, long_slack=1, flag_missing_vowel=True, flag_extra_vowel=True,
                       flag_extra_letter=False, flag_extra_shaddah=False, flag_waqf_no_pause=False,
                       sifat_levels=("qalqla", "ghonna", "tafkheem_or_taqeeq")),
    "strict": Preset("strict", conf_letter=0.50, conf_vowel=0.50, conf_sifat=0.65,
                     short_slack=0, long_slack=0, flag_missing_vowel=True, flag_extra_vowel=True,
                     flag_extra_letter=True, flag_extra_shaddah=True, flag_waqf_no_pause=True,
                     sifat_levels=("qalqla", "ghonna", "tafkheem_or_taqeeq")),
}


def get_preset(name: Optional[str]) -> Preset:
    return PRESETS.get((name or "balanced").strip().lower(), PRESETS["balanced"])


# ==============================================================================
# Character classes (structural, NOT based on guessed QPS symbols)
# ==============================================================================
FATHA, DAMMA, KASRA, SHADDA = "\u064e", "\u064f", "\u0650", "\u0651"
CORE_VOWELS: Dict[str, str] = {FATHA: "الفتحة", DAMMA: "الضمة", KASRA: "الكسرة"}
VOWEL_HOW: Dict[str, str] = {
    FATHA: "افتح فمك قليلًا وانطق الحرف بصوت «أَ».",
    DAMMA: "اضمم شفتيك مستديرتين وانطق الحرف بصوت «أُ».",
    KASRA: "اخفض فكّك وافتح شفتيك للجانبين وانطق الحرف بصوت «إِ».",
}
BASIC_LETTERS = set("ءابتثجحخدذرزسشصضطظعغفقكلمنهويىةأإؤئآ")
MADD_LETTERS = set("اويىۥۦ")
ALWAYS_MADD = set("اۥۦ")

LETTER_NAMES: Dict[str, str] = {
    "ء": "الهمزة", "ا": "الألف", "ب": "الباء", "ت": "التاء", "ث": "الثاء", "ج": "الجيم",
    "ح": "الحاء", "خ": "الخاء", "د": "الدال", "ذ": "الذال", "ر": "الراء", "ز": "الزاي",
    "س": "السين", "ش": "الشين", "ص": "الصاد", "ض": "الضاد", "ط": "الطاء", "ظ": "الظاء",
    "ع": "العين", "غ": "الغين", "ف": "الفاء", "ق": "القاف", "ك": "الكاف", "ل": "اللام",
    "م": "الميم", "ن": "النون", "ه": "الهاء", "و": "الواو", "ي": "الياء", "ى": "الياء",
    "ۥ": "الواو", "ۦ": "الياء", "ة": "التاء المربوطة",
}
MAKHRAJ: Dict[str, str] = {
    "ء": "يخرج من أقصى الحلق", "ه": "يخرج من أقصى الحلق",
    "ع": "يخرج من وسط الحلق", "ح": "يخرج من وسط الحلق",
    "غ": "يخرج من أدنى الحلق", "خ": "يخرج من أدنى الحلق",
    "ق": "يخرج من أقصى اللسان مع الحنك الأعلى", "ك": "يخرج من أقصى اللسان أسفل مخرج القاف قليلًا",
    "ج": "يخرج من وسط اللسان مع الحنك الأعلى", "ش": "يخرج من وسط اللسان مع الحنك الأعلى",
    "ي": "يخرج من وسط اللسان مع الحنك الأعلى",
    "ض": "يخرج من حافة اللسان مع الأضراس العليا",
    "ل": "يخرج من حافة اللسان مع اللثة العليا", "ن": "يخرج من طرف اللسان مع اللثة العليا",
    "ر": "يخرج من طرف اللسان مع تكرار خفيف",
    "ط": "يخرج من طرف اللسان مع أصول الثنايا العليا", "د": "يخرج من طرف اللسان مع أصول الثنايا العليا",
    "ت": "يخرج من طرف اللسان مع أصول الثنايا العليا",
    "ص": "يخرج من طرف اللسان مع أطراف الثنايا السفلى (حرف صفير)",
    "س": "يخرج من طرف اللسان مع أطراف الثنايا السفلى (حرف صفير)",
    "ز": "يخرج من طرف اللسان مع أطراف الثنايا السفلى (حرف صفير)",
    "ظ": "يخرج من طرف اللسان مع أطراف الثنايا العليا", "ذ": "يخرج من طرف اللسان مع أطراف الثنايا العليا",
    "ث": "يخرج من طرف اللسان مع أطراف الثنايا العليا",
    "ف": "يخرج من باطن الشفة السفلى مع أطراف الثنايا العليا",
    "ب": "يخرج من انطباق الشفتين", "م": "يخرج من انطباق الشفتين مع غنة",
    "و": "يخرج من استدارة الشفتين",
}


GHUNNA_LETTERS = set("نمويى")          # only these can carry ghunna (noon, meem, idgham bi-ghunna)
TAFKHEEM_LETTERS = set("رلصضطظقغخ")      # letters whose heaviness is a real tajweed question


def is_mark(c: str) -> bool:
    return unicodedata.category(c) == "Mn"


def lname(c: str) -> str:
    return LETTER_NAMES.get(c, "")


def clean_phonemes(text: str) -> str:
    """Drops whitespace and any <special-token> the tokenizer may emit."""
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", "", str(text))
    return "".join(ch for ch in text if not ch.isspace())


def letters_only(s: str) -> List[str]:
    return [c for c in s if not is_mark(c)]


def collapse_runs(chars: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in chars:
        if not out or out[-1] != c:
            out.append(c)
    return out


def split_runs(s: str) -> List[Tuple[str, int, int]]:
    runs: List[Tuple[str, int, int]] = []
    i, n = 0, len(s)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        runs.append((s[i], i, j))
        i = j
    return runs


def prev_letter(s: str, idx: int) -> str:
    """Nearest letter (non-mark) at or before idx — the consonant a vowel mark belongs to."""
    k = idx
    while k >= 0:
        if not is_mark(s[k]):
            return s[k]
        k -= 1
    return ""


# ==============================================================================
# Arabic notes (what the user sees when tapping a coloured word)
# ==============================================================================
def hint(letter: str) -> str:
    h = MAKHRAJ.get(letter)
    return f" ({h})" if h else ""


def _disp(c: str) -> str:
    """How a phoneme is shown to the user (QPS madd marks are shown by name, never as raw symbols)."""
    if c in ALWAYS_MADD:
        return {"ا": "مدّ الألف", "ۥ": "مدّ الواو", "ۦ": "مدّ الياء"}[c]
    return f"«{c}»"


def note_replace(e: str, p: str) -> str:
    return f"نطقتَ {_disp(p)} بدل {_disp(e)}. الصواب أن تنطق {_disp(e)}{hint(e)}."


def note_missing_letter(e: str) -> str:
    return f"لم يُسمع حرف «{e}» في هذه الكلمة. انطقه بوضوح{hint(e)}."


def note_extra_letter(p: str) -> str:
    return f"زدتَ حرفًا غير موجود في الكلمة (سُمع «{p}»). اقرأ الكلمة كما هي دون زيادة."


NOTE_SKIPPED = "لم تقرأ هذه الكلمة (تجاوزتها). ارجع واقرأها قبل أن تنتقل لما بعدها."
NOTE_UNCLEAR = "لم تُنطق هذه الكلمة بوضوح كافٍ، أو فيها أكثر من خطأ في الحروف. أعد قراءتها ببطء ووضوح."
NOTE_RULE = ("لم يُطبَّق حكم التجويد المطلوب في هذا الموضع (كالإخفاء أو الإدغام أو الغنة أو القلب). "
             "راجع حكم هذه الكلمة ثم أعد قراءتها.")


def _on_letter(letter: str) -> str:
    return f" على حرف «{letter}»" if letter in BASIC_LETTERS else ""


def note_vowel_swap(letter: str, e: str, p: str) -> str:
    return f"المطلوب{_on_letter(letter)} {CORE_VOWELS[e]} ولكنك نطقتَ {CORE_VOWELS[p]}. {VOWEL_HOW[e]}"


def note_vowel_missing(letter: str, e: str) -> str:
    return f"لم تنطق {CORE_VOWELS[e]}{_on_letter(letter)}. انطق الحرف بحركته كاملة. {VOWEL_HOW[e]}"


def note_vowel_extra(letter: str, p: str) -> str:
    return f"نطقتَ {CORE_VOWELS[p]}{_on_letter(letter)} وهو ساكن في الكلمة. اقرأه ساكنًا دون حركة."


def madd_name(c: str) -> str:
    return {"ا": "الألف", "و": "الواو", "ۥ": "الواو", "ي": "الياء", "ى": "الياء", "ۦ": "الياء"}.get(c, "حرف المد")


MADD_VOWEL = {"ا": "الفتحة", "ۥ": "الضمة", "ۦ": "الكسرة"}


def note_madd_vowel_swap(c: str, p: str) -> str:
    return (f"حرف المدّ: المطلوب {_disp(c)} (من {MADD_VOWEL.get(c, '')}) ولكنك مددتَ {_disp(p)} "
            f"(من {MADD_VOWEL.get(p, '')}). انتبه لحركة الحرف قبل المدّ.")


def note_waqf_no_pause(vowel: str) -> str:
    name = CORE_VOWELS.get(vowel, "حركة")
    how = VOWEL_HOW.get(vowel, "")
    return (f"آخر هذه الكلمة: المطلوب {name} (أو تنوين) لأنك تصل القراءة بما بعدها، "
            f"لكنك سكّنتَ آخرها دون أن تقف. التسكين يكون عند الوقف فقط. {how}").strip()


def note_madd_short(c: str, need_text: str) -> str:
    return (f"المدّ ({madd_name(c)}): قصّرتَ في مدّه. حقّه {need_text}؛ "
            "مُدّ الصوت بهدوء، ومقدار الحركة كقبض الإصبع وبسطه.")


def note_madd_dropped(c: str, need_text: str) -> str:
    return f"المدّ ({madd_name(c)}): لم تمدّ الصوت إطلاقًا. حقّه {need_text}؛ مُدّه بهدوء وبمقدار ثابت."


def note_madd_long(c: str, e: int) -> str:
    return f"المدّ ({madd_name(c)}): أطلتَ فيه أكثر من اللازم. اقتصر على نحو {e} حركات."


def note_shaddah_dropped(letter: str) -> str:
    extra = " مع الغنة" if letter in "نم" else ""
    return (f"التشديد: حرف «{letter}» مشدَّد ولم تُشدّده. اضغط عليه لحظة ثم أطلقه{extra}.")


def note_shaddah_extra(letter: str) -> str:
    return f"شدّدتَ حرف «{letter}» وهو غير مشدَّد في هذا الموضع. انطقه مخفّفًا."


def note_qalqla(letter: str) -> str:
    where = f" الحرف «{letter}»" if letter in BASIC_LETTERS else " هذا الحرف"
    return ("القلقلة:" + where + " ساكن أو موقوف عليه ويلزمه قلقلة؛ أي نبرة خفيفة مرتدّة تُسمع عند "
            "إطلاقه (حروفها: ق ط ب ج د).")


def note_ghonna(letter: str, expected_ghonna: bool) -> str:
    l = f"«{letter}»" if letter in BASIC_LETTERS else "هذا الحرف"
    if expected_ghonna:
        return f"الغنة: ينقص صوت الغنة في {l}. أخرج الغنة من الخيشوم بمقدار حركتين."
    return f"أظهرتَ غنة في {l} وليس فيه غنة هنا. انطقه دون غنة."


def note_tafkheem(letter: str, expected_heavy: bool) -> str:
    l = f"«{letter}»" if letter in BASIC_LETTERS else "هذا الحرف"
    if expected_heavy:
        return f"التفخيم: حقّ {l} التفخيم هنا؛ فخّمه بتغليظ الصوت مع رفع مؤخرة اللسان نحو الحنك."
    return f"الترقيق: حقّ {l} الترقيق هنا؛ رقّقه بإخراج الصوت رقيقًا مع خفض مؤخرة اللسان."


def note_sifa_generic(letter: str) -> str:
    l = f"«{letter}»" if letter in BASIC_LETTERS else "هذا الحرف"
    return f"صفة {l} لم تظهر كما ينبغي. استمع للقارئ ثم أعد قراءة الكلمة."


# ==============================================================================
# Phoneme aligner (edit distance with run-aware gap costs)
# ==============================================================================
@lru_cache(maxsize=16384)
def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    ma, mb = is_mark(a), is_mark(b)
    if ma and mb:
        return 0.6 if (a in CORE_VOWELS and b in CORE_VOWELS) else 0.8
    if ma != mb:
        return 1.7                      # never turn a vowel mark into a letter
    if a in MADD_LETTERS and b in MADD_LETTERS:
        return 0.9
    return 1.0


def gap_costs(s: str) -> List[float]:
    """Per-character gap cost; a repeated character (madd / shaddah length) is cheap to add or drop."""
    out: List[float] = []
    prev = ""
    for c in s:
        if is_mark(c):
            out.append(0.7)
        elif c == prev:
            out.append(0.25)
        else:
            out.append(1.0)
        prev = c
    return out


@dataclass
class Alignment:
    pairs: List[Tuple[int, int]]       # (ref_idx, pred_idx); -1 = gap on that side
    i_end: int                         # reference chars consumed
    j_end: int                         # predicted chars consumed
    cost: float


def align(
    ref: str,
    pred: str,
    *,
    lead_free: int = 0,
    lead_pen: float = 0.0,
    pred_lead_free: int = 0,
    pred_lead_pen: float = 0.08,
    tail_free_ref: bool = True,
    tail_ins_chars: int = 0,
    tail_ins_pen: float = 0.08,
) -> Alignment:
    """
    Anchored-start alignment.
      lead_free / lead_pen   : the first `lead_free` reference chars may be skipped at a tiny cost
                               (the context words already judged, or a few words after an overload)
      pred_lead_free         : the first predicted chars may be inserted cheaply (opening du'a, basmala)
      tail_free_ref          : the reference tail is free (the reciter has not reached it yet)
      tail_ins_chars         : when the reference is fully consumed, up to this many trailing
                               predicted chars may be ignored cheaply (e.g. "sadaqa Allah...")
    """
    M, L = len(ref), len(pred)
    ins = gap_costs(pred)
    dele = gap_costs(ref)
    dp = [[0.0] * (L + 1) for _ in range(M + 1)]
    bt = [[0] * (L + 1) for _ in range(M + 1)]
    for j in range(1, L + 1):
        dp[0][j] = dp[0][j - 1] + (pred_lead_pen if j <= pred_lead_free else ins[j - 1])
        bt[0][j] = 2
    for i in range(1, M + 1):
        dp[i][0] = dp[i - 1][0] + (lead_pen if i <= lead_free else dele[i - 1])
        bt[i][0] = 1
        rc = ref[i - 1]
        di = dele[i - 1]
        prev, cur, btr = dp[i - 1], dp[i], bt[i]
        for j in range(1, L + 1):
            d = prev[j - 1] + sub_cost(rc, pred[j - 1])
            u = prev[j] + di
            v = cur[j - 1] + ins[j - 1]
            if d <= u and d <= v:
                cur[j] = d
                btr[j] = 0
            elif u <= v:
                cur[j] = u
                btr[j] = 1
            else:
                cur[j] = v
                btr[j] = 2

    best_c, best_i, best_j = float("inf"), M, L
    if tail_free_ref:
        for i in range(M + 1):
            c = dp[i][L]
            if c < best_c - 1e-9:
                best_c, best_i, best_j = c, i, L
    else:
        best_c, best_i, best_j = dp[M][L], M, L
    if tail_ins_chars > 0:
        for j in range(max(0, L - tail_ins_chars), L):
            c = dp[M][j] + tail_ins_pen * (L - j)
            if c < best_c - 1e-9:
                best_c, best_i, best_j = c, M, j

    pairs: List[Tuple[int, int]] = []
    i, j = best_i, best_j
    while i > 0 and j > 0:
        t = bt[i][j]
        if t == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif t == 1:
            pairs.append((i - 1, -1))
            i -= 1
        else:
            pairs.append((-1, j - 1))
            j -= 1
    while j > 0:
        pairs.append((-1, j - 1))
        j -= 1
    while i > 0:
        if i > lead_free:
            pairs.append((i - 1, -1))
        i -= 1
    pairs.reverse()
    return Alignment(pairs, best_i, best_j, float(best_c))


# ==============================================================================
# Sifat helpers (per-letter properties predicted by the model)
# ==============================================================================
def _unwrap_unit(v: Any) -> Tuple[Optional[str], Optional[float]]:
    if v is None:
        return None, None
    prob = None
    if isinstance(v, dict):
        label, prob = v.get("text"), v.get("prob", v.get("probs"))
    elif hasattr(v, "text"):
        label = getattr(v, "text", None)
        prob = getattr(v, "prob", None)
        if prob is None:
            prob = getattr(v, "probs", None)
    else:
        label = getattr(v, "value", v)
    if prob is not None:
        try:
            if hasattr(prob, "detach"):
                prob = prob.detach().float().cpu().numpy()
            arr = np.asarray(prob, dtype=np.float64).reshape(-1)
            prob = float(arr.max()) if arr.size else None
        except Exception:
            prob = None
    return (str(label) if label is not None else None), prob


def convert_sifat(items: Any) -> Optional[List[Dict[str, Any]]]:
    """-> [{'g': group_text, 'L': {level: (label, prob)}}] or None when unavailable."""
    if not items:
        return None
    out: List[Dict[str, Any]] = []
    try:
        for it in items:
            g = ""
            for name in ("phonemes_group", "phonemes", "group"):
                val = it.get(name) if isinstance(it, dict) else getattr(it, name, None)
                if isinstance(val, str) and val:
                    g = val
                    break
            levels: Dict[str, Tuple[Optional[str], Optional[float]]] = {}
            for lv in SIFAT_LEVELS:
                raw = it.get(lv) if isinstance(it, dict) else getattr(it, lv, None)
                levels[lv] = _unwrap_unit(raw)
            out.append({"g": clean_phonemes(g), "L": levels})
    except Exception:
        return None
    return out or None


def group_index_map(groups: Optional[List[Dict[str, Any]]], text: str) -> Optional[List[int]]:
    """char index -> sifat group index; only if the groups tile the phoneme string exactly."""
    if not groups:
        return None
    if "".join(g["g"] for g in groups) != text:
        return None
    m: List[int] = []
    for gi, g in enumerate(groups):
        m.extend([gi] * len(g["g"]))
    return m


def tolerant_group_map(groups: Optional[List[Dict[str, Any]]], text: str) -> Optional[List[int]]:
    """Like group_index_map, but survives small differences (e.g. a doubled vowel mark at a window
    edge) by aligning the concatenated group text to the phoneme string."""
    if not groups or not text:
        return None
    joined = "".join(g["g"] for g in groups)
    if not joined:
        return None
    al = align(joined, text, tail_free_ref=False)
    if al.cost > 0.15 * max(len(joined), len(text)):
        return None
    owner: List[int] = []
    for gi, g in enumerate(groups):
        owner.extend([gi] * len(g["g"]))
    out = [-1] * len(text)
    for ri, pj in al.pairs:
        if ri >= 0 and pj >= 0:
            out[pj] = owner[ri]
    last = 0
    for j in range(len(out)):                  # unmatched chars inherit the previous group
        if out[j] < 0:
            out[j] = last
        else:
            last = out[j]
    return out


# ==============================================================================
# Reference (one ayah): words, wasl/waqf phoneme forms, expected sifat
# ==============================================================================
_BASE_RE = re.compile("[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]")


def split_words(uthmani: str) -> Tuple[List[str], List[str], List[int]]:
    """Splits on spaces and drops stop-sign-only tokens (they are not words)."""
    raw = uthmani.split()
    words: List[str] = []
    idx: List[int] = []
    for i, t in enumerate(raw):
        if _BASE_RE.search(t):
            words.append(t)
            idx.append(i)
    return raw, words, idx


def split_groups(spaced_phonemes: str) -> List[str]:
    return [clean_phonemes(g) for g in str(spaced_phonemes).split(" ") if clean_phonemes(g)]


def proportional_groups(words: List[str], phonemes: str) -> List[str]:
    weights = [max(1, len(_BASE_RE.findall(w))) for w in words]
    total = float(sum(weights))
    n = len(phonemes)
    out: List[str] = []
    acc, prev = 0.0, 0
    for k, w in enumerate(weights):
        acc += w
        cut = n if k == len(weights) - 1 else int(round(acc / total * n))
        cut = max(prev, min(n, cut))
        out.append(phonemes[prev:cut])
        prev = cut
    return out


class RefAya:
    def __init__(self, sura: int, aya: int, words: List[str], wasl: List[str],
                 waqf: List[Optional[str]], approx: bool, full_out: Any,
                 exp_groups: Optional[List[Dict[str, Any]]], uthmani: str = "",
                 info: Optional[Dict[str, Any]] = None,
                 word_exact: Optional[List[bool]] = None):
        self.sura, self.aya = sura, aya
        self.info: Dict[str, Any] = info or {}
        self.word_approx: List[bool] = [not e for e in (word_exact or [not approx] * len(words))]
        approx = any(self.word_approx)
        self.words = words
        self.wasl = wasl
        self.waqf = waqf
        self.approx = approx
        self.full_out = full_out
        self.uthmani = uthmani
        self.n = len(words)
        self.full_ph = "".join(wasl)
        self.starts: List[int] = [0]
        self.word_of_char: List[int] = []
        for k, f in enumerate(wasl):
            self.starts.append(self.starts[-1] + len(f))
            self.word_of_char.extend([k] * len(f))
        self.exp_groups = exp_groups
        self.group_of_char = group_index_map(exp_groups, self.full_ph)
        self.core: List[List[str]] = [collapse_runs(letters_only(f)) for f in wasl]

    def forms(self, k: int) -> List[str]:
        f = [self.wasl[k]]
        if self.waqf[k] and self.waqf[k] != self.wasl[k]:
            f.append(self.waqf[k])
        return f

    def describe(self) -> Dict[str, Any]:
        return {
            "sura": self.sura, "aya": self.aya, "n_words": self.n, "approx_segmentation": self.approx,
            "sifat_available": self.group_of_char is not None, "info": self.info,
            "words": [{"i": k, "text": self.words[k], "wasl": self.wasl[k], "waqf": self.waqf[k]}
                      for k in range(self.n)],
        }


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def refine_bounds(full_ph: str, bounds: List[int]) -> List[int]:
    """A word cannot start with a vowel mark or a silah/madd letter: those stay with the previous word
    (the live log showed the previous word's final vowel leaking into the next word's expected form)."""
    out = list(bounds)
    n = len(full_ph)
    for k in range(1, len(out) - 1):
        b = out[k]
        while b < n - 1 and b < out[k + 1] - 1 and (is_mark(full_ph[b]) or full_ph[b] in "ۥۦ"):
            b += 1
        out[k] = b
    return out


def _segment_by_isolated_words(lib: Any, words: List[str], full_ph: str,
                               info: Dict[str, Any]) -> Optional[List[int]]:
    """
    Word boundaries that do not rely on the library's spaced output: phonetize every word on its own,
    align the concatenation against the full phoneme string, read the boundaries off the alignment.
    """
    iso: List[str] = []
    for w in words:
        try:
            iso.append(clean_phonemes(lib.phonetize(w, True).phonemes))
        except Exception as exc:
            info["iso_error"] = repr(exc)[:160]
            return None
    if any(not s for s in iso):
        info["iso_error"] = "empty isolated phonemes"
        return None
    S = "".join(iso)
    cum = [0]
    for s in iso:
        cum.append(cum[-1] + len(s))
    al = align(S, full_ph, tail_free_ref=False)
    s2f = {ri: pj for ri, pj in al.pairs if ri >= 0 and pj >= 0}
    info["iso_align_cost_per_char"] = round(al.cost / max(1, len(full_ph)), 4)
    bounds = [0]
    for k in range(1, len(words)):
        j = None
        for r in range(cum[k], len(S)):
            if r in s2f:
                j = s2f[r]
                break
        if j is None:
            return None
        bounds.append(j)
    bounds.append(len(full_ph))
    if any(bounds[i + 1] <= bounds[i] for i in range(len(bounds) - 1)):
        info["iso_error"] = "non-monotonic boundaries"
        return None
    return refine_bounds(full_ph, bounds)


def _snap_to_library_groups(bounds: List[int], group_starts: List[int],
                            tol: int = 3) -> Tuple[List[int], List[bool]]:
    """Replace estimated boundaries by the library's own (exact) group boundaries where one is close."""
    out = list(bounds)
    exact = [False] * len(bounds)
    exact[0] = exact[-1] = True
    used = set()
    for k in range(1, len(bounds) - 1):
        cands = [g for g in group_starts if abs(g - bounds[k]) <= tol and g not in used]
        if cands:
            g = min(cands, key=lambda x: abs(x - bounds[k]))
            if out[k - 1] < g:
                out[k], exact[k] = g, True
                used.add(g)
    for k in range(1, len(out)):                 # keep strictly increasing
        if out[k] <= out[k - 1]:
            out[k], exact[k] = out[k - 1] + 1, False
    return out, exact


def build_ref_aya(sura: int, aya: int, lib: Any, with_waqf: bool = True) -> RefAya:
    """
    `lib` must provide:  get_uthmani(sura, aya) -> str
                         phonetize(text, remove_spaces) -> object with .phonemes (str) [and .sifat]

    Word segmentation ladder (first that validates wins; the winner is recorded in `info`):
      spaced-exact    the library's own spaced output matches the no-space output exactly
      spaced-adopted  same number of words, <= 2 % different: use the spaced words as the reference
      aligned         isolated-word phonetization aligned to the full string (about +-1 char at edges)
      proportional    last resort (flagged approximate)
    """
    uth = lib.get_uthmani(sura, aya)
    raw, words, tok_idx = split_words(uth)
    n = len(words)
    full = lib.phonetize(uth, True)
    F = clean_phonemes(full.phonemes)
    info: Dict[str, Any] = {"n_words": n, "full_len": len(F), "full_head": F[:90]}

    groups: Optional[List[str]] = None
    strategy = ""
    word_exact: Optional[List[bool]] = None
    lib_groups: Optional[List[str]] = None
    try:
        spaced_raw = str(lib.phonetize(uth, False).phonemes)
        g = split_groups(spaced_raw)
        joined = "".join(g)
        info.update({"spaced_head": spaced_raw[:90], "n_groups": len(g), "join_equal": joined == F})
        if joined != F:
            info["first_diff_at"] = _lcp(joined, F)
        else:
            lib_groups = g
        if len(g) == n:
            if joined == F:
                groups, strategy = g, "spaced-exact"
            else:
                diff = align(joined, F, tail_free_ref=False).cost / max(1, len(F))
                info["spaced_diff"] = round(diff, 4)
                if diff <= 0.02:
                    groups, strategy = g, "spaced-adopted"
    except Exception as exc:
        info["spaced_error"] = repr(exc)[:200]
    if groups is None:
        bounds = _segment_by_isolated_words(lib, words, F, info)
        if bounds is not None:
            strategy = "aligned"
            exact = [False] * len(bounds)
            exact[0] = exact[-1] = True
            if lib_groups:                      # merged words in the library output: snap to its exact edges
                gs, acc = [], 0
                for gr in lib_groups[:-1]:
                    acc += len(gr)
                    gs.append(acc)
                bounds, exact = _snap_to_library_groups(bounds, gs)
                info["snapped_boundaries"] = sum(1 for e in exact[1:-1] if e)
                if info["snapped_boundaries"]:
                    strategy = "spaced+aligned"
            groups = [F[bounds[i]:bounds[i + 1]] for i in range(n)]
            word_exact = [exact[i] and exact[i + 1] for i in range(n)]
    if groups is None:
        groups, strategy = proportional_groups(words, F), "proportional"
    info["segmentation"] = strategy
    if word_exact is None:
        word_exact = [strategy in ("spaced-exact", "spaced-adopted")] * n
    approx = not all(word_exact)
    info["inexact_words"] = sum(1 for e in word_exact if not e)

    # ---- waqf (stop) form of every word: tail of the phonemes of "text up to this word"
    waqf: List[Optional[str]] = [None] * n
    if with_waqf and strategy != "proportional":
        starts = [0]
        for gr in groups:
            starts.append(starts[-1] + len(gr))
        ok_cnt = 0
        for k in range(n - 1):
            try:
                P = clean_phonemes(lib.phonetize(" ".join(raw[: tok_idx[k] + 1]), True).phonemes)
            except Exception:
                continue
            start = starts[k]
            if _lcp(P, F) < start - 1:
                continue
            tail = P[start:]
            if 1 <= len(tail) <= len(groups[k]) + 6 and tail != groups[k]:
                waqf[k] = tail
                ok_cnt += 1
        info["waqf_forms"] = ok_cnt

    raw_sifat = getattr(full, "sifat", None)
    exp_groups = convert_sifat(raw_sifat)
    try:
        info["exp_sifat_n"] = len(raw_sifat) if raw_sifat else 0
        if raw_sifat:
            info["exp_sifat_sample"] = repr(raw_sifat[0])[:300]
    except Exception:
        pass
    ref = RefAya(sura, aya, words, groups, waqf, approx, full, exp_groups, uth, info, word_exact)
    info["sifat_tiles_reference"] = ref.group_of_char is not None
    return ref


# ==============================================================================
# Word judge
# ==============================================================================
@dataclass
class Issue:
    t: str                 # "letters" | "tashkeel" | "tajweed"
    k: str                 # kind (machine readable)
    n: str                 # Arabic note
    conf: float
    pos: int = 0           # position inside the expected form (debug / boundary filter)
    exp: str = ""
    pred: str = ""
    focus: str = ""

    def to_client(self) -> Dict[str, Any]:
        d = {"t": self.t, "k": self.k, "n": self.n}
        if self.focus:
            d["f"] = self.focus
        return d

    def to_debug(self) -> Dict[str, Any]:
        return {"t": self.t, "k": self.k, "conf": round(self.conf, 3), "pos": self.pos,
                "exp": self.exp, "pred": self.pred, "note": self.n}


SEVERITY = {"letters": 3, "tashkeel": 2, "tajweed": 1}


def status_of(issues: List[Issue]) -> str:
    if not issues:
        return "ok"
    return max(issues, key=lambda i: SEVERITY[i.t]).t


def _conf(probs: Optional[List[float]], idxs: Sequence[int], default: float = 0.92) -> float:
    if probs is None:
        return default
    vals = [probs[i] for i in idxs if 0 <= i < len(probs)]
    return float(sum(vals) / len(vals)) if vals else default


def _madd_is_run(exp: str, s: int, e: int) -> bool:
    c = exp[s]
    if c in ALWAYS_MADD:
        return True
    if c in "ويى":
        nxt = exp[e] if e < len(exp) else ""
        return nxt not in CORE_VOWELS      # a waw/ya followed by a vowel is a doubled consonant
    return False


def _need_text(lo: int, e: int) -> str:
    if lo >= 4:
        return "أربع حركات على الأقل"
    return "حركتان" if e <= 2 else f"نحو {e} حركات"


def madd_bounds(exp: str, s: int, e: int) -> Tuple[int, int]:
    """
    Accepted number of harakat (lo, hi) for the madd run exp[s:e].  The live model counts agree with the
    moshaf almost exactly (natural = 2, muttasil/munfasil = 4), so the base range is tight and the
    preset slacks (short_slack / long_slack) decide how much extra is tolerated.
    """
    E = e - s
    after = [ch for ch in exp[e:] if not is_mark(ch)]
    n_after = len(after)
    muttasil = bool(after) and after[0] == "ء"
    if E >= 6:
        return 5, 7                              # lazim
    if muttasil and E >= 3:
        return 4, 6                              # muttasil
    if n_after == 0:
        return (3, 7) if E >= 3 else (2, 2)      # munfasil / silah kubra  |  natural, silah sughra
    if n_after == 1:
        return (3, 7) if E >= 3 else (2, 6)      # ... | possible 'aarid madd at a stop (2/4/6)
    return (3, 6) if E >= 3 else (2, 2)          # natural madd inside the word


def judge_word(
    forms: List[str],
    seg: str,
    probs: Optional[List[float]],
    preset: Preset,
    lead: int = 0,
    trail: int = 0,
    approx: bool = False,
    ctx_chars: str = "",
) -> Tuple[List[Issue], Optional[Alignment], str, Dict[str, float]]:
    """
    Compares ONE spoken word (`seg` = its predicted phonemes, with up to `lead`/`trail` chars of
    unassigned neighbours) against its expected form(s). Returns (issues, alignment, chosen_form).
    Only errors that survive the tolerance rules and the confidence gate are returned.
    """
    best: Optional[Tuple[str, Alignment]] = None
    form_costs: Dict[str, float] = {}
    for f in dict.fromkeys(x for x in forms if x):
        al = align(f, seg, tail_free_ref=False, pred_lead_free=lead, tail_ins_chars=trail)
        form_costs[f] = al.cost
        if best is None or al.cost < best[1].cost - 1e-9:
            best = (f, al)
    if best is None:
        return [], None, "", form_costs
    exp, al = best
    uniq = [f for f in dict.fromkeys(x for x in forms if x)]
    if len(uniq) > 1 and exp != uniq[0]:
        # a stop form won on cost, but if the reciter kept a final vowel he was NOT stopping:
        # judge against the continuation form so the vowel mistake is reported properly
        core_end = len(seg) - trail
        letter_idx = [i for i in range(lead, core_end) if not is_mark(seg[i]) and seg[i] not in ALWAYS_MADD]
        if letter_idx and any(seg[i] in CORE_VOWELS for i in range(letter_idx[-1] + 1, core_end)):
            exp = uniq[0]
            al = align(exp, seg, tail_free_ref=False, pred_lead_free=lead, tail_ins_chars=trail)
    M, L = len(exp), len(seg)
    r2p = {ri: pj for ri, pj in al.pairs if ri >= 0 and pj >= 0}
    aligned_p = set(r2p.values())
    aligned_p_to_r = {pj: ri for ri, pj in r2p.items()}

    def left_p(ri: int) -> Optional[int]:
        for k in range(ri - 1, -1, -1):
            if k in r2p:
                return r2p[k]
        return None

    def right_p(ri: int) -> Optional[int]:
        for k in range(ri + 1, M):
            if k in r2p:
                return r2p[k]
        return None

    def del_conf(ri: int) -> float:
        return _conf(probs, [x for x in (left_p(ri), right_p(ri)) if x is not None])

    issues: List[Issue] = []

    # ---------------------------------------------------------------- vowel marks
    for ri, c in enumerate(exp):
        if not is_mark(c) or c not in CORE_VOWELS:
            continue
        owner = prev_letter(exp, ri)
        pj = r2p.get(ri)
        if pj is not None:
            pc = seg[pj]
            if pc != c and pc in CORE_VOWELS:
                issues.append(Issue("tashkeel", "vowel_swap", note_vowel_swap(owner, c, pc),
                                    _conf(probs, [pj]), ri, c, pc, owner))
            continue
        if all(is_mark(x) for x in exp[ri + 1:]):
            continue                      # final vowel: dropped at a stop (waqf) is legitimate
        if preset.flag_missing_vowel:
            issues.append(Issue("tashkeel", "vowel_missing", note_vowel_missing(owner, c),
                                del_conf(ri), ri, c, "", owner))

    # ---------------------------------------------------------------- letters and runs
    for c, s, e in split_runs(exp):
        if is_mark(c):
            continue
        E = e - s
        same = [r2p[ri] for ri in range(s, e) if ri in r2p and seg[r2p[ri]] == c]
        if same:
            a = b = same[0]
            while a - 1 >= 0 and seg[a - 1] == c:
                a -= 1
            while b + 1 < L and seg[b + 1] == c:
                b += 1
            p, conf = b - a + 1, _conf(probs, range(a, b + 1))
        else:
            p, conf = 0, 0.0

        if p == 0:
            subs = [r2p[ri] for ri in range(s, e) if ri in r2p and not is_mark(seg[r2p[ri]])]
            if subs:
                pj = subs[0]
                pc = seg[pj]
                conf = _conf(probs, [pj])
                if c in ALWAYS_MADD and pc in ALWAYS_MADD and c != pc:
                    # a different madd letter (fatha-madd / damma-madd / kasra-madd) = the vowel changed
                    issues.append(Issue("tashkeel", "madd_vowel_swap", note_madd_vowel_swap(c, pc),
                                        conf, s, c, pc, ""))
                elif (c in BASIC_LETTERS or c in ALWAYS_MADD) and (pc in BASIC_LETTERS or pc in ALWAYS_MADD):
                    issues.append(Issue("letters", "replace", note_replace(c, pc), conf, s, c, pc,
                                        c if c in BASIC_LETTERS else ""))
                else:
                    issues.append(Issue("tajweed", "rule", NOTE_RULE, conf, s, c, pc, ""))
                continue
            conf = del_conf(s)
            if _madd_is_run(exp, s, e) and E >= 2:
                lo, _hi = madd_bounds(exp, s, e)
                if 0 < lo - preset.short_slack:
                    issues.append(Issue("tajweed", "madd_dropped", note_madd_dropped(c, _need_text(lo, E)),
                                        conf, s, c * E, "", c))
            elif c in BASIC_LETTERS:
                issues.append(Issue("letters", "missing", note_missing_letter(c), conf, s, c, "", c))
            elif c not in MADD_LETTERS:
                issues.append(Issue("tajweed", "rule", NOTE_RULE, conf, s, c, "", ""))
            continue

        if p == E:
            continue
        sub_in_run = [r2p[ri] for ri in range(s, e)
                      if ri in r2p and seg[r2p[ri]] != c and not is_mark(seg[r2p[ri]])]
        if sub_in_run and c in BASIC_LETTERS and seg[sub_in_run[0]] in BASIC_LETTERS:
            pc = seg[sub_in_run[0]]          # a different letter was spoken inside the run: it wins
            issues.append(Issue("letters", "replace", note_replace(c, pc),
                                _conf(probs, [sub_in_run[0]]), s, c, pc, c))
            continue
        if _madd_is_run(exp, s, e):
            lo, hi = madd_bounds(exp, s, e)
            if p < lo - preset.short_slack:
                issues.append(Issue("tajweed", "madd_short", note_madd_short(c, _need_text(lo, E)),
                                    conf, s, c * E, c * p, c))
            elif p > hi + preset.long_slack:
                issues.append(Issue("tajweed", "madd_long", note_madd_long(c, min(E, hi)),
                                    conf, s, c * E, c * p, c))
        elif c in BASIC_LETTERS:
            if preset.name == "strict":
                min_ok = E
            elif preset.name == "balanced":
                min_ok = 2 if E == 2 else E - 1
            else:
                min_ok = 1
            if E >= 2 and p < min_ok:
                issues.append(Issue("tajweed", "shaddah_dropped", note_shaddah_dropped(c),
                                    conf, s, c * E, c * p, c))
            elif E >= 1 and p > E and preset.flag_extra_shaddah:
                issues.append(Issue("tajweed", "shaddah_extra", note_shaddah_extra(c),
                                    conf, s, c, c * p, c))

    # ---------------------------------------------------------------- extra predicted chars
    first_core, last_core = lead, L - trail
    for pj in range(first_core + 1, last_core - 1):
        if pj in aligned_p:
            continue
        pc = seg[pj]
        if pc in CORE_VOWELS and preset.flag_extra_vowel and (pj - 1) in aligned_p:
            ri_prev = aligned_p_to_r[pj - 1]
            nxt = exp[ri_prev + 1] if ri_prev + 1 < M else ""
            if not is_mark(seg[pj - 1]) and nxt not in CORE_VOWELS:
                owner = seg[pj - 1]
                issues.append(Issue("tashkeel", "vowel_extra", note_vowel_extra(owner, pc),
                                    _conf(probs, [pj]), ri_prev, "", pc, owner))
        elif (preset.flag_extra_letter and pc in BASIC_LETTERS and pc != seg[pj - 1]
              and pj + 1 < L and pc != seg[pj + 1]):
            issues.append(Issue("letters", "extra", note_extra_letter(pc), _conf(probs, [pj]), 0, "", pc, pc))

    # ---------------------------------------------------------------- confidence gate
    kept: List[Issue] = []
    edge_kinds = ("missing", "replace", "extra", "rule", "vowel_missing")
    for it in issues:
        need = preset.conf_vowel if it.t == "tashkeel" else preset.conf_letter
        if it.conf < need:
            continue
        if approx and it.k in edge_kinds and (it.pos < 1 or it.pos >= M - 1) and ctx_chars:
            # boundaries are only +-1 char accurate: an edge issue is ambiguous only when the letter
            # involved also sits at the neighbouring word's adjacent edge (it may belong to that word)
            if it.exp[:1] in ctx_chars or it.pred[:1] in ctx_chars:
                continue
        kept.append(it)
    if any(i.k == "vowel_swap" for i in kept):          # same mistake seen twice (vowel + its madd letter)
        kept = [i for i in kept if i.k != "madd_vowel_swap"]
    kept.sort(key=lambda i: (i.pos, -SEVERITY[i.t]))

    letter_issues = [i for i in kept if i.t == "letters"]
    if len(letter_issues) >= UNCLEAR_ISSUES:
        kept = [Issue("letters", "unclear", NOTE_UNCLEAR,
                      min(i.conf for i in letter_issues), 0, exp, seg, "")]
    return kept[:3], al, exp, form_costs


# ==============================================================================
# Audio buffer + voice activity
# ==============================================================================
class AudioBuffer:
    """Thread-safe growing mono buffer (float32, 16 kHz) with 20 ms frame energies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf = np.zeros(SR * 30, dtype=np.float32)
        self.n = 0
        self._frames: List[float] = []

    def append(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        with self._lock:
            room = MAX_SESSION_S * SR - self.n
            if room <= 0:
                return
            x = x[:room]
            need = self.n + x.size
            if need > self._buf.size:
                nb = np.zeros(max(need, self._buf.size * 2), dtype=np.float32)
                nb[: self.n] = self._buf[: self.n]
                self._buf = nb
            self._buf[self.n:need] = x
            self.n = need
            nf = self.n // FRAME
            have = len(self._frames)
            if nf > have:
                seg = self._buf[have * FRAME: nf * FRAME].reshape(-1, FRAME)
                self._frames.extend(np.sqrt((seg * seg).mean(axis=1) + 1e-12).tolist())

    def snapshot(self, a: int, b: int) -> np.ndarray:
        with self._lock:
            return self._buf[max(0, a): min(self.n, b)].copy()

    def frames_array(self) -> np.ndarray:
        with self._lock:
            return np.asarray(self._frames, dtype=np.float64)


def voiced_mask(frames: np.ndarray, thr: float, fill: int = 4, min_run: int = 2) -> np.ndarray:
    m = frames >= thr
    n = len(m)
    out = m.copy()
    i = 0
    while i < n:                           # bridge short gaps (stop consonants, breaths)
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= fill:
                out[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < n:                           # drop tiny islands (clicks)
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_run:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


@dataclass
class Recognition:
    text: str
    probs: Optional[List[float]] = None
    sifat: Optional[List[Dict[str, Any]]] = None


# ==============================================================================
# Streaming session engine
# ==============================================================================
class SessionEngine:
    """
    One reciter, one ayah. Feed audio with `add_audio`, call `step` whenever `ready_to_run()`.

      cursor    number of words already judged (final, never revisited)
      tracked   index of the word the reciter is on right now (last word reached)
      win_start first sample of the next model window (always a bit before the first unjudged word)
    """

    def __init__(self, ref: RefAya, recognize: Callable[[np.ndarray], Recognition],
                 preset: Preset, meta: Optional[Dict[str, Any]] = None) -> None:
        self.ref = ref
        self.recognize = recognize
        self.preset = preset
        self.buf = AudioBuffer()
        self.cursor = 0
        self.tracked = -1
        self.win_start = 0
        self.last_run_end = 0
        self.pause_checked_at = -1
        self.results: Dict[int, Tuple[str, List[Issue]]] = {}
        self.seq = 0
        self.rtf: Optional[float] = None
        self.complete = False
        self.last_window: Tuple[int, int] = (0, 0)
        self.skip_mode = False
        self._flush_iters = 0
        self.debug: Dict[str, Any] = {
            "created": time.time(), "sura": ref.sura, "aya": ref.aya, "preset": preset.name,
            "approx_segmentation": ref.approx, "ref_info": ref.info, "meta": meta or {}, "steps": [],
        }

    # ------------------------------------------------------------------ audio in
    def add_audio(self, x: np.ndarray) -> None:
        if not self.complete:
            self.buf.append(x)

    @property
    def audio_ms(self) -> int:
        return int(self.buf.n * 1000 / SR)

    # ------------------------------------------------------------------ vad helpers
    @staticmethod
    def _thr(frames: np.ndarray) -> float:
        if len(frames) < 8:
            return 0.010
        floor = float(np.percentile(frames[-400:], 10))
        return float(min(max(3.0 * floor, 0.005), 0.015))

    def _mask_from(self, frames: np.ndarray, start: int, stop: Optional[int] = None) -> np.ndarray:
        """Voiced mask for frames[start:stop] (local indexing)."""
        seg = frames[start:stop]
        if len(seg) == 0:
            return np.zeros(0, dtype=bool)
        return voiced_mask(seg, self._thr(frames))

    # ------------------------------------------------------------------ scheduling
    def ready_to_run(self) -> bool:
        if self.complete:
            return False
        frames = self.buf.frames_array()
        nfr = len(frames)
        if nfr < 5:
            return False
        n = self.buf.n
        new_s = (n - self.last_run_end) / SR
        f0 = min(nfr, self.last_run_end // FRAME)
        if new_s >= MIN_NEW_AUDIO_S:
            voiced_new = float(self._mask_from(frames, f0).sum()) * FRAME / SR
            if voiced_new >= 0.2:
                return True
        if self.tracked >= self.cursor:
            tail0 = max(0, nfr - 200)
            vidx = np.nonzero(self._mask_from(frames, tail0))[0]
            if len(vidx):
                last_v = tail0 + int(vidx[-1])
                if (nfr - 1 - last_v) * FRAME / SR >= TAIL_SILENCE_S and last_v != self.pause_checked_at:
                    return True            # the reciter just paused: settle the last word
        return False

    # ------------------------------------------------------------------ one model window
    def step(self, flush: bool = False) -> Optional[Dict[str, Any]]:
        """Judges the next chunk of audio chronologically (never skips a backlog)."""
        if self.complete:
            return None
        t0 = time.time()
        n = self.buf.n
        frames = self.buf.frames_array()
        nfr = len(frames)
        if nfr < 5:
            return None

        ws = self.win_start
        forced = self.skip_mode
        if n - ws > int(MAX_BACKLOG_S * SR):                  # hopelessly behind: skip old audio
            ws = n - int(MAX_WINDOW_S * SR)
            forced = True
        we = min(n, ws + int(MAX_WINDOW_S * SR))
        self.last_run_end = we
        at_live_edge = we >= n - FRAME
        wf0, wf1 = ws // FRAME, min(nfr, we // FRAME)
        vidx = np.nonzero(self._mask_from(frames, wf0, wf1))[0]
        if len(vidx) * FRAME / SR < MIN_VOICED_S:
            if not at_live_edge:                               # silent stretch: move on
                self.win_start = max(self.win_start, we - int(0.5 * SR))
            return None
        ts = max(0, int(vidx[0]) - 12)
        te = min(wf1 - wf0, int(vidx[-1]) + 19)
        wave = self.buf.snapshot((wf0 + ts) * FRAME, (wf0 + te) * FRAME)
        last_v_abs = wf0 + int(vidx[-1])
        tail_sil = (wf1 - 1 - last_v_abs) * FRAME / SR
        pause_end = tail_sil >= TAIL_SILENCE_S
        settle = pause_end or (flush and at_live_edge)
        vsel = vidx[(vidx >= ts) & (vidx < te)] + wf0          # voiced frames inside the window
        if wave.size < int(0.25 * SR):
            return None

        self.last_window = ((wf0 + ts) * FRAME, (wf0 + te) * FRAME)
        t_inf = time.time()
        rec = self.recognize(wave)
        inf_s = time.time() - t_inf
        r = inf_s / max(0.25, wave.size / SR)
        self.rtf = r if self.rtf is None else 0.7 * self.rtf + 0.3 * r

        pred = clean_phonemes(rec.text)
        probs = rec.probs if (rec.probs is not None and pred == rec.text
                              and len(rec.probs) == len(pred)) else None
        pred_gmap = (group_index_map(rec.sifat, pred) or tolerant_group_map(rec.sifat, pred)) if rec.sifat else None
        step_dbg: Dict[str, Any] = {
            "seq": self.seq + 1, "win_s": [round(wf0 * FRAME / SR, 2), round(n / SR, 2)],
            "wave_s": round(wave.size / SR, 2), "inf_s": round(inf_s, 2), "forced": forced,
            "flush": flush, "pause_end": pause_end, "pred": pred, "cursor_in": self.cursor,
        }
        step_dbg["sifat"] = {"pred_groups": len(rec.sifat) if rec.sifat else 0,
                             "pred_mapped": pred_gmap is not None,
                             "ref_tiles": self.ref.group_of_char is not None,
                             "probs": probs is not None}
        self.seq += 1
        updates: List[Dict[str, Any]] = []
        tracked_before = self.tracked

        cursor_in = self.cursor
        if pred:
            updates = self._align_and_commit(
                pred, probs, rec, pred_gmap, vsel, forced, settle, flush, step_dbg, we)
        if pause_end and at_live_edge:
            self.pause_checked_at = last_v_abs
        if self.cursor > cursor_in:
            self.skip_mode = False
        elif (we - ws) >= int(MAX_WINDOW_S * SR) - 5 * FRAME and not self.complete:
            self.win_start = max(self.win_start, ws + int(4 * SR))   # nothing usable in 10 s: move on
            self.skip_mode = True
        step_dbg["total_s"] = round(time.time() - t0, 2)
        step_dbg["tracked"] = self.tracked
        step_dbg["cursor_out"] = self.cursor
        dbg = self.debug["steps"]
        dbg.append(step_dbg)
        if len(dbg) > DEBUG_KEEP_STEPS:
            del dbg[0]
        if not updates and self.tracked == tracked_before:
            return None
        return self._progress(updates, round((time.time() - t0) * 1000), we)

    # ------------------------------------------------------------------ alignment + commit
    def _align_and_commit(self, pred: str, probs: Optional[List[float]], rec: Recognition,
                          pred_gmap: Optional[List[int]], vsel: np.ndarray, forced: bool,
                          settle: bool, flush: bool, dbg: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        ref, N, c = self.ref, self.ref.n, self.cursor
        a = max(0, c - CTX_WORDS)
        b = min(N, c + REGION_AHEAD_WORDS)
        base = ref.starts[a]
        ref_str = ref.full_ph[base: ref.starts[b]]
        lead_free = ref.starts[c] - base
        lead_pen = 0.0
        if forced:
            lead_free = ref.starts[min(N, c + 8)] - base
            lead_pen = 0.05
        al = align(
            ref_str, pred,
            lead_free=lead_free, lead_pen=lead_pen,
            pred_lead_free=60 if c == 0 else 0,
            tail_free_ref=True,
            tail_ins_chars=60 if b == N else 0,
        )
        L = len(pred)

        # ---- per-word bookkeeping
        n_loc = b - a
        pj_of_word: Dict[int, List[int]] = {}
        run_id: List[int] = [-1] * len(ref_str)
        rid = -1
        last_letter, last_w = "", -1
        for i, ch in enumerate(ref_str):
            w = ref.word_of_char[base + i]
            if w != last_w:
                last_letter, last_w = "", w
            if is_mark(ch):
                continue
            if ch != last_letter:
                rid += 1
            last_letter = ch
            run_id[i] = rid
        matched_runs: Dict[int, set] = {}
        matched_letters = 0
        for ri, pj in al.pairs:
            if ri < 0 or pj < 0:
                continue
            w = ref.word_of_char[base + ri]
            pj_of_word.setdefault(w, []).append(pj)
            if ref_str[ri] == pred[pj] and not is_mark(pred[pj]):
                matched_runs.setdefault(w, set()).add(run_id[ri])
                if w >= c:
                    matched_letters += 1

        def cov(w: int) -> float:
            total = len(ref.core[w])
            return 1.0 if total == 0 else len(matched_runs.get(w, ())) / total

        consumed = max(1, al.i_end - lead_free)
        cost_pc = al.cost / consumed
        reached = -1
        for w in range(c, b):
            if cov(w) >= WORD_REACH_COV and w in pj_of_word:
                reached = w
        dbg.update({
            "region": [a, b], "cost_per_char": round(cost_pc, 3), "matched_letters": matched_letters,
            "reached": reached, "i_end": al.i_end, "lead_free": lead_free,
        })
        quality_ok = cost_pc <= QUALITY_MAX_COST and matched_letters >= MIN_MATCHED_LETTERS
        dbg["quality_ok"] = quality_ok
        if not quality_ok or reached < c:
            return []

        self.tracked = max(self.tracked, reached)
        end_elig = reached + 1 if (settle and cov(reached) >= LAST_WORD_SETTLE_COV) else reached
        dbg["end_elig"] = end_elig
        if end_elig <= c:
            return []

        words_with_pj = sorted(pj_of_word)
        first_matched_word = next((w for w in words_with_pj if w >= c and cov(w) >= WORD_SKIP_COV), c)
        updates: List[Dict[str, Any]] = []
        judged_dbg: List[Dict[str, Any]] = []
        for w in range(c, end_elig):
            if forced and w < first_matched_word:
                self.results[w] = ("unchecked", [])
            elif w not in pj_of_word or cov(w) < WORD_SKIP_COV:
                self.results[w] = ("letters", [Issue("letters", "skipped", NOTE_SKIPPED, 1.0)])
            else:
                issues, jinfo = self._judge(w, pj_of_word, words_with_pj, pred, probs, rec, pred_gmap,
                                            al, base, ref_str, vsel)
                self.results[w] = (status_of(issues), issues)
            st, iss = self.results[w]
            updates.append(self._update_item(w))
            entry = {"w": w, "s": st, "issues": [i.to_debug() for i in iss]}
            entry.update(locals().get("jinfo", {}) if st != "unchecked" else {})
            judged_dbg.append(entry)
        dbg["judged"] = judged_dbg
        self.cursor = end_elig
        if self.cursor >= N:
            self.complete = True
        else:
            self._advance_window(pj_of_word, words_with_pj, L, vsel, n, dbg)
        return updates

    def _judge(self, w: int, pj_of_word: Dict[int, List[int]], words_with_pj: List[int], pred: str,
               probs: Optional[List[float]], rec: Recognition, pred_gmap: Optional[List[int]],
               al: Alignment, base: int, ref_str: str,
               vsel: np.ndarray) -> Tuple[List[Issue], Dict[str, Any]]:
        ref, L = self.ref, len(pred)
        pjs = pj_of_word[w]
        first, last = min(pjs), max(pjs) + 1
        k = words_with_pj.index(w)
        prev_end = (max(pj_of_word[words_with_pj[k - 1]]) + 1) if k > 0 else 0
        next_start = min(pj_of_word[words_with_pj[k + 1]]) if k + 1 < len(words_with_pj) else L
        lo = max(prev_end, first - 3)
        hi = min(next_start, last + 3)
        hi = max(hi, last)
        seg = pred[lo:hi]
        seg_probs = probs[lo:hi] if probs is not None else None
        approx_w = ref.word_approx[w]
        ctx = ""
        if approx_w:
            if w > 0:
                ctx += ref.wasl[w - 1][-2:]
            if w + 1 < ref.n:
                ctx += ref.wasl[w + 1][:2]
        issues, _wal, form, costs = judge_word(
            ref.forms(w), seg, seg_probs, self.preset,
            lead=first - lo, trail=hi - last, approx=approx_w, ctx_chars=ctx)
        extra = self._waqf_without_pause(w, form, costs, pj_of_word, L, vsel)
        if extra is not None:
            issues = issues + [extra]
        trace: List[Any] = []
        if self.preset.sifat_levels and ref.group_of_char is not None and pred_gmap is not None and rec.sifat:
            s_issues, trace = self._sifat_issues(w, al, base, ref_str, pred, pred_gmap, rec.sifat)
            issues = issues + s_issues
        info = {"seg": seg, "exp": form, "form": "waqf" if (form != ref.wasl[w]) else "wasl",
                "sifat_cmp": trace[:14], "exact": not approx_w}
        return issues[:3], info

    def _waqf_without_pause(self, w: int, form: str, costs: Dict[str, float],
                            pj_of_word: Dict[int, List[int]], L: int,
                            vsel: np.ndarray) -> Optional[Issue]:
        """Word-final vowel dropped (stop form) although the reciter kept going: tashkeel mistake.
        A real stop shows up as a silence in the audio next to the word boundary; no silence = no stop."""
        ref = self.ref
        waqf = ref.waqf[w]
        if (not self.preset.flag_waqf_no_pause or not waqf or waqf == ref.wasl[w] or form != waqf
                or w + 1 >= ref.n or (w + 1) not in pj_of_word or len(vsel) < 6):
            return None
        if costs.get(ref.wasl[w], 99.0) - costs.get(waqf, 99.0) < 0.3:
            return None                    # the stop form is not clearly better than the continuation
        t_end = self._pred_to_sample(max(pj_of_word[w]), L, vsel)
        t_next = self._pred_to_sample(min(pj_of_word[w + 1]), L, vsel)
        if t_end is None or t_next is None:
            return None
        lo_f = max(min(t_end, t_next) // FRAME - 25, int(vsel[0]))      # +-0.5 s around the boundary
        hi_f = min(max(t_end, t_next) // FRAME + 25, int(vsel[-1]))
        if hi_f - lo_f < 10:
            return None
        vset = set(int(x) for x in vsel)
        run = best = 0
        for f in range(lo_f, hi_f + 1):
            if f in vset:
                run = 0
            else:
                run += 1
                best = max(best, run)
        if best >= 5:                      # >= 100 ms of silence: a real pause, the stop form is fine
            return None
        vowel = next((ch for ch in reversed(ref.wasl[w]) if ch in CORE_VOWELS), "")
        return Issue("tashkeel", "waqf_no_pause", note_waqf_no_pause(vowel), 0.9, 0, waqf, ref.wasl[w], "")

    def _sifat_issues(self, w: int, al: Alignment, base: int, ref_str: str, pred: str,
                      pred_gmap: List[int], pred_sifat: List[Dict[str, Any]]) -> Tuple[List[Issue], List[Any]]:
        """Compares the model's per-letter properties with the expected ones. Returns (issues, trace)
        where trace lists [letter, level, expected, predicted, prob, flagged] for the report."""
        ref = self.ref
        out: List[Issue] = []
        trace: List[Any] = []
        seen = set()
        pos = lambda s, key: key in s and "not_" not in s and "no_" not in s
        for ri, pj in al.pairs:
            if ri < 0 or pj < 0 or ref.word_of_char[base + ri] != w:
                continue
            if ref_str[ri] != pred[pj] or is_mark(pred[pj]):
                continue
            g, h = ref.group_of_char[base + ri], pred_gmap[pj]
            if (g, h) in seen or g >= len(ref.exp_groups) or h >= len(pred_sifat):
                continue
            seen.add((g, h))
            eg, pg = ref.exp_groups[g], pred_sifat[h]
            el = next(iter(letters_only(eg["g"])), "")
            pl = next(iter(letters_only(pg["g"])), "")
            if not el or el != pl:
                continue
            for lv in self.preset.sifat_levels:
                elab, _ = eg["L"].get(lv, (None, None))
                plab, pprob = pg["L"].get(lv, (None, None))
                if elab is None or plab is None:
                    continue
                if "PAD" in elab or "PAD" in plab:
                    continue                     # the model abstains for letters where this property is undefined
                if lv == "ghonna" and el not in GHUNNA_LETTERS:
                    continue                     # ghunna on ل/ت/... is impossible: it is a label misalignment
                if lv == "tafkheem_or_taqeeq" and el not in TAFKHEEM_LETTERS:
                    continue
                conf = pprob if pprob is not None else 0.92
                flagged = False
                if elab != plab and self._label_shift(lv, g, h, elab, plab, pred_sifat):
                    trace.append([el, lv + "/shift-ignored", elab, plab, round(conf, 3), False])
                    continue
                if elab != plab and conf >= self.preset.conf_sifat:
                    if lv == "qalqla" and pos(elab, "moqalqal") and not pos(plab, "moqalqal"):
                        out.append(Issue("tajweed", "qalqla", note_qalqla(el), conf, 0, elab, plab, el))
                        flagged = True
                    elif lv == "ghonna":
                        e_yes, p_yes = pos(elab, "maghnoon"), pos(plab, "maghnoon")
                        if e_yes != p_yes:
                            out.append(Issue("tajweed", "ghonna", note_ghonna(el, e_yes), conf, 0, elab, plab, el))
                            flagged = True
                    elif lv == "tafkheem_or_taqeeq":
                        e_h = "mofakham" in elab and "low" not in elab
                        p_h = "mofakham" in plab and "low" not in plab
                        e_l, p_l = "moraqaq" in elab, "moraqaq" in plab
                        if (e_h and p_l) or (e_l and p_h):
                            out.append(Issue("tajweed", "tafkheem", note_tafkheem(el, e_h), conf, 0, elab, plab, el))
                            flagged = True
                interesting = (lv == "tafkheem_or_taqeeq" and elab != plab) or \
                              (lv != "tafkheem_or_taqeeq" and (pos(elab, "moqalqal") or pos(plab, "moqalqal")
                                                                 or pos(elab, "maghnoon") or pos(plab, "maghnoon")))
                if interesting:
                    trace.append([el, lv, elab, plab, round(conf, 3), flagged])
        return out, trace

    def _label_shift(self, lv: str, g: int, h: int, elab: str, plab: str,
                     pred_sifat: List[Dict[str, Any]]) -> bool:
        """The live report showed neighbouring letters with SWAPPED labels (م loses its ghunna while the
        next ل gains one): that is the model's label sequence sliding by one group, not a mistake."""
        groups = self.ref.exp_groups
        for dg in (-1, 1):
            g2, h2 = g + dg, h + dg
            if 0 <= g2 < len(groups) and 0 <= h2 < len(pred_sifat):
                e2 = groups[g2]["L"].get(lv, (None, None))[0]
                p2 = pred_sifat[h2]["L"].get(lv, (None, None))[0]
                if e2 is not None and p2 is not None and e2 == plab and p2 == elab and e2 != elab:
                    return True
        return False

    # ------------------------------------------------------------------ window management
    def _pred_to_sample(self, pj: int, L: int, vsel: np.ndarray) -> Optional[int]:
        if len(vsel) == 0 or L == 0:
            return None
        k = min(len(vsel) - 1, int((pj + 0.5) / L * len(vsel)))
        return int(vsel[k]) * FRAME

    def _snap(self, sample: int) -> int:
        frames = self.buf.frames_array()
        if len(frames) == 0:
            return max(0, sample)
        f = sample // FRAME
        lo, hi = max(0, f - 12), min(len(frames), f + 13)
        if hi <= lo:
            return max(0, sample)
        return (lo + int(np.argmin(frames[lo:hi]))) * FRAME

    def _advance_window(self, pj_of_word: Dict[int, List[int]], words_with_pj: List[int],
                        L: int, vsel: np.ndarray, n: int, dbg: Dict[str, Any]) -> None:
        ctx = self.cursor - 1
        pj = None
        for w in words_with_pj:
            if w >= ctx:
                pj = min(pj_of_word[w])
                break
        if pj is None:
            return
        t = self._pred_to_sample(pj, L, vsel)
        if t is None:
            return
        new_ws = self._snap(t - int(CTX_MARGIN_S * SR))
        new_ws = min(new_ws, max(self.win_start, n - int(0.8 * SR)))
        if new_ws > self.win_start:
            self.win_start = new_ws
        dbg["next_win_start_s"] = round(self.win_start / SR, 2)

    # ------------------------------------------------------------------ messages
    def _update_item(self, w: int) -> Dict[str, Any]:
        st, issues = self.results[w]
        d: Dict[str, Any] = {"i": w, "s": st}
        if issues:
            d["issues"] = [i.to_client() for i in issues]
        return d

    def _progress(self, updates: List[Dict[str, Any]], busy_ms: int, n: int) -> Dict[str, Any]:
        return {
            "type": "progress", "seq": self.seq, "audio_ms": int(n * 1000 / SR),
            "tracked": self.tracked, "committed": self.cursor, "updates": updates,
            "busy_ms": busy_ms, "rtf": round(self.rtf, 3) if self.rtf is not None else None,
        }

    def final_message(self) -> Dict[str, Any]:
        words = [self._update_item(w) for w in sorted(self.results)]
        counts = {"ok": 0, "tashkeel": 0, "tajweed": 0, "letters": 0, "unchecked": 0}
        for w in self.results:
            counts[self.results[w][0]] = counts.get(self.results[w][0], 0) + 1
        summary = dict(counts)
        summary["total"] = self.ref.n
        summary["judged"] = len(self.results)
        summary["complete"] = bool(self.complete or self.cursor >= self.ref.n)
        return {"type": "final", "audio_ms": self.audio_ms, "tracked": self.tracked,
                "committed": self.cursor, "words": words, "summary": summary}

    def flush_step(self) -> Tuple[Optional[Dict[str, Any]], bool]:
        """One more window while finishing (the reciter pressed stop). Returns (progress message, done)."""
        before = (self.cursor, self.tracked, self.win_start, self.last_run_end)
        msg = self.step(flush=True)
        after = (self.cursor, self.tracked, self.win_start, self.last_run_end)
        self._flush_iters += 1
        done = (self.complete or self._flush_iters >= 80
                or (after == before and self.last_run_end >= self.buf.n - FRAME))
        return msg, done

    def flush(self) -> Dict[str, Any]:
        """Judge everything that was heard, in order, then build the final report."""
        while True:
            _msg, done = self.flush_step()
            if done:
                break
        return self.final_message()


def run_offline(ref: RefAya, recognize: Callable[[np.ndarray], Recognition], preset: Preset,
                wave_f32: np.ndarray, hop_s: float = 0.5) -> Dict[str, Any]:
    """Replays a finished recording through the same streaming engine (used by /analyze and tests)."""
    eng = SessionEngine(ref, recognize, preset, meta={"mode": "offline"})
    hop = int(hop_s * SR)
    for s in range(0, len(wave_f32), hop):
        eng.add_audio(wave_f32[s:s + hop])
        while eng.ready_to_run():
            eng.step()
            if eng.complete:
                break
        if eng.complete:
            break
    msg = eng.flush()
    msg["debug"] = eng.debug
    return msg


# ==============================================================================
# Audio decoding helpers
# ==============================================================================
def decode_audio_bytes(raw: bytes) -> np.ndarray:
    """RIFF/WAV (any rate, 8/16-bit, mono/stereo) or raw 16 kHz int16 PCM -> float32 mono 16 kHz."""
    if not raw:
        return np.zeros(0, dtype=np.float32)
    if len(raw) >= 44 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(raw), "rb") as wf:
                sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            if sw == 2:
                arr = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            else:
                arr = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            if ch > 1:
                arr = arr[: (arr.size // ch) * ch].reshape(-1, ch).mean(axis=1)
            if sr != SR and arr.size:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
            return arr.astype(np.float32)
        except Exception:
            pass
    even = (len(raw) // 2) * 2
    return np.frombuffer(raw[:even], dtype="<i2").astype(np.float32) / 32768.0 if even else np.zeros(0, np.float32)


def prepare_wave(x: np.ndarray) -> np.ndarray:
    """Remove DC offset and clip. (The model's feature extractor normalises loudness itself.)"""
    x = x.astype(np.float32)
    if x.size:
        x = x - float(np.mean(x))
        np.clip(x, -1.0, 1.0, out=x)
    return x


# ==============================================================================
# Model runtime (loaded once per container)
# ==============================================================================
class QuranLib:
    """Thin wrapper around quran-transcript (the only place that touches that library)."""

    def __init__(self) -> None:
        from quran_transcript import MoshafAttributes
        self.moshaf = MoshafAttributes(
            rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4,
            madd_mottasel_waqf=4, madd_aared_len=4,
        )

    def get_uthmani(self, sura: int, aya: int) -> str:
        from quran_transcript import Aya
        return Aya(sura, aya).get().uthmani

    def phonetize(self, text: str, remove_spaces: bool) -> Any:
        from quran_transcript import quran_phonetizer
        return quran_phonetizer(text, self.moshaf, remove_spaces=remove_spaces)


SESSIONS: "deque[Dict[str, Any]]" = deque(maxlen=3)      # debug traces of the latest sessions


class ModelRuntime:
    def __init__(self) -> None:
        self.muaalem: Any = None
        self.lib: Optional[QuranLib] = None
        self.device = "cpu"
        self.int8 = False
        self.bench_rtf: Optional[float] = None
        self.bench_rtf_int8: Optional[float] = None
        self.load_seconds: Optional[float] = None
        self.load_error: Optional[str] = None
        self.diag: Dict[str, Any] = {}
        self._refs: Dict[Tuple[int, int], RefAya] = {}
        self._load_lock = threading.Lock()
        self._ref_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # ---------------------------------------------------------------- loading
    def ensure_loaded(self) -> None:
        if self.muaalem is not None:
            return
        with self._load_lock:
            if self.muaalem is not None:
                return
            t0 = time.time()
            import torch
            from quran_muaalem import Muaalem

            torch.set_num_threads(NUM_THREADS)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.lib = QuranLib()
            m = Muaalem(model_name_or_path=MODEL_ID, device=self.device, dtype=dtype)
            try:
                ref: Optional[RefAya] = self.get_ref(DEFAULT_SURA, DEFAULT_AYA)   # also pre-computes waqf forms
            except Exception:
                ref = None                                          # retried at the first session
            if self.device == "cpu" and ref is not None:
                self._tune_precision(m, ref)
            self.muaalem = m
            self.load_seconds = round(time.time() - t0, 1)

    def _time_inference(self, m: Any, ref: RefAya, seconds: float = 3.0) -> Optional[float]:
        """Real-time factor of one forward pass (content independent). None if it cannot be measured."""
        import torch
        rng = np.random.default_rng(0)
        wave_arr = (rng.standard_normal(int(SR * seconds)) * 0.03).astype(np.float32)
        try:
            with torch.inference_mode():
                m([wave_arr], [ref.full_out], sampling_rate=SR)           # warm-up
                t = time.time()
                m([wave_arr], [ref.full_out], sampling_rate=SR)
                return (time.time() - t) / seconds
        except Exception:
            pass
        try:
            proc, model = getattr(m, "processor", None), getattr(m, "model", None)
            if proc is None or model is None:
                return None
            feats = proc([wave_arr], sampling_rate=SR, return_tensors="pt")
            with torch.inference_mode():
                model(**feats)
                t = time.time()
                model(**feats)
                return (time.time() - t) / seconds
        except Exception:
            return None

    def _tune_precision(self, m: Any, ref: RefAya) -> None:
        """Keeps fp32 when the CPU is fast enough; otherwise tries dynamic int8 (and keeps it only
        if it really is faster and still runs). Any failure leaves the original fp32 model."""
        rtf = self._time_inference(m, ref)
        self.bench_rtf = None if rtf is None else round(rtf, 3)
        want_int8 = PRECISION == "int8" or (PRECISION == "auto" and rtf is not None and rtf > AUTO_INT8_ABOVE_RTF)
        if not want_int8 or getattr(m, "model", None) is None:
            return
        import torch
        old = m.model
        try:
            qfn = getattr(getattr(torch, "ao", None), "quantization", None) or torch.quantization
            q = qfn.quantize_dynamic(old, {torch.nn.Linear}, dtype=torch.qint8)
            m.model = q
            rq = self._time_inference(m, ref)
            if rq is not None and (rtf is None or rq < rtf * 0.9):
                self.int8 = True
                self.bench_rtf_int8 = round(rq, 3)
                del old
            else:
                m.model = old
                del q
        except Exception:
            m.model = old
        finally:
            gc.collect()

    # ---------------------------------------------------------------- references
    def get_ref(self, sura: int, aya: int) -> RefAya:
        key = (int(sura), int(aya))
        ref = self._refs.get(key)
        if ref is not None:
            return ref
        with self._ref_lock:
            ref = self._refs.get(key)
            if ref is None:
                if self.lib is None:
                    self.lib = QuranLib()
                ref = build_ref_aya(key[0], key[1], self.lib)
                self._refs[key] = ref
        return ref

    # ---------------------------------------------------------------- inference
    def recognize(self, wave_arr: np.ndarray, ref: RefAya) -> Recognition:
        self.ensure_loaded()
        import torch
        x = prepare_wave(wave_arr)
        with self._infer_lock:
            with torch.inference_mode():
                outs = self.muaalem([x], [ref.full_out], sampling_rate=SR)
        out = outs[0]
        units = getattr(out, "phonemes", None)
        text = str(getattr(units, "text", "") or "")
        probs: Optional[List[float]] = None
        raw_probs = getattr(units, "probs", None)
        if raw_probs is not None:
            try:
                if hasattr(raw_probs, "detach"):
                    raw_probs = raw_probs.detach().float().cpu().numpy()
                arr = np.asarray(raw_probs, dtype=np.float64).reshape(-1)
                probs = arr.tolist() if arr.size == len(text) else None
            except Exception:
                probs = None
        raw_sifat = getattr(out, "sifat", None)
        sifat = convert_sifat(raw_sifat)
        if "pred_sifat_n" not in self.diag or (raw_sifat and "pred_sifat_sample" not in self.diag):
            try:                                   # shape of the library objects, for the debug report
                self.diag["pred_sifat_n"] = len(raw_sifat) if raw_sifat else 0
                if raw_sifat:
                    self.diag["pred_sifat_sample"] = repr(raw_sifat[0])[:300]
                self.diag["pred_sifat_converted"] = bool(sifat)
                self.diag["phoneme_probs"] = probs is not None
                self.diag["out_fields"] = [a for a in dir(out) if not a.startswith("_")][:20]
            except Exception:
                pass
        return Recognition(text, probs, sifat)

    def status(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.muaalem is not None else ("error" if self.load_error else "loading"),
            "engine": "rafiq-v14.2", "model": MODEL_ID, "device": self.device, "int8": self.int8,
            "bench_rtf_fp32": self.bench_rtf, "bench_rtf_int8": self.bench_rtf_int8,
            "load_seconds": self.load_seconds, "load_error": self.load_error, "diag": self.diag,
            "refs": [{"sura": k[0], "aya": k[1], "words": v.n, "approx": v.approx,
                      "segmentation": v.info.get("segmentation"), "waqf_forms": v.info.get("waqf_forms"),
                      "sifat": v.group_of_char is not None} for k, v in self._refs.items()],
        }


RUNTIME = ModelRuntime()


def debug_snapshot(eng: "SessionEngine", runtime: ModelRuntime, steps: int = 40) -> Dict[str, Any]:
    """Compact trace of a session, sent with the final message (the server may scale to zero right
    after a session, so the app keeps its own copy for the 'technical report' button)."""
    d = dict(eng.debug)
    d["steps"] = list(d.get("steps", []))[-steps:]
    d["server"] = runtime.status()
    return d


# ==============================================================================
# FastAPI application
# ==============================================================================
def build_fastapi_app(runtime: ModelRuntime = RUNTIME):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    api = FastAPI(title="Rafiq Quran recitation corrector", version="14.0.0")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @api.get("/health")
    async def health() -> Dict[str, Any]:
        return runtime.status()

    @api.get("/debug/last")
    async def debug_last(steps: int = 30) -> JSONResponse:
        if not SESSIONS:
            return JSONResponse({"info": "no session yet", "server": runtime.status()})
        d = dict(SESSIONS[-1])
        keep = max(1, min(int(steps), DEBUG_KEEP_STEPS))
        d["steps"] = list(d.get("steps", []))[-keep:]
        d["server"] = runtime.status()
        return JSONResponse(d)

    @api.get("/debug/ref")
    async def debug_ref(sura_idx: int = DEFAULT_SURA, aya_idx: int = DEFAULT_AYA) -> JSONResponse:
        ref = await asyncio.to_thread(runtime.get_ref, sura_idx, aya_idx)
        return JSONResponse(ref.describe())

    @api.post("/analyze")
    async def analyze(
        file: UploadFile = File(...),
        sura_idx: int = Form(DEFAULT_SURA),
        aya_idx: int = Form(DEFAULT_AYA),
        strictness: str = Form("balanced"),
    ) -> JSONResponse:
        raw = await file.read()
        wave_arr = decode_audio_bytes(raw)
        if wave_arr.size < int(SR * 0.5):
            raise HTTPException(status_code=400, detail="التسجيل قصير جدًا")
        await asyncio.to_thread(runtime.ensure_loaded)
        ref = await asyncio.to_thread(runtime.get_ref, sura_idx, aya_idx)
        msg = await asyncio.to_thread(
            run_offline, ref, lambda w: runtime.recognize(w, ref), get_preset(strictness), wave_arr)
        return JSONResponse(msg)

    @api.websocket("/ws/live-recitation")
    async def ws_live(websocket: WebSocket) -> None:
        await websocket.accept()
        qp = websocket.query_params
        cfg = {
            "sura": int(qp.get("sura_idx", DEFAULT_SURA)),
            "aya": int(qp.get("aya_idx", DEFAULT_AYA)),
            "strictness": qp.get("strictness", "balanced"),
        }
        state: Dict[str, Any] = {"engine": None}
        wake, stop_evt = asyncio.Event(), asyncio.Event()
        send_lock = asyncio.Lock()

        async def send(obj: Dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_text(json.dumps(obj, ensure_ascii=False))

        async def new_engine() -> SessionEngine:
            await asyncio.to_thread(runtime.ensure_loaded)
            ref = await asyncio.to_thread(runtime.get_ref, cfg["sura"], cfg["aya"])
            eng = SessionEngine(ref, lambda w: runtime.recognize(w, ref), get_preset(cfg["strictness"]),
                                meta={"int8": runtime.int8, "device": runtime.device})
            SESSIONS.append(eng.debug)
            return eng

        async def worker() -> None:
            try:
                while not stop_evt.is_set():
                    await wake.wait()
                    wake.clear()
                    eng: SessionEngine = state["engine"]
                    while not stop_evt.is_set() and not eng.complete and eng.ready_to_run():
                        try:
                            msg = await asyncio.to_thread(eng.step, False)
                        except Exception as exc:            # keep the session alive on a bad window
                            await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                            break
                        if msg:
                            await send(msg)
                    if state["engine"].complete:
                        break
                eng = state["engine"]
                while True:                    # catch up on any backlog, streaming progress to the app
                    msg, done = await asyncio.to_thread(eng.flush_step)
                    if msg:
                        await send(msg)
                    if done:
                        break
                final = eng.final_message()
                final["debug"] = debug_snapshot(eng, runtime)
                await send(final)
            except Exception as exc:
                try:
                    await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                except Exception:
                    pass
            finally:
                try:
                    await websocket.close()
                except Exception:
                    pass

        task: Optional["asyncio.Task[None]"] = None
        try:
            state["engine"] = await new_engine()
            ref = state["engine"].ref
            await send({"type": "ready", "n_words": ref.n, "words": ref.words,
                        "approx": ref.approx, "int8": runtime.int8,
                        "strictness": state["engine"].preset.name})
            task = asyncio.create_task(worker())
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    k = len(data) // 2
                    if k:
                        state["engine"].add_audio(np.frombuffer(data[: k * 2], dtype="<i2").astype(np.float32) / 32768.0)
                        wake.set()
                txt = msg.get("text")
                if txt:
                    try:
                        payload = json.loads(txt)
                    except Exception:
                        continue
                    kind = payload.get("type", "")
                    if kind == "stop":
                        stop_evt.set()
                        wake.set()
                    elif kind == "reset":
                        state["engine"] = await new_engine()
                        await send({"type": "ready", "n_words": state["engine"].ref.n,
                                    "words": state["engine"].ref.words, "approx": state["engine"].ref.approx,
                                    "int8": runtime.int8, "strictness": state["engine"].preset.name})
        except Exception as exc:
            try:
                await send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            stop_evt.set()
            if task is not None and not task.done():
                task.cancel()

    return api


# ==============================================================================
# Modal deployment (CPU by default)
# ==============================================================================
if modal is not None:
    app = modal.App(APP_NAME)

    def download_muaalem_weights() -> None:
        """Runs once at image build time so cold starts do not download the 2 GB model."""
        import torch
        from quran_muaalem import Muaalem
        from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        Muaalem(model_name_or_path=MODEL_ID, device=device, dtype=dtype)
        moshaf = MoshafAttributes(
            rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4,
            madd_mottasel_waqf=4, madd_aared_len=4,
        )
        quran_phonetizer(Aya(2, 255).get().uthmani, moshaf, remove_spaces=True)

    muaalem_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install(
            "quran-muaalem>=0.2.2",
            "transformers>=4.45.0,<5.0.0",
            "accelerate",
            "quran-transcript",
            "librosa",
            "torch",
            "fastapi[standard]",
            "websockets",
            "numba>=0.61.2",
            "soundfile",
            "numpy",
        )
        .run_function(download_muaalem_weights)
    )

    _fn_kwargs: Dict[str, Any] = dict(
        image=muaalem_image, cpu=float(NUM_THREADS), memory=4096, scaledown_window=10, timeout=900,
    )
    if USE_MEMORY_SNAPSHOT:
        _fn_kwargs["enable_memory_snapshot"] = True
    if USE_GPU:
        _fn_kwargs["gpu"] = "T4"
    _concurrent = getattr(modal, "concurrent", None)
    if _concurrent is None:                                  # very old Modal versions
        _fn_kwargs["allow_concurrent_inputs"] = 4

    def _deploy(fn: Callable[[], Any]) -> Any:
        # Same as stacking:  @app.function(...)  @modal.concurrent(max_inputs=8)  @modal.asgi_app()
        fn = modal.asgi_app()(fn)
        if _concurrent is not None:
            try:                                  # share one awake container between 2 reciters, then scale out
                fn = _concurrent(max_inputs=4, target_inputs=2)(fn)
            except TypeError:
                fn = _concurrent(max_inputs=4)(fn)
        return app.function(**_fn_kwargs)(fn)

    @_deploy
    def fastapi_asgi_app():
        try:
            RUNTIME.ensure_loaded()
        except Exception:                      # stay up and explain at /health instead of crash-looping
            RUNTIME.load_error = traceback.format_exc()[-1500:]
        return build_fastapi_app(RUNTIME)
else:  # pragma: no cover
    app = None

# Memory snapshot: everything executed at import time inside the container (after the image is built) is
# captured by Modal, so later cold starts restore a ready-to-serve model instead of loading it again.
if modal is not None and USE_MEMORY_SNAPSHOT and hasattr(modal, "is_local") and not modal.is_local():
    try:
        RUNTIME.ensure_loaded()                   # loads the model and runs warm-up inferences
    except Exception:
        RUNTIME.load_error = traceback.format_exc()[-1500:]
