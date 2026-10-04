"""Stage 4, step 1: look at the raw input BEFORE any model call.

1. classify  : hanzi | pinyin_tone | pinyin_plain | mixed
2. hard abstain (rules only, no model): empty, too long, injection-like, or Latin text that is not Pinyin
3. convert   : Pinyin -> candidate Hanzi using ONLY the HSK 1-3 vocab list (longest match).
               Toneless Pinyin is ambiguous, so alternatives are kept and unmatched syllables stay as Pinyin.

    from src.preprocess import preprocess
    p = preprocess("wo neng hanyu")
    # p["abstain"], p["input_type"], p["hanzi_candidate"], p["ambiguous"], p["unresolved"]
"""
import json, os, re, unicodedata
from functools import lru_cache
from pypinyin import lazy_pinyin, pinyin, Style
from src.vocab_check import GRAMMAR_ALLOWLIST

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(HERE, "..", "data", "hsk_vocab.json")
MAX_LEN = 60
INJECTION = ("ignore", "instruction", "system prompt", "previous", "prompt", "忽略", "指令", "提示词", "无视")

HAN = re.compile(r"[\u4e00-\u9fff]")
TONE_MARKS = re.compile(r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]")
TOKEN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z\u00fc\u00dcāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+|[^\u4e00-\u9fffA-Za-z\u00fc\u00dcāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+")


def strip_tones(s):
    """'Wǒ xǐhuan' -> 'wo xihuan'. ü becomes v (lü -> lv); both v and u spellings are accepted later."""
    s = s.lower().replace("ü", "v")
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch) != "Mn":
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)).replace("ǖ", "v")


@lru_cache(maxsize=1)
def _syllables():
    """All toneless Pinyin syllables, built from every CJK character (about 400 syllables)."""
    syl = set()
    for cp in range(0x4E00, 0x9FFF + 1):
        for group in pinyin(chr(cp), style=Style.NORMAL, heteronym=True, errors="ignore"):
            for s in group:
                s = s.replace("ü", "v")
                syl.add(s)
                if "v" in s:
                    syl.add(s.replace("v", "u"))    # people type 'lu' for 'lü'
    return frozenset(syl)


def segment(word):
    """Split a toneless Latin string into Pinyin syllables (longest first, with backtracking), or None."""
    syl = _syllables()
    def go(i):
        if i == len(word):
            return []
        for n in range(min(6, len(word) - i), 0, -1):
            if word[i:i + n] in syl:
                rest = go(i + n)
                if rest is not None:
                    return [word[i:i + n]] + rest
        return None
    return go(0)


@lru_cache(maxsize=1)
def _vocab_index():
    """Pinyin key -> list of vocab words. Longest words are tried first when matching."""
    words = [w["hanzi"] for w in json.load(open(VOCAB_PATH, encoding="utf-8"))]
    words = list(dict.fromkeys(words + sorted(GRAMMAR_ALLOWLIST)))      # same extra words the vocab check accepts
    plain, toned = {}, {}
    for w in words:
        if not all(HAN.match(c) for c in w):
            continue
        kp = tuple(s.replace("ü", "v") for s in lazy_pinyin(w, style=Style.NORMAL))
        kt = tuple(s.replace("ü", "v") for s in lazy_pinyin(w, style=Style.TONE))
        plain.setdefault(kp, []).append(w)
        toned.setdefault(kt, []).append(w)
    maxn = max((len(k) for k in plain), default=1)
    return plain, toned, maxn


def _to_hanzi(sylls, toned_sylls):
    """Longest-match a run of syllables against the vocab. Returns (text, ambiguous, unresolved)."""
    plain, toned, maxn = _vocab_index()
    out, amb, unres, i = [], [], [], 0
    while i < len(sylls):
        for n in range(min(maxn, len(sylls) - i), 0, -1):
            kp = tuple(sylls[i:i + n])
            kt = tuple(toned_sylls[i:i + n])
            cands = toned.get(kt) if any(TONE_MARKS.search(t) for t in kt) else None
            cands = cands or plain.get(kp)
            if cands:
                out.append(cands[0])
                if len(cands) > 1:
                    amb.append({"pinyin": " ".join(kp), "chosen": cands[0], "alternatives": cands[1:]})
                i += n
                break
        else:
            out.append(sylls[i]); unres.append(sylls[i]); i += 1
    return "".join(out), amb, unres


def _classify(text):
    has_han, has_lat, has_tone = bool(HAN.search(text)), bool(re.search(r"[A-Za-z]|[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜü]", text)), bool(TONE_MARKS.search(text))
    if has_han and has_lat: return "mixed"
    if has_han: return "hanzi"
    if has_lat: return "pinyin_tone" if has_tone else "pinyin_plain"
    return "none"


def preprocess(text):
    raw = text or ""
    text = raw.strip()
    res = {"raw": raw, "input_type": None, "abstain": False, "abstain_reason": None, "message": None,
           "hanzi_candidate": text, "ambiguous": [], "unresolved": []}

    def stop(reason, msg):
        res.update(abstain=True, abstain_reason=reason, message=msg)
        return res

    if not text:
        return stop("empty", "Please type a Chinese sentence or Pinyin to check.")
    if len(text) > MAX_LEN:
        return stop("too_long", "That is too long for a beginner check. Please try one short sentence.")
    low = text.lower()
    if any(k in low for k in INJECTION):
        return stop("injection", "I can only check Mandarin practice sentences. Please type a Chinese or Pinyin sentence.")

    res["input_type"] = _classify(text)
    if res["input_type"] == "none":
        return stop("no_chinese", "I could not find any Chinese or Pinyin in that. Please type a Mandarin sentence.")

    # merge Latin words that sit next to each other (only spaces between them) into runs
    toks = TOKEN.findall(text)
    pieces, run = [], []        # run = list of (original word)
    def flush():
        if run:
            pieces.append(("lat", list(run))); run.clear()
    for t in toks:
        if HAN.match(t[0]):
            flush(); pieces.append(("han", t))
        elif re.match(r"[A-Za-z\u00fc\u00dcāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]", t[0]):
            run.append(t)
        elif t.isspace() and run:
            continue
        else:
            flush(); pieces.append(("other", t))
    flush()

    out = []
    for kind, val in pieces:
        if kind != "lat":
            out.append(val); continue
        sylls, toned = [], []
        for word in val:
            segs = segment(strip_tones(word))
            if segs is None:
                return stop("not_pinyin", f"'{word}' does not look like Pinyin. Please type Hanzi or Pinyin only.")
            # split the original (toned) word the same way
            plain, pos = strip_tones(word), 0
            for s in segs:
                toned.append(unicodedata.normalize("NFC", word.replace("ü", "v").replace("ǖ", "v").replace("Ü", "V")[pos:pos + len(s)]).lower())
                sylls.append(s); pos += len(s)
        hz, amb, unres = _to_hanzi(sylls, toned)
        out.append(hz); res["ambiguous"] += amb; res["unresolved"] += unres
    res["hanzi_candidate"] = re.sub(r"\s+(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])\s+", "", "".join(out))
    return res