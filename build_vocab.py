"""Stage 2: build data/hsk_vocab.json (HSK 2.0 Levels 1-3 ONLY) from complete-hsk-vocabulary (MIT).
Source: https://github.com/drkameleon/complete-hsk-vocabulary  (definitions: CC-CEDICT - attribute it)
Run: python build_vocab.py

Each dataset entry can have several 'forms' (readings). The first form is often a rare reading
(个 'gě', 还 'Huán', 都 'Dū', 说 'shuì'), so we pick the form whose reading matches pypinyin's reading and whose
meaning is not a 'used in / variant of / surname' stub.
"""
import json, re, urllib.request
from pypinyin import lazy_pinyin, Style

URL = "https://raw.githubusercontent.com/drkameleon/complete-hsk-vocabulary/main/complete.json"
raw = json.load(urllib.request.urlopen(URL))

JUNK = re.compile(r"^(used in|variant of|old variant|surname|abbr\.|see |erhua variant|\(archaic\)|\(old\))", re.I)


def reading_matches(hanzi, form):
    """True / False / None (None = cannot compare, e.g. 儿 erhua)."""
    ps = lazy_pinyin(hanzi, style=Style.TONE3, v_to_u=True)
    fs = form["transcriptions"]["numeric"].lower().split()
    if len(ps) != len(fs):
        return None
    for p, f in zip(ps, fs):
        pb, pt = re.sub(r"\d", "", p), re.sub(r"[^\d]", "", p)
        fb, ft = re.sub(r"\d", "", f), re.sub(r"[^\d]", "", f)
        if pb != fb:
            return False
        if ft not in ("5", "") and pt not in ("", "5") and ft != pt:
            return False
    return True


def best_form(hanzi, forms):
    def key(i_f):
        i, f = i_f
        m = reading_matches(hanzi, f)
        junk = bool(f.get("meanings")) and bool(JUNK.match(f["meanings"][0]))
        neutral = "5" in f["transcriptions"]["numeric"]
        return (m is False, junk, not neutral, i)      # lower is better
    return min(enumerate(forms), key=key)[1]


out = []
for e in raw:
    nums = [int(m.group(1)) for l in e["level"] if (m := re.fullmatch(r"old-(\d)", l))]
    if nums and min(nums) <= 3:
        f = best_form(e["simplified"], e["forms"])
        out.append({
            "hanzi": e["simplified"],
            "pinyin": f["transcriptions"]["pinyin"],
            "pinyin_numeric": f["transcriptions"]["numeric"],
            "meanings": f.get("meanings", [])[:3],
            "pos": e.get("pos", []),
            "level": min(nums),            # HSK 2.0 level (1-3)
        })

json.dump(out, open("data/hsk_vocab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("HSK 2.0 L1-3 entries:", len(out), "| per level:", {l: sum(w["level"] == l for w in out) for l in (1, 2, 3)})