"""Stage 2: build data/hsk_vocab.json from complete-hsk-vocabulary (MIT).
Keeps words in HSK 2.0 levels 1-3 ('old-N') and/or HSK 3.0 levels 1-3 ('new-N').
Run: python build_vocab.py
"""
import json, re, urllib.request

URL = "https://raw.githubusercontent.com/drkameleon/complete-hsk-vocabulary/main/complete.json"
raw = json.load(urllib.request.urlopen(URL))


def lvl(levels, prefix):
    nums = [int(m.group(1)) for l in levels if (m := re.fullmatch(prefix + r"-(\d)", l))]
    return min(nums) if nums else None


out = []
for e in raw:
    old, new = lvl(e["level"], "old"), lvl(e["level"], "new")
    if (old and old <= 3) or (new and new <= 3):
        f = e["forms"][0]
        out.append({
            "hanzi": e["simplified"],
            "pinyin": f["transcriptions"]["pinyin"],
            "pinyin_numeric": f["transcriptions"]["numeric"],
            "meanings": f.get("meanings", [])[:3],
            "pos": e.get("pos", []),
            "old_level": old if old and old <= 3 else None,   # HSK 2.0
            "new_level": new if new and new <= 3 else None,   # HSK 3.0
        })

json.dump(out, open("data/hsk_vocab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("total:", len(out),
      "| HSK2.0 L1-3:", sum(1 for w in out if w["old_level"]),
      "| HSK3.0 L1-3:", sum(1 for w in out if w["new_level"]))