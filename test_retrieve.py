"""Quick retrieval check. NOT your answer key: these are throw-away dev sentences with the rule(s) we hope to see.
Run:  python test_retrieve.py              (full hybrid, needs the embedding model)
      python test_retrieve.py --lexical    (Chinese marker part only, no model)
"""
import sys
from src.retrieve import retrieve

DEV = [  # (sentence, acceptable rule ids)
    ("我去学校昨天。", {"G1-10"}),
    ("你是谁吗？", {"G1-07"}),
    ("他有十岁。", {"G1-24"}),
    ("我不有书。", {"G1-13"}),
    ("他比我很大。", {"G2-03"}),
    ("昨天我不去商店。", {"G1-14"}),
    ("我们去一起学校。", {"G2-06"}),
    ("我吃了完饭。", {"G2-10"}),
    ("这个书很好。", {"G2-16", "G2-20", "G1-09"}),
    ("你喜欢茶吗你？", {"G1-04"}),
    ("他最很高。", {"G2-04"}),
    ("明天我又来。", {"G3-03"}),
]

dense = "--lexical" not in sys.argv
h1 = h3 = 0
for text, want in DEV:
    top = retrieve(text, k=3, use_dense=dense)
    ids = [r["rule_id"] for r in top]
    h1 += ids[0] in want
    h3 += bool(want & set(ids))
    print(f"{text:<14} want {sorted(want)}  got {ids}  {'OK' if want & set(ids) else 'MISS'}")
print(f"\nhit@1 {h1}/{len(DEV)}   hit@3 {h3}/{len(DEV)}   ({'hybrid' if dense else 'lexical only'})")
from src.retrieve import _collection
print("Rules indexed:", _collection().count())

from src.retrieve import retrieve, _load_rules, _markers

rules = _load_rules()
g107 = next(r for r in rules if r["rule_id"] == "G1-07")
print("G1-07 title:", g107["title"])
print("G1-07 pattern:", g107["pattern"])
print("G1-07 markers:", _markers(g107))

print()
for r in retrieve("你是谁吗？", k=8):
    print(r["rule_id"], "score", r["score"], "dense", r["dense"], "lex", r["lex"], "|", r["title"])

g125 = next(r for r in rules if r["rule_id"] == "G1-25")
print()
print("G1-25 title:", g125["title"])
print("G1-25 pattern:", g125["pattern"])
print("G1-25 markers:", _markers(g125))