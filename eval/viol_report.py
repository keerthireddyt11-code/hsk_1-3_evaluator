"""Which words cause the vocab-violation count, and where do they appear?

    python eval/viol_report.py eval/results_before_fixes eval/results

Reads raw_*.jsonl in each folder (no model calls). Uses violations_before, the count before the guardrail.
"""
import glob, json, os, sys
from collections import Counter

dirs = sys.argv[1:] or ["eval/results"]
for d in dirs:
    for path in sorted(glob.glob(os.path.join(d, "raw_*.jsonl"))):
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        words, n_items, answered = Counter(), 0, 0
        print(f"\n== {path}")
        for r in rows:
            a = r.get("assessment")
            if r.get("status") != "ok" or not a:
                continue
            answered += 1
            v = r.get("violations_before") or []
            if not v:
                continue
            n_items += 1
            for w in v:
                words[w] += 1
                where = [name for name, txt in (("correction", a["corrected_hanzi"]), ("explanation", a["explanation"])) if w in txt]
                print(f"  id={r['id']:>3}  {w}  in {'+'.join(where) or '?'}   correction: {a['corrected_hanzi']}")
        print(f"  -> {n_items}/{answered} answered items had a violation. Most common: {words.most_common(10)}")