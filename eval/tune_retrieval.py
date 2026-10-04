"""Tune the dense/lexical weight and k on a SEPARATE set, with no LLM calls.

Tuning set = each rule card's own wrong_example sentence; the expected rule is that card's rule_id.
It never reads data/answer_key.csv.

    python eval/tune_retrieval.py                 # table of hit@1/3/5 for dense weights 0.0 ... 1.0
    python eval/tune_retrieval.py --misses 0.3    # also list the sentences missed at k=3 for that dense weight

Caveat: the dense side is flattered here, because the card's own embedded text often quotes its wrong sentence.
If the curve is flat, prefer the LOWER dense weight.
"""
import argparse, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from src import retrieve as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", type=float, help="dense weight for which to list k=3 misses")
    args = ap.parse_args()

    rules = R._load_rules()
    tune = [(r["rule_id"], r["wrong_example"]["zh"]) for r in rules if r.get("wrong_example")]
    print(f"Tuning set: {len(tune)} sentences (rules with no wrong_example are skipped)\n")

    cache = [(rid, s, R._lexical_scores(s), R._dense_scores(s)) for rid, s in tune]

    def rank(lex, dense, wd):
        sc = {x: round(wd * dense[x] + (1 - wd) * lex[x], 4) for x in lex}
        return sorted(sc, key=lambda x: (-sc[x], x))      # same tie-break as retrieve()

    def hit(wd, k):
        return 100 * sum(rid in rank(lex, dense, wd)[:k] for rid, _, lex, dense in cache) / len(cache)

    print("dense_w  lex_w   hit@1   hit@3   hit@5")
    for i in range(11):
        wd = i / 10
        print(f"{wd:5.1f}   {1 - wd:5.1f}   {hit(wd, 1):5.1f}   {hit(wd, 3):5.1f}   {hit(wd, 5):5.1f}")

    if args.misses is not None:
        print(f"\nMisses at k=3 with dense weight {args.misses}:")
        for rid, s, lex, dense in cache:
            top = rank(lex, dense, args.misses)[:3]
            if rid not in top:
                print(f"  wanted {rid}  got {top}  {s}")


if __name__ == "__main__":
    main()