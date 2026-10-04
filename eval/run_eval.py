"""Stage 5: run the 50-item answer key through the three conditions and score them.

    python eval/run_eval.py --retrieval-only          # free, no LLM: is the right rule retrieved? (run this FIRST)
    python eval/run_eval.py --limit 5                 # quick trial of the whole loop on 5 items
    python eval/run_eval.py                           # full run: plain, rag, rag_guardrail  (150 calls)
    python eval/run_eval.py --score-only              # re-score saved outputs without calling the model
    python eval/run_eval.py --fresh                   # ignore saved outputs and start over

Settings come from the environment, same as the app: LLM_BACKEND, LLM_MODEL, OPENROUTER_API_KEY. temperature is 0 in llm.py.
Saved per run (eval/results/): raw_<cond>_<model>.jsonl (every output, resumable), failures_<cond>_<model>.csv,
review_<cond>_<model>.csv (fill explanation_ok = 1/0 by hand for the error-type number), summary_<model>.json / .md
"""
import argparse, csv, json, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
KEY = os.path.join(ROOT, "data", "answer_key.csv")
OUT = os.path.join(ROOT, "eval", "results")
CONDS = ["plain", "rag", "rag_guardrail"]
norm = lambda s: re.sub(r"[\s。，！？、,.!?;；:：\"“”'‘’]", "", s or "")
pct = lambda a, b: round(100 * a / b, 1) if b else None


def load_key():
    rows = []
    for r in csv.DictReader(open(KEY, encoding="utf-8-sig", newline="")):
        r["accepted"] = [c.strip() for c in r["accepted_corrections"].split("|") if c.strip()]
        r["key_correct"] = {"true": True, "false": False}.get(r["is_correct"].strip().lower())
        rows.append(r)
    return rows


def tag():
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"{os.getenv('LLM_BACKEND', 'ollama')}_{os.getenv('LLM_MODEL', 'qwen2.5:3b')}")


# ------------------------------------------------------------------ running
def run_condition(cond, rows, fresh):
    path = os.path.join(OUT, f"raw_{cond}_{tag()}.jsonl")
    done = {}
    if os.path.exists(path) and not fresh:
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            done[d["id"]] = d
    elif os.path.exists(path):
        os.remove(path)
    from src import pipeline
    with open(path, "a", encoding="utf-8") as f:
        for n, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            try:
                res = pipeline.assess(r["input"], cond)
            except Exception as e:                                   # keep going; this item counts as unanswered
                res = {"status": "error", "message": f"{type(e).__name__}: {e}", "assessment": None, "seconds": 0.0}
            res["id"] = r["id"]
            done[r["id"]] = res
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{cond}] {n}/{len(rows)} id={r['id']} {res['status']} {res.get('seconds', 0):.1f}s", flush=True)
    return done


# ------------------------------------------------------------------ scoring
def score(cond, rows, results):
    m = dict(n=0, answered=0, unanswered_errors=0, verdict_ok=0, n_incorrect=0, false_accept=0, n_correct=0,
             over_correction=0, n_match=0, correction_ok=0, abstain_expected=0, abstain_ok=0, false_abstain=0,
             viol_before=0, viol_after=0, soft_n=0, soft_ok=0, hit1=0, hit3=0, n_rule=0, secs=0.0, item_ok=0)
    fails, review = [], []
    for r in rows:
        res = results.get(r["id"])
        if res is None:
            continue
        st, a = res["status"], res.get("assessment")
        abst = st == "abstained"
        if r["expected_behavior"] == "abstain":
            m["abstain_expected"] += 1
            m["abstain_ok"] += abst
            m["item_ok"] += abst
            if not abst:
                fails.append((r, res, "should_have_abstained"))
            continue
        m["n"] += 1
        if abst:
            m["false_abstain"] += 1
            fails.append((r, res, "false_abstain"))
            continue
        if st != "ok" or a is None:
            m["unanswered_errors"] += 1
            fails.append((r, res, "no_valid_output"))
            continue
        m["answered"] += 1
        m["secs"] += res.get("seconds", 0)
        if res.get("violations_before"):
            m["viol_before"] += 1
        if res.get("violations_after"):
            m["viol_after"] += 1
        pred = a["is_correct"]
        ok_verdict = pred == r["key_correct"]
        m["verdict_ok"] += ok_verdict
        item_ok = ok_verdict
        reason = None
        if r["key_correct"] is False:
            m["n_incorrect"] += 1
            if r["correction_scoring"] == "match":
                m["n_match"] += 1          # every wrong sentence counts, so missing it (false accept) lowers correction accuracy
            if pred:
                m["false_accept"] += 1
                reason = "false_accept"
            elif r["correction_scoring"] == "match":
                good = norm(a["corrected_hanzi"]) in {norm(c) for c in r["accepted"]}
                m["correction_ok"] += good
                if not good:
                    reason, item_ok = "wrong_correction", False
            if not pred:
                review.append({"id": r["id"], "input": r["input"], "key_error_type": r["error_type"], "key_rule": r["rule_id"],
                               "model_error_type": a["error_type"], "model_correction": a["corrected_hanzi"],
                               "accepted": r["accepted_corrections"], "explanation": a["explanation"], "explanation_ok": ""})
            if r["rule_id"] and cond != "plain":
                ids = [x["rule_id"] for x in res.get("retrieved", [])]
                m["n_rule"] += 1
                m["hit1"] += r["rule_id"] in ids[:1]
                m["hit3"] += r["rule_id"] in ids[:3]
        elif r["key_correct"] is True:
            m["n_correct"] += 1
            if not pred:
                m["over_correction"] += 1
                reason = "over_correction"
        if not ok_verdict and reason is None:
            reason = "wrong_verdict"
        m["item_ok"] += item_ok
        if r["expected_behavior"] == "soft_warning":
            m["soft_n"] += 1
            words = {w for s in res.get("soft_warnings", []) for w in s["words"]} | set(a["out_of_scope_words"])
            m["soft_ok"] += r["expected_out_of_scope"] in words
        if reason:
            fails.append((r, res, reason))
    s = {
        "condition": cond, "model": tag(), "items_scored": m["n"] + m["abstain_expected"],
        "verdict_accuracy_%": pct(m["verdict_ok"], m["n"]),
        "verdict_accuracy_answered_only_%": pct(m["verdict_ok"], m["answered"]),
        "correction_accuracy_%": pct(m["correction_ok"], m["n_match"]),
        "false_accept_rate_%": pct(m["false_accept"], m["n_incorrect"]),
        "over_correction_rate_%": pct(m["over_correction"], m["n_correct"]),
        "overall_item_accuracy_%": pct(m["item_ok"], m["n"] + m["abstain_expected"]),
        "vocab_violation_before_%": pct(m["viol_before"], m["answered"]),
        "vocab_violation_after_%": pct(m["viol_after"], m["answered"]),
        "rule_hit@1_%": pct(m["hit1"], m["n_rule"]) if cond != "plain" else None,
        "rule_hit@3_%": pct(m["hit3"], m["n_rule"]) if cond != "plain" else None,
        "abstain_recall_%": pct(m["abstain_ok"], m["abstain_expected"]),
        "false_abstain_%": pct(m["false_abstain"], m["n"]),
        "soft_warning_recall_%": pct(m["soft_ok"], m["soft_n"]),
        "no_valid_output": m["unanswered_errors"],
        "avg_seconds_per_item": round(m["secs"] / m["answered"], 1) if m["answered"] else None,
    }
    # manual error-type number, if the review file has been filled in
    rp = os.path.join(OUT, f"review_{cond}_{tag()}.csv")
    if os.path.exists(rp):
        graded = [x for x in csv.DictReader(open(rp, encoding="utf-8-sig")) if x["explanation_ok"].strip() in ("0", "1")]
        if graded:
            s["error_type_accuracy_manual_%"] = pct(sum(x["explanation_ok"].strip() == "1" for x in graded), len(graded))
            s["error_type_graded_n"] = len(graded)
    return s, fails, review


def save_extras(cond, fails, review):
    with open(os.path.join(OUT, f"failures_{cond}_{tag()}.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "reason", "input", "key_verdict", "accepted", "key_error_type", "key_rule", "model_verdict",
                    "model_correction", "model_error_type", "explanation", "retrieved_rules", "status"])
        for r, res, why in fails:
            a = res.get("assessment") or {}
            w.writerow([r["id"], why, r["input"], r["is_correct"], r["accepted_corrections"], r["error_type"], r["rule_id"],
                        a.get("is_correct"), a.get("corrected_hanzi"), a.get("error_type"), a.get("explanation"),
                        " ".join(x["rule_id"] for x in res.get("retrieved", [])), res["status"]])
    rp = os.path.join(OUT, f"review_{cond}_{tag()}.csv")
    if not os.path.exists(rp) and review:                 # never overwrite hand-graded work
        with open(rp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(review[0]))
            w.writeheader()
            w.writerows(review)


def retrieval_only(rows):
    from src import preprocess as pre, retrieve
    print("Retrieval check (no LLM): is the key's rule among the top-k?\n")
    stats = {"hybrid": [0, 0, 0], "lexical-only": [0, 0, 0]}
    n, misses = 0, []
    for r in rows:
        if r["key_correct"] is not False or not r["rule_id"]:
            continue
        p = pre.preprocess(r["input"])
        if p["abstain"]:
            continue
        n += 1
        for name, dense in (("hybrid", True), ("lexical-only", False)):
            ids = [x["rule_id"] for x in retrieve.retrieve(p["hanzi_candidate"], k=5, use_dense=dense)]
            for j, k in enumerate((1, 3, 5)):
                stats[name][j] += r["rule_id"] in ids[:k]
            if name == "hybrid" and r["rule_id"] not in ids[:3]:
                misses.append((r["id"], r["rule_id"], ids[:3], r["input"]))
    for name, (a, b, c) in stats.items():
        print(f"{name:13s} hit@1 {pct(a, n)}%   hit@3 {pct(b, n)}%   hit@5 {pct(c, n)}%   (n={n})")
    print("\nHybrid misses at k=3 (id, wanted rule, got, input):")
    for x in misses:
        print("  ", x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=CONDS, choices=CONDS)
    ap.add_argument("--limit", type=int, help="only the first N rows")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = load_key()
    if args.retrieval_only:
        return retrieval_only(rows)
    if args.limit:
        rows = rows[:args.limit]
    summaries = []
    for cond in args.conditions:
        if args.score_only:
            p = os.path.join(OUT, f"raw_{cond}_{tag()}.jsonl")
            if not os.path.exists(p):
                print(f"no saved run for {cond}")
                continue
            results = {d["id"]: d for d in map(json.loads, open(p, encoding="utf-8"))}
        else:
            print(f"== {cond} ({tag()})")
            results = run_condition(cond, rows, args.fresh)
        s, fails, review = score(cond, rows, results)
        save_extras(cond, fails, review)
        summaries.append(s)
    if not summaries:
        return
    json.dump(summaries, open(os.path.join(OUT, f"summary_{tag()}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    keys = [k for k in summaries[0] if k not in ("condition", "model")]
    lines = ["| metric | " + " | ".join(s["condition"] for s in summaries) + " |", "|---|" + "---|" * len(summaries)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(s.get(k, "")) for s in summaries) + " |")
    table = "\n".join(lines)
    open(os.path.join(OUT, f"summary_{tag()}.md"), "w", encoding="utf-8").write(f"Model: {tag()}\n\n{table}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()