"""Validate data/answer_key.csv BEFORE any evaluation run. Fix every ERROR; read every WARN.
Run from the project root:  python eval/check_answer_key.py

Columns: id, input, input_type, expected, accepted_corrections ('|' separated), error_type, rule_id, notes
  input_type: hanzi | pinyin_tone | pinyin_plain | mixed | gibberish
  expected:   correct | incorrect | abstain
"""
import csv, json, os, re, sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from src.vocab_check import check            # noqa: E402

KEY = os.path.join(ROOT, "data", "answer_key.csv")
RULES = {r["rule_id"]: r for r in json.load(open(os.path.join(ROOT, "data", "hsk_grammar.json"), encoding="utf-8"))["rules"]}
HAN = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
TONE = re.compile(r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]")
TYPES = {"hanzi", "pinyin_tone", "pinyin_plain", "mixed", "gibberish"}
EXPECTED = {"correct", "incorrect", "abstain"}
norm = lambda s: re.sub(r"\s+", "", s)

errors, warns = [], []
E = lambda i, m: errors.append(f"row {i}: {m}")
W = lambda i, m: warns.append(f"row {i}: {m}")

rows = list(csv.DictReader(open(KEY, encoding="utf-8-sig", newline="")))
# adapt the richer key format: derive 'expected' and normalise input_type
for _r in rows:
    if "expected" not in _r:
        _eb, _ic = _r.get("expected_behavior", "").strip(), _r.get("is_correct", "").strip().lower()
        _r["expected"] = "abstain" if _eb == "abstain" else ("correct" if _ic == "true" else "incorrect")
    if _r["input_type"].strip() == "pinyin_no_tone":
        _r["input_type"] = "pinyin_plain"
need = {"id", "input", "input_type", "expected", "accepted_corrections", "error_type", "rule_id", "notes"}
if not rows or not need <= set(rows[0]):
    sys.exit(f"ERROR: missing columns. Need: {sorted(need)}")

# every Chinese sentence that appears inside a rule card (input must never equal one of these -> leakage)
rule_sentences = set()
for r in RULES.values():
    rule_sentences.add(norm(r["correct_example"]["zh"]))
    if r.get("wrong_example"):
        rule_sentences.add(norm(r["wrong_example"]["zh"]))

ids = Counter(r["id"] for r in rows)
for n, r in enumerate(rows, start=2):
    i, text = r["id"], r["input"].strip()
    t, ex = r["input_type"].strip(), r["expected"].strip()
    corr = [c.strip() for c in r["accepted_corrections"].split("|") if c.strip()]
    rid = r["rule_id"].strip()
    if ids[i] > 1: E(n, f"duplicate id {i}")
    if not text: E(n, "empty input"); continue
    if t not in TYPES: E(n, f"input_type '{t}' not in {sorted(TYPES)}")
    if ex not in EXPECTED: E(n, f"expected '{ex}' not in {sorted(EXPECTED)}")

    # input_type must match what is really typed
    has_han, has_lat, has_tone = bool(HAN.search(text)), bool(LATIN.search(text)), bool(TONE.search(text))
    if t == "hanzi" and (not has_han or has_lat): E(n, "input_type=hanzi but the text is not pure Chinese characters")
    if t == "pinyin_tone" and (has_han or not has_tone): E(n, "input_type=pinyin_tone needs tone marks and no Hanzi")
    if t == "pinyin_plain" and (has_han or has_tone or not has_lat): E(n, "input_type=pinyin_plain must be letters only, no tone marks, no Hanzi")
    if t == "mixed" and not (has_han and has_lat): E(n, "input_type=mixed needs both Hanzi and letters")

    # leakage
    if norm(text) in rule_sentences:
        E(n, "input is identical to an example sentence inside a rule card (leakage) - write a fresh sentence")

    if ex == "incorrect":
        if not corr: E(n, "incorrect rows need at least one accepted correction")
        if any(norm(c) == norm(text) for c in corr): E(n, "an accepted correction equals the input")
        if len(corr) > 3: W(n, "more than 3 accepted corrections - are they all really right?")
        if not rid: E(n, "incorrect rows need a rule_id (used for the retrieval hit@k check)")
        elif rid not in RULES: E(n, f"rule_id {rid} not in hsk_grammar.json")
        elif RULES[rid].get("error_strength") != "clear" and RULES[rid]["kind"] != "do-not-flag":
            W(n, f"rule {rid} is '{RULES[rid].get('error_strength')}' - not a safe definite error")
        if not r["error_type"].strip(): W(n, "missing error_type")
    elif ex == "correct":
        if not corr: E(n, "correct rows: put the Hanzi form of the sentence in accepted_corrections")
        if rid and rid not in RULES: E(n, f"rule_id {rid} not in hsk_grammar.json")
    elif ex == "abstain":
        if corr: W(n, "abstain rows normally have no accepted_corrections")

    # vocabulary: everything the learner should be corrected TO must be inside HSK 2.0 L1-3
    for c in corr:
        bad = check(c)["out_of_scope_lenient"]
        if bad: W(n, f"accepted correction '{c}' has out-of-scope words {bad}")
    if t in ("hanzi", "mixed"):
        bad = check(text)["out_of_scope_lenient"]
        if bad: W(n, f"input has out-of-scope words {bad} (fine only if you are testing the soft warning)")

# overall mix
by_exp, by_type = Counter(r["expected"] for r in rows), Counter(r["input_type"] for r in rows)
print(f"\n{len(rows)} rows (target 50)")
print("expected  :", dict(by_exp), " target: incorrect 30-35, correct 10-15, abstain 3-6")
print("input_type:", dict(by_type), " target: all four kinds of input present")
if len(rows) != 50: W(0, f"{len(rows)} rows, target is 50")
if not 30 <= by_exp["incorrect"] <= 35: W(0, "incorrect count outside 30-35")
if not 10 <= by_exp["correct"] <= 15: W(0, "correct count outside 10-15")
if not 3 <= by_exp["abstain"] <= 6: W(0, "abstain count outside 3-6")
for k in ("pinyin_tone", "pinyin_plain", "mixed"):
    if by_type[k] < 3: W(0, f"fewer than 3 '{k}' rows")
used = Counter(r["rule_id"] for r in rows if r["expected"] == "incorrect" and r["rule_id"])
print(f"rules covered by incorrect rows: {len(used)} of {len(RULES)}  (most used: {used.most_common(3)})")

print("\n".join(["\nWARN  " + w for w in warns]) if warns else "\nno warnings")
if errors:
    print("\n".join(["\nERROR " + e for e in errors]))
    sys.exit(f"\n{len(errors)} error(s) - fix before running any evaluation")
print("\nOK: no errors")