"""Stage 4: the whole flow in one call.   pipeline.assess(text, condition="rag_guardrail")

  preprocess (rules) -> vocab check on the input -> retrieve rules -> prompt -> LLM -> output guardrail -> result

Conditions (these are the three evaluation conditions):
  plain          : original text + base system prompt. No rules, no Pinyin->Hanzi hint, no guardrail.
  rag            : + top-k grammar rules + Hanzi hint for Pinyin input.
  rag_guardrail  : + vocab check on the model's output; regenerate once; then soft warning.
Rule-based hard abstention (empty / too long / not Pinyin / injection / too advanced) applies to ALL conditions,
because it is not what the conditions compare. Pass hard_abstain=False to turn it off.

    python -m src.pipeline "我去商店昨天。" rag_guardrail
"""
import json, re, sys
from src import llm, preprocess as pre, vocab_check

CONDITIONS = ("plain", "rag", "rag_guardrail")
TOO_ADVANCED_MIN = 3        # this many out-of-scope words IN THE INPUT -> hard abstain (1-2 -> soft warning only)
TOP_K = 3

RAG_ADDENDUM = """

Grammar rules that may apply to this sentence (use a rule ONLY if it really applies):
{rules}

Extra instructions:
- If a rule says "Do NOT mark sentences like this as wrong", do not mark such a sentence wrong.
- If you use a rule, mention its idea in your explanation in simple English.
- Write the explanation and the correction using only simple HSK 1-3 words.
- Before you answer, check that your corrected sentence really follows the rule, and that corrected_hanzi contains only Chinese characters.
- Rules are hints, call it wrong only if a rule clearly applies or it is clearly unnatural" line"""

RETRY_ADDENDUM = """

IMPORTANT: your previous answer used words outside HSK 1-3: {bad}.
Answer again. Use only very simple HSK 1-3 words in corrected_hanzi and in the explanation. Keep the same JSON format."""


FIX_ADDENDUM = """

IMPORTANT: your previous answer had a problem: {problems}
Answer again with the same JSON format and fix this problem."""

_NORM = re.compile(r"[\s。，！？、,.!?;；:：\"“”'‘’]")
_LATIN = re.compile(r"[A-Za-z]")


def _same(x, y):
    return _NORM.sub("", x or "") == _NORM.sub("", y or "")


def _problems(a, text, p):
    """Cheap consistency checks on the model's answer (verdict vs correction, stray English letters)."""
    out, cand = [], p["hanzi_candidate"]
    if _LATIN.search(a.corrected_hanzi):
        out.append("corrected_hanzi must contain only Chinese characters and Chinese punctuation, no English or Pinyin letters.")
    same = _same(a.corrected_hanzi, text) or _same(a.corrected_hanzi, cand)
    if not a.is_correct and same:
        out.append("you said the sentence is wrong, but corrected_hanzi is the same as the learner's sentence. "
                   "Give a real correction, or set is_correct to true.")
    if a.is_correct and not same and not p["unresolved"] and not p["ambiguous"]:
        out.append("you said the sentence is correct, but corrected_hanzi is different from the learner's sentence. "
                   "If a change is needed, is_correct must be false.")
    return out


def _violations(a, candidate):
    """Out-of-scope words the MODEL introduced (words already in the learner's own sentence are not counted)."""
    mine = vocab_check.check(a.corrected_hanzi + "\n" + a.explanation)["out_of_scope_lenient"]
    own = set(vocab_check.check(candidate)["out_of_scope_lenient"])
    return [w for w in mine if w not in own]


def assess(text, condition="rag_guardrail", k=TOP_K, hard_abstain=True):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    res = {"input": text, "condition": condition, "model": llm.MODEL, "status": "ok", "abstain_reason": None,
           "message": None, "assessment": None, "soft_warnings": [], "retrieved": [], "regenerated": False,
           "violations_before": [], "violations_after": [], "validation_problems": [], "validation_retry": False,
           "seconds": 0.0, "raw": ""}

    p = pre.preprocess(text)
    res["input_type"], res["hanzi_candidate"] = p["input_type"], p["hanzi_candidate"]
    res["unresolved_pinyin"], res["ambiguous_pinyin"] = p["unresolved"], p["ambiguous"]

    def abstain(reason, msg):
        res.update(status="abstained", abstain_reason=reason, message=msg)
        return res

    if hard_abstain and p["abstain"]:
        return abstain(p["abstain_reason"], p["message"])
    if p["abstain"]:                                  # abstention switched off but nothing to send
        if not p["input_type"] or p["input_type"] == "none":
            return abstain(p["abstain_reason"], p["message"])

    candidate = p["hanzi_candidate"]
    in_scope_check = vocab_check.check(candidate)["out_of_scope_lenient"]
    if hard_abstain and len(in_scope_check) >= TOO_ADVANCED_MIN:
        return abstain("too_advanced", "This sentence uses many words beyond HSK 1-3, so I can't check it reliably. "
                                       "Please try a simpler sentence.")
    if in_scope_check:
        res["soft_warnings"].append({"type": "input_out_of_scope", "words": in_scope_check,
                                     "message": f"These words are outside HSK 1-3: {', '.join(in_scope_check)}. "
                                                "They are fine to use, but I will still check your grammar."})

    # ---- build the prompt for this condition
    user_text, system_prompt = text, llm.SYSTEM_PROMPT
    if condition != "plain":
        from src import retrieve                      # lazy: 'plain' never loads ChromaDB / the embedder
        rules = retrieve.retrieve(candidate, k=k)
        res["retrieved"] = [{"rule_id": r["rule_id"], "score": r["score"], "dense": r["dense"], "lex": r["lex"]} for r in rules]
        system_prompt += RAG_ADDENDUM.format(rules=retrieve.format_rules(rules))
        if p["input_type"] != "hanzi":
            user_text = f"{text}\n(Possible Hanzi reading from a simple converter, may be wrong: {candidate})"

    # ---- LLM call
    a, raw, secs = llm.assess(user_text, system_prompt)
    res["seconds"] += secs
    res["raw"] = raw
    if a is None:
        res.update(status="failed", message="The model did not return a valid answer. Please try again.")
        return res

    # ---- consistency check (all conditions): one retry if the answer contradicts itself
    probs = _problems(a, text, p)
    res["validation_problems"] = probs
    if probs:
        a2, raw2, secs2 = llm.assess(user_text, system_prompt + FIX_ADDENDUM.format(problems=" ".join(probs)))
        res["seconds"] += secs2
        res["validation_retry"] = True
        if a2 is not None:
            a, res["raw"] = a2, raw2

    # ---- output guardrail (vocab bounds are enforced in code, not measured by the model)
    res["violations_before"] = _violations(a, candidate)
    res["violations_after"] = res["violations_before"]
    if condition == "rag_guardrail" and res["violations_before"]:
        res["regenerated"] = True
        a2, raw2, secs2 = llm.assess(user_text, system_prompt + RETRY_ADDENDUM.format(bad=", ".join(res["violations_before"])))
        res["seconds"] += secs2
        if a2 is not None:
            a, res["raw"] = a2, raw2
            res["violations_after"] = _violations(a, candidate)
        if res["violations_after"]:
            res["soft_warnings"].append({"type": "output_out_of_scope", "words": res["violations_after"],
                                         "message": f"The answer uses words beyond HSK 1-3: {', '.join(res['violations_after'])}."})

    res["inconsistent"] = bool(_problems(a, text, p))        # still contradicting itself after the retry (must come BEFORE the override)
    res["model_verdict"] = a.is_correct                      # what the model originally said, kept for the error analysis
    if not (p["unresolved"] or p["ambiguous"]):              # unclear Pinyin: keep the model's own verdict
        changed = not (_same(a.corrected_hanzi, text) or _same(a.corrected_hanzi, candidate))
        a.is_correct = not changed

    d = a.model_dump()
    d["out_of_scope_words"] = list(dict.fromkeys(d["out_of_scope_words"] + res["violations_after"]))
    res["assessment"] = d
    res["rule_used"] = res["retrieved"][0]["rule_id"] if res["retrieved"] else None
    return res


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "我去商店昨天。"
    c = sys.argv[2] if len(sys.argv) > 2 else "rag_guardrail"
    r = assess(t, c)
    r.pop("raw", None)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"\n[{r['model']}] {r['seconds']:.1f} s for this request")