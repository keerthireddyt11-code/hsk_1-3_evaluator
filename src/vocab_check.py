"""Deterministic HSK vocab check (set membership) - the GUARDRAIL, not a metric.
Vocabulary standard: HSK 2.0 Levels 1-3 (~600 words), from data/hsk_vocab.json.

Method: Jieba splits text into tokens; any token not in the vocab set is broken down
by greedy longest-match against the vocab. Whatever cannot be matched is reported as
out-of-scope. (Known limit: a word built from several in-scope words, e.g. a
compound, passes. Mention this in your limitations.)
"""
import json, os, re
import jieba

_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hsk_vocab.json")

VOCAB = {w["hanzi"] for w in json.load(open(_PATH, encoding="utf-8"))}
# Function words the grammar source teaches at HSK 1-3 but that the vocab list files under a higher
# level (e.g. 过 is 'old-4' in complete-hsk-vocabulary). Keep this list SHORT and document it in the README.
GRAMMAR_ALLOWLIST = {"过"}
VOCAB |= GRAMMAR_ALLOWLIST
VOCAB_CHARS = {c for w in VOCAB for c in w}   # every character that appears in some HSK 2.0 L1-3 word
_MAXLEN = max(len(w) for w in VOCAB)

for w in VOCAB:                      # keep HSK words whole when Jieba segments
    jieba.add_word(w, freq=1_000_000)

_HAN = re.compile(r"[\u4e00-\u9fff]+")


def _unmatched(token: str) -> list:
    """Greedy longest-match decomposition; return runs of characters not covered by vocab."""
    runs, cur, i = [], "", 0
    while i < len(token):
        for n in range(min(_MAXLEN, len(token) - i), 0, -1):
            if token[i:i + n] in VOCAB:
                if cur:
                    runs.append(cur); cur = ""
                i += n
                break
        else:
            cur += token[i]; i += 1
    if cur:
        runs.append(cur)
    return runs


def check(text: str) -> dict:
    """Two strictness levels (decide in the pipeline which one triggers what):
    out_of_scope          STRICT : token not in the word list (e.g. 说 or 吃饭 are flagged because the list stores 说话 / 吃 + 饭 differently)
    out_of_scope_lenient  LENIENT: token containing a character that appears in NO HSK 2.0 L1-3 word (e.g. 酷, 复杂, 工程师)
    """
    toks, parts, lenient = [], [], []
    for seg in _HAN.findall(text):
        for tok in jieba.lcut(seg, HMM=False):
            if tok not in VOCAB:
                u = _unmatched(tok)
                if u:
                    toks.append(tok)
                    parts.extend(u)
                    if any(c not in VOCAB_CHARS for run in u for c in run):
                        lenient.append(tok)
    dedup = lambda xs: list(dict.fromkeys(xs))
    return {"out_of_scope": dedup(toks), "out_of_scope_parts": dedup(parts), "out_of_scope_lenient": dedup(lenient)}