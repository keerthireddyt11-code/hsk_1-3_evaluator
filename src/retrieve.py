"""Stage 4, step 3: find the grammar rules most relevant to a learner sentence.

HYBRID retrieval over the 59 rule cards in data/hsk_grammar.json:
  dense  = sentence-embedding similarity (ChromaDB + all-MiniLM-L6-v2, local CPU)
  lexical = Chinese marker words (了, 没, 比, 把 ...) found in BOTH the sentence and the rule's pattern/title,
            weighted so rare markers count more, plus two small word classes (time words, numbers)
Why hybrid: MiniLM is English-centred, so Chinese input alone matches rules weakly. The lexical part carries the
Chinese-specific signal. Whether it is enough is something eval/ measures (hit@k) - swap the model with EMBED_MODEL.

Usage:
    from src.retrieve import retrieve, format_rules
    rules = retrieve("我去学校昨天。", k=3)
    prompt_block = format_rules(rules)
"""
import hashlib, json, math, os, re
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
GRAMMAR_PATH = os.path.join(HERE, "..", "data", "hsk_grammar.json")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(HERE, "..", "chroma_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAKE_EMBED = os.getenv("HSK_FAKE_EMBED") == "1"        # tests the plumbing only, no model download; NOT for real use
W_DENSE, W_LEX = 0.5, 0.5

_HAN = re.compile(r"[\u4e00-\u9fff]+")

# Small word classes for rules whose pattern has no Chinese marker (e.g. 'Subj. + Time + Verb').
CLASSES = [
    ({"今天", "明天", "昨天", "现在", "早上", "上午", "中午", "下午", "晚上", "星期", "年", "月", "号", "点", "时候", "以前", "以后"}, ["G1-10"]),
    ({"一", "二", "两", "三", "四", "五", "六", "七", "八", "九", "十", "几", "多少"}, ["G1-09", "G2-20"]),
]

QUESTION_WORDS = {"谁", "什么", "哪", "哪儿", "哪里", "几", "多少", "怎么", "怎么样", "为什么", "什么时候"}

@lru_cache(maxsize=1)
def _load_rules():
    return json.load(open(GRAMMAR_PATH, encoding="utf-8"))["rules"]

def _load_rules():
    return json.load(open(GRAMMAR_PATH, encoding="utf-8"))["rules"]


def _markers(rule):
    return set(_HAN.findall(rule["pattern"] + " " + rule["title"]))


@lru_cache(maxsize=1)
def _lexicon():
    rules = _load_rules()
    marks = {r["rule_id"]: _markers(r) for r in rules}
    df = {}
    for ms in marks.values():
        for m in ms:
            df[m] = df.get(m, 0) + 1
    return rules, marks, df


def _lexical_scores(text):
    rules, marks, df = _lexicon()
    scores = {}
    for r in rules:
        rid = r["rule_id"]
        ms = marks[rid]
        if "……" in (r["pattern"] + r["title"]) and len(ms) >= 2:
            s = 1.0 if all(m in text for m in ms) else 0.0      # frame rule: needs all its markers together
        else:
            s = sum(1.0 / df[m] for m in ms if m in text)       # rare markers count more
        scores[rid] = s
    for words, ids in CLASSES:
        if any(w in text for w in words):
            for rid in ids:
                scores[rid] += 1.0
    if "吗" in text and any(w in text for w in QUESTION_WORDS):
        scores["G1-07"] += 1.0
    return {rid: min(1.0, s) for rid, s in scores.items()}

# ----------------------------------------------------------------- dense part
def _embed(texts):
    if FAKE_EMBED:
        import numpy as np
        out = []
        for t in texts:
            v = np.zeros(128)
            for i in range(len(t) - 1):
                v[int(hashlib.md5(t[i:i + 2].encode()).hexdigest(), 16) % 128] += 1
            out.append((v / (np.linalg.norm(v) or 1)).tolist())
        return out
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def _doc(rule):
    # English summary + Chinese pattern + correct example, so both languages are in the vector
    return f"{rule['embed_text']} Example: {rule['correct_example']['zh']}"


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    rules = _load_rules()
    tag = hashlib.md5((open(GRAMMAR_PATH, "rb").read() + EMBED_MODEL.encode() + str(FAKE_EMBED).encode())).hexdigest()[:8]
    name = f"hsk_grammar_{tag}"
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    for c in client.list_collections():                     # drop stale indexes (older grammar file or model)
        cname = getattr(c, "name", c)
        if cname.startswith("hsk_grammar_") and cname != name:
            client.delete_collection(cname)
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    if col.count() != len(rules):                           # (re)build once
        docs = [_doc(r) for r in rules]
        col.upsert(ids=[r["rule_id"] for r in rules], documents=docs, embeddings=_embed(docs),
                   metadatas=[{"level": r["level"], "kind": r["kind"], "title": r["title"]} for r in rules])
    return col


def _dense_scores(text):
    col = _collection()
    res = col.query(query_embeddings=_embed([text]), n_results=col.count())
    sims = {rid: 1.0 - d for rid, d in zip(res["ids"][0], res["distances"][0])}     # cosine similarity
    lo, hi = min(sims.values()), max(sims.values())
    return {rid: (s - lo) / (hi - lo) if hi > lo else 0.0 for rid, s in sims.items()}  # 0-1 within this query


# ----------------------------------------------------------------- public API
def retrieve(text, k=3, use_dense=True):
    """Return the top-k rule cards for a sentence, each with score, dense and lex parts (kept for logging/eval)."""
    rules = _load_rules()
    lex = _lexical_scores(text)
    dense = _dense_scores(text) if use_dense else {r["rule_id"]: 0.0 for r in rules}
    wd, wl = (W_DENSE, W_LEX) if use_dense else (0.0, 1.0)
    out = []
    for r in rules:
        rid = r["rule_id"]
        out.append({**r, "score": round(wd * dense[rid] + wl * lex[rid], 4),
                    "dense": round(dense[rid], 4), "lex": round(lex[rid], 4)})
    out.sort(key=lambda x: (-x["score"], x["rule_id"]))
    return out[:k]


def format_rules(rules):
    """Turn retrieved rule cards into a text block for the LLM prompt."""
    lines = []
    for r in rules:
        lines.append(f"Rule {r['rule_id']} (HSK {r['level']}): {r['title']}")
        lines.append(f"  Pattern: {r['pattern']}")
        lines.append(f"  Explanation: {r['explanation']}")
        lines.append(f"  Correct: {r['correct_example']['zh']} ({r['correct_example']['pinyin']}) = {r['correct_example']['en']}")
        w = r.get("wrong_example")
        if w:
            lines.append(f"  Common mistake: {w['zh']} - {w['why']}")
            if r.get("error_strength") in ("non-canonical", "standard-written"):
                lines.append("  Note: this mistake is understandable; say it is 'not the usual way', not 'wrong'.")
        else:
            lines.append("  Do NOT mark sentences like this as wrong.")
        lines.append("")
    return "\n".join(lines).strip()
    