# HSK 1–3 Sentence Assessor

Checks a beginner Mandarin sentence (Hanzi or Pinyin) against HSK 1–3 grammar rules and explains the mistake in simple English. Project report and demo link are submitted separately.

## Where to look

| Folder / file | What it is | Details |
|---|---|---|
| `src/` | Product logic (pipeline, retrieval, LLM call, guardrail) | [src/README.md](src/README.md) |
| `data/` | Vocabulary, grammar rule cards, 50-item answer key | [data/README.md](data/README.md) |
| `eval/` | Evaluation scripts and saved results | [eval/README.md](eval/README.md) |
| `app.py` | Streamlit app | |
| `build_vocab.py` | One-off: builds `data/hsk_vocab.json` | |
| `build_grammar.py` | One-off: builds `data/hsk_grammar.json` and checks its examples | |
| `smoke_test.py` | Quick test that the model connection works | |
| `requirements.txt` | Python packages | |

Not committed (generated or secret): `.venv/`, `chroma_db/`, `__pycache__/`, `.streamlit/secrets.toml`.

## Setup (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set the backend and your OpenRouter key in the same terminal (see the top of `src/llm.py` for the exact variable names):

```powershell
$env:LLM_BACKEND="openrouter"
$env:OPENROUTER_API_KEY="your-openrouter-key"
$env:LLM_MODEL="qwen/qwen-2.5-7b-instruct"
```

Mac / Linux: use `export NAME="value"` instead of `$env:NAME="value"`.

Check the `model` line printed at the start of each run to confirm which model answered.

## Run

Command line: `python -m src.pipeline "<sentence>" <condition>` where condition is `plain`, `rag` or `rag_guardrail`.

Examples (`rag_guardrail`):

```powershell
# 1. Hanzi, wrong (word order): expect "needs a fix" -> 我昨天去商店
python -m src.pipeline "我去商店昨天。" rag_guardrail

# 2. Hanzi, correct: expect "correct", sentence unchanged
python -m src.pipeline "请等一下。" rag_guardrail

# 3. Pinyin only, wrong: converted to Hanzi first, then checked
python -m src.pipeline "wo qu shangdian zuotian" rag_guardrail

# 4. Hanzi + Pinyin mixed, wrong word order
python -m src.pipeline "我 qu 商店 昨天" rag_guardrail

# 5. Gibberish: expect a refusal (abstain)
python -m src.pipeline "asdfgh qwerty zxcv" rag_guardrail
```

Compare conditions on the same sentence by changing the last word, e.g. `plain` vs `rag`.

Streamlit app. `.streamlit/secrets.toml` is not in the repo (it holds the API key), so create it first, then run the app.

Create `.streamlit/secrets.toml` with:

```toml
LLM_BACKEND = "openrouter"
LLM_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_API_KEY = "your-openrouter-key"
```

Run:

```powershell
python -m streamlit run app.py
```

Full evaluation (see `eval/README.md`):

```powershell
python eval/run_eval.py
```