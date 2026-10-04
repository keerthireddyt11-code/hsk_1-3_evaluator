# src/ — product logic

Read in this order. Each file starts with a docstring describing its role.

| File | Role |
|---|---|
| `schema.py` | Shape of the model's JSON answer (verdict, correction, explanation, …) |
| `preprocess.py` | Detects Hanzi / Pinyin / mixed, converts Pinyin to Hanzi, refuses empty, gibberish, injection-style and too-advanced input |
| `retrieve.py` | Finds the top-3 grammar rules: ChromaDB (MiniLM embeddings) plus Chinese marker-word matching |
| `llm.py` | Calls Ollama or OpenRouter, validates the JSON answer, retries once |
| `vocab_check.py` | Flags words outside HSK 1–3 (the guardrail); small `GRAMMAR_ALLOWLIST` (过, 等) |
| `pipeline.py` | Chains all steps. Conditions: `plain`, `rag`, `rag_guardrail`. Called by `app.py` and `eval/run_eval.py` |
| `__init__.py` | Empty; makes `src` importable |

Flow: `preprocess` → `retrieve` → `llm` (+ `schema`) → consistency check → `vocab_check` → result.