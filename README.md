# HSK 1-3 RAG Sentence Evaluator

## Setup
python -m venv .venv
.venv\Scripts\activate        (Windows)   |   source .venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
ollama pull qwen2.5:3b

## Run smoke test (Stage 1)
python smoke_test.py

## Data sources (fill in as you go)
- Vocab: complete-hsk-vocabulary (MIT), HSK 2.0 Levels 1-3 only (definitions via CC-CEDICT: check license/attribution)
- Grammar: ___ (license: ___ , version: ___ , rule count: ___)

## Data sources & licenses (update as you go)
| Item | Source | License / terms | Notes |
|---|---|---|---|
| Vocabulary (HSK 2.0 L1-3, 595 entries) | complete-hsk-vocabulary (drkameleon) | MIT | Old HSK lists originate from clem109/hsk-vocabulary |
| Word definitions | CC-CEDICT (via the repo above) | CC BY-SA (confirm) | Attribute when shown in the UI |
| Grammar point list + patterns (53 rules) | Chinese Grammar Wiki, AllSet Learning, HSK 1/2/3 grammar point pages, accessed 2026-10-03 | (c) AllSet Learning; non-commercial use with attribution (Creative Commons) - confirm exact CC terms (NC / SA) on the wiki | Explanations + examples are original wording |
| Not used | SN Mandarin 'HSK1 Grammar Cracker' (marketing booklet, no license stated); Marco Polo 'HSK 1-3 (2021 standard)' PDF (HSK 3.0, image-only) | - | Do not copy from these |

Caveats to state in the write-up: HSK 2.0 had no official grammar standard, so grammar levels are the wiki's estimates.
vocab_check has a small GRAMMAR_ALLOWLIST (過 only) because the vocab list files 过 under a higher level than the wiki does.