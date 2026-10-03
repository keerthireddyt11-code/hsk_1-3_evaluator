# HSK 1-3 RAG Sentence Evaluator

## Setup
python -m venv .venv
.venv\Scripts\activate        (Windows)   |   source .venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
ollama pull qwen2.5:3b

## Run smoke test (Stage 1)
python smoke_test.py

## Data sources (fill in as you go)
- Vocab: complete-hsk-vocabulary (HSK standard: ___ , license: ___ , date: ___)
- Grammar: ___ (license: ___ , version: ___ , rule count: ___)