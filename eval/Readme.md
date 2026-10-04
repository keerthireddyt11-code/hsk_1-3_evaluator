# eval/

Runs the 50-item answer key (`data/answer_key.csv`) through three conditions with Qwen2.5-7B and scores the results.

| File / folder | Purpose |
|---|---|
| `run_eval.py` | Main script: runs all items through `plain`, `rag`, `rag_guardrail`, saves outputs and computes the metrics |
| `check_answer_key.py` | One-off check that the answer key is well-formed |
| `tune_retrieval.py` | Tunes retrieval weights using the rule cards' own examples (not the answer key) |
| `viol_report.py` | Lists vocabulary violations found in the saved outputs |
| `results/` | Outputs and metrics of the final run (includes `failures_*.csv`) |
| `results_before_fixes/` | Earlier run, kept for transparency (see note below) |

## Run

Set the OpenRouter variables first (see the main README), then:

```powershell
python eval/run_eval.py
```

About 150 model calls, a few cents on OpenRouter. Results are written to `results/`.

## Conditions

- `plain`: sentence and basic instruction only
- `rag`: plain plus the 3 retrieved grammar rules
- `rag_guardrail`: rag plus vocabulary check on the answer, retry once

## Metrics

| Metric | Definition | Better |
|---|---|---|
| Verdict accuracy | Correct "correct" / "needs a fix" calls, out of 46 | higher |
| Correction accuracy | Corrections matching the key, out of 31 wrong sentences (missed errors count as misses) | higher |
| False-accept rate | Wrong sentences marked correct, out of 32 | lower |
| Over-correction rate | Correct sentences marked wrong or changed, out of 14 | lower |
| Vocabulary violations | Answers using words outside HSK 1–3 | lower |
| Abstention recall | Refusals among the 4 items that should be refused | higher |

## Results (final run, n = 46)

| Metric | Plain | RAG | RAG + guardrail |
|---|---|---|---|
| Verdict accuracy | 58.7% | 80.4% | 80.4% |
| Correction accuracy | 22.6% | 45.2% | 45.2% |
| False-accept rate | 53.1% | 15.6% | 15.6% |
| Over-correction rate | 14.3% | 28.6% | 28.6% |
| Vocabulary violations | 13.0% | 17.4% | 8.7% |
| Abstention recall | 100% | 100% | 100% |

## Notes and limitations

- Fixes were made after reading failures on this same 50-item set: Pinyin conversion allowlist, a consistency check with one retry, and a stricter RAG prompt. Results may overstate performance on new sentences. Pre-fix results are in `results_before_fixes/`.
- Four likely-valid model answers (items 10, 17, 20, 25) failed the strict match, so correction accuracy is probably understated.
- Abstention recall is 100% by design (rule-based refusal).
- One run, one model family, handmade key.