# data/

| File | Content |
|---|---|
| `hsk_vocab.json` | 595 HSK 2.0 Level 1–3 words (the allowed vocabulary). Built by `build_vocab.py` |
| `hsk_grammar.json` | 59 grammar rule cards (pattern, explanation, examples). Built by `build_grammar.py`, which also checks that every example uses only allowed words |
| `answer_key.csv` | Frozen 50-item test set: 32 wrong, 14 correct, 4 should-abstain. Written before any model was run and never edited after seeing outputs. Columns include `is_correct`, `expected_behavior`, `pinyin_no_tone` |

## Sources and licences

| Item | Source | Licence / terms |
|---|---|---|
| Vocabulary | complete-hsk-vocabulary (drkameleon), HSK 2.0 L1–3 only | MIT |
| Definitions | CC-CEDICT (via the repo above) | CC BY-SA `<confirm>` |
| Grammar rules | Chinese Grammar Wiki (AllSet Learning), HSK 1/2/3 grammar pages, accessed 2026-10-03 | © AllSet Learning, non-commercial with attribution `<confirm CC terms>` |


Rule explanations and examples are my own wording. Answer-key sentences were written by me, none copied from the grammar cards.

## Caveats

- HSK 2.0 has no official grammar standard, so grammar levels are the wiki's estimates.
- The vocabulary list omits or levels some basic words differently, so `src/vocab_check.py` has a small allowlist (过, 等).
