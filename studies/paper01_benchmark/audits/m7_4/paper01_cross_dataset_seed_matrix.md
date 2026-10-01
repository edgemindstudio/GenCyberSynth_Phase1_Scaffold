# M7.4 — Paper 1 Cross-dataset / Cross-seed Regression Matrix

**Status:** PASS

## Summary

- Views: 6/6 passed
- Canonical experiments: 42/42
- Matrix checks: 6/6 passed
- JSONL structural checks: 222/222 passed
- Runtime filesystem checks: 348/348 passed

## Matrix

| Dataset | Seed | Experiments | JSONL | Runtime | Core capabilities |
|---|---:|---:|---:|---:|---|
| `ustc_tfc2016` | 42 | 7 | PASS | PASS | counts=7/7; generative_similarity=7/7; downstream_utility=7/7; manifest_path=7/7 |
| `ustc_tfc2016` | 43 | 7 | PASS | PASS | counts=7/7; generative_similarity=7/7; downstream_utility=7/7; manifest_path=7/7 |
| `ustc_tfc2016` | 44 | 7 | PASS | PASS | counts=7/7; generative_similarity=7/7; downstream_utility=7/7; manifest_path=7/7 |
| `cicmaldroid2020` | 42 | 7 | PASS | PASS | counts=7/7; generative_similarity=7/7; downstream_utility=7/7; manifest_path=7/7 |
| `cicmaldroid2020` | 43 | 7 | PASS | PASS | counts=7/7; generative_similarity=7/7; downstream_utility=7/7; manifest_path=7/7 |
| `cicmaldroid2020` | 44 | 7 | PASS | PASS | counts=7/7; generative_similarity=7/7; downstream_utility=7/7; manifest_path=7/7 |

## Authority statement

All 42 Paper 1 experiments are exercised through explicit dataset+seed canonical export authority and canonical-derived compatibility/runtime interfaces. Historical experiments are not rerun and filesystem fallback is not scientific authority.

> M7.4 validates the already-adjudicated evidence chain. It does not rerun generators/evaluators, rewrite summaries, or promote filesystem discovery into scientific authority.
