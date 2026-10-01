# M7.2A — Canonical-derived JSONL Compatibility Audit

**Status:** PASS

- Dataset: `ustc_tfc2016`
- Seed: `42`
- Rows: 7
- Structural checks: 37/37 passed
- P2 consumers covered: 12

## Optional capability coverage

| Capability | Available rows | All 7 | Any |
|---|---:|---:|---:|
| `calibration` | 0/7 | no | no |
| `counts` | 7/7 | yes | yes |
| `downstream_utility` | 7/7 | yes | yes |
| `generative_similarity` | 7/7 | yes | yes |
| `manifest_path` | 7/7 | yes | yes |
| `ms_ssim_per_class` | 0/7 | no | no |
| `nn_distance` | 0/7 | no | no |
| `per_class_counts` | 0/7 | no | no |
| `per_class_f1` | 0/7 | no | no |

## Consumer compatibility

| Consumer | Structural contract | Optional capability status |
|---|---|---|
| `scripts/jsonl_to_csv.py` | PASS | counts=7/7; generative_similarity=7/7; downstream_utility=7/7 |
| `scripts/plots/_common.py` | PASS | none required |
| `scripts/plots/core/calibration_curves.py` | PASS | calibration=0/7 |
| `scripts/plots/core/pareto_downstream_vs_similarity.py` | PASS | generative_similarity=7/7; downstream_utility=7/7 |
| `scripts/plots/core/per_class_delta_f1.py` | PASS | per_class_f1=0/7 |
| `scripts/plots/diversity/ms_ssim_hist.py` | PASS | generative_similarity=7/7; ms_ssim_per_class=0/7 |
| `scripts/plots/diversity/nn_distance_distrib.py` | PASS | nn_distance=0/7 |
| `scripts/plots/hparams/ablation_bars.py` | PASS | downstream_utility=7/7 |
| `scripts/plots/hparams/parallel_coords.py` | PASS | generative_similarity=7/7; downstream_utility=7/7 |
| `scripts/plots/imbalance/class_counts_before_after.py` | PASS | per_class_counts=0/7 |
| `scripts/plots/imbalance/simple_stats_sanity.py` | PASS | manifest_path=7/7 |
| `scripts/plots/qual/class_triptychs.py` | PASS | manifest_path=7/7 |

## Authority statement

All structural rows derive from explicit dataset+seed canonical export authority. Optional metrics are reported but never invented or used to select authority.

> Missing optional metrics are capability gaps, not authority gaps. This audit does not invent metrics or substitute another historical execution.
