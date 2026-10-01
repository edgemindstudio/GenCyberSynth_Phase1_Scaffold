# M7.1 — Paper 1 P2 Consumer / Interface Regression Audit

**Status:** PASS

## Summary

- P2 surfaces: 13
- JSONL consumers: 12
- JSONL-only consumers: 10
- Filesystem-coupled JSONL consumers: 2
- Makefile safety-review surfaces: 1
- Checks: 26/26 passed

## Consumer matrix

| Consumer | Interface | Authority role | M7.2 runtime required |
|---|---|---|---|
| `scripts/jsonl_to_csv.py` | jsonl_only | consumer_only | no |
| `scripts/plots/_common.py` | jsonl_only | consumer_only | no |
| `scripts/plots/core/calibration_curves.py` | jsonl_only | consumer_only | no |
| `scripts/plots/core/pareto_downstream_vs_similarity.py` | jsonl_only | consumer_only | no |
| `scripts/plots/core/per_class_delta_f1.py` | jsonl_only | consumer_only | no |
| `scripts/plots/diversity/ms_ssim_hist.py` | jsonl_only | consumer_only | no |
| `scripts/plots/diversity/nn_distance_distrib.py` | jsonl_only | consumer_only | no |
| `scripts/plots/hparams/ablation_bars.py` | jsonl_only | consumer_only | no |
| `scripts/plots/hparams/parallel_coords.py` | jsonl_only | consumer_only | no |
| `scripts/plots/imbalance/class_counts_before_after.py` | jsonl_only | consumer_only | no |
| `scripts/plots/imbalance/simple_stats_sanity.py` | jsonl_plus_manifest_filesystem | consumer_only | yes |
| `scripts/plots/qual/class_triptychs.py` | jsonl_plus_manifest_filesystem | consumer_only | yes |

## Makefile handoff

Mixed historical and canonical orchestration, including paper1_build plus destructive clean targets. M7.3 must review safety and authority handoff before any modification.

## M7 handoff

- **M7.2:** Verify canonical-derived JSONL compatibility across the 12 P2 consumers, with explicit runtime checks for the two manifest/filesystem-coupled visualizers.
- **M7.3:** Review Makefile orchestration separately; do not modify it automatically.
