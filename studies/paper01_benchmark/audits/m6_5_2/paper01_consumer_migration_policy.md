# Paper 1 Consumer Migration Policy — M6.5.2

- Audit version: `M6.5.2-v1.0`
- Study: `paper01_benchmark`
- Repository HEAD: `d88a717465f175a6e3025eb95d1274bab21a0e04`
- Source audit: `M6.5.1-v1.0`
- Primary policy records: **59**

## Policy principles

- `POLICY_DECISION != CODE_MODIFICATION`
- `SCIENTIFIC_EXPERIMENT_IDENTITY != HISTORICAL_EXECUTION_IDENTITY`
- `latest.json != AUTHORITATIVE_EVIDENCE`
- `historical_artifacts_modified=NO`
- `eval/runner.py remains protected`
- `configuration dependency != executable migration target`
- `canonical-derived compatibility export may preserve legacy consumer contracts`

## Treatment census

| Treatment | Count |
|---|---:|
| `ALREADY_CANONICAL` | 4 |
| `CONFIGURATION_ONLY` | 22 |
| `GOVERNANCE_TOOL` | 1 |
| `MIGRATE_TO_CANONICAL_INTERFACE` | 7 |
| `PRESERVE_HISTORICAL` | 7 |
| `REVIEW_BEFORE_MIGRATION` | 1 |
| `USE_CANONICAL_DERIVED_EXPORT` | 12 |
| `WRAP_WITH_COMPATIBILITY_LAYER` | 5 |

## Priority census

| Priority | Count |
|---|---:|
| `NONE` | 34 |
| `P0` | 2 |
| `P1` | 10 |
| `P2` | 13 |

## Decisions

| Path | Source class | Treatment | Priority | Target |
|---|---|---|---|---|
| `Makefile` | `historical_evidence_mutator_or_producer` | `REVIEW_BEFORE_MIGRATION` | `P2` | `none` |
| `configs/paper1_cicmaldroid.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `configs/paper1_config.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `configs/paper1_final.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `configs/paper1_final_rerun.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `configs/paper1_real.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `configs/paper1_smoke.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `eval/runner.py` | `historical_pipeline` | `PRESERVE_HISTORICAL` | `NONE` | `none` |
| `papers/paper2_conditional_generation_done_right/configs/paper2_cicmaldroid_fakeclass_seed42.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper2_conditional_generation_done_right/configs/paper2_cicmaldroid_fakeclass_seed43.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper2_conditional_generation_done_right/configs/paper2_cicmaldroid_fakeclass_seed44.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper2_conditional_generation_done_right/configs/paper2_cicmaldroid_fakeclass_smoke_seed42.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_class_repair_topk5000_b2000_s42.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_class_repair_topk5000_b2000_s43.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_class_repair_topk5000_b2000_s44.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_confidence_ranked_topk1000_b2000_s42.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_confidence_ranked_topk1000_b2000_s43.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_confidence_ranked_topk1000_b2000_s44.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_keep_all_b2000_s42.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_keep_all_b2000_s43.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_keep_all_b2000_s44.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_realonly_balanced_s42.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_realonly_balanced_s43.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_realonly_balanced_s44.yaml` | `configuration_dependency` | `CONFIGURATION_ONLY` | `NONE` | `none` |
| `scripts/backfill_kid_and_downstream.py` | `historical_evidence_mutator_or_producer` | `PRESERVE_HISTORICAL` | `NONE` | `none` |
| `scripts/backfill_ms_ssim_per_class.py` | `historical_evidence_mutator_or_producer` | `PRESERVE_HISTORICAL` | `NONE` | `none` |
| `scripts/build_jsonl.sh` | `legacy_or_direct_consumer` | `WRAP_WITH_COMPATIBILITY_LAYER` | `P1` | `proposed:trustforge.paper01_exports` |
| `scripts/eval_write_summary.py` | `historical_evidence_mutator_or_producer` | `PRESERVE_HISTORICAL` | `NONE` | `none` |
| `scripts/generate_paper01_execution_evidence_linkage.py` | `canonical_consumer` | `ALREADY_CANONICAL` | `NONE` | `none` |
| `scripts/inventory_paper01_execution_evidence.py` | `governance_auditor` | `GOVERNANCE_TOOL` | `NONE` | `none` |
| `scripts/jsonl_to_csv.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/metrics/aggregate.py` | `legacy_or_direct_consumer` | `MIGRATE_TO_CANONICAL_INTERFACE` | `P1` | `trustforge.paper01_execution_evidence.load_paper01_linkage_study` |
| `scripts/metrics/local_fid_kid.py` | `historical_evidence_mutator_or_producer` | `PRESERVE_HISTORICAL` | `NONE` | `none` |
| `scripts/metrics/print_cfid_table.py` | `legacy_or_direct_consumer` | `MIGRATE_TO_CANONICAL_INTERFACE` | `P1` | `trustforge.paper01_execution_evidence.load_paper01_linkage_study` |
| `scripts/normalize_summaries.py` | `historical_evidence_mutator_or_producer` | `PRESERVE_HISTORICAL` | `NONE` | `none` |
| `scripts/phase1_html.py` | `historical_evidence_mutator_or_producer` | `MIGRATE_TO_CANONICAL_INTERFACE` | `P1` | `trustforge.paper01_execution_evidence.load_paper01_linkage_study` |
| `scripts/phase1_report.py` | `historical_evidence_mutator_or_producer` | `MIGRATE_TO_CANONICAL_INTERFACE` | `P1` | `trustforge.paper01_execution_evidence.load_paper01_linkage_study` |
| `scripts/plots/_common.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/core/calibration_curves.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/core/pareto_downstream_vs_similarity.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/core/per_class_delta_f1.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/diversity/ms_ssim_hist.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/diversity/nn_distance_distrib.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/hparams/ablation_bars.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/hparams/parallel_coords.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/imbalance/class_counts_before_after.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/imbalance/simple_stats_sanity.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/plots/qual/class_triptychs.py` | `legacy_or_direct_consumer` | `USE_CANONICAL_DERIVED_EXPORT` | `P2` | `proposed:trustforge.paper01_exports` |
| `scripts/summaries_to_jsonl.py` | `historical_evidence_mutator_or_producer` | `WRAP_WITH_COMPATIBILITY_LAYER` | `P1` | `proposed:trustforge.paper01_exports` |
| `scripts/trustforge_doctor.py` | `canonical_consumer` | `ALREADY_CANONICAL` | `NONE` | `none` |
| `scripts/tuning_dashboard.py` | `legacy_or_direct_consumer` | `MIGRATE_TO_CANONICAL_INTERFACE` | `P1` | `trustforge.paper01_execution_evidence.load_paper01_linkage_study` |
| `scripts/utils/backfill_counts.py` | `historical_evidence_mutator_or_producer` | `WRAP_WITH_COMPATIBILITY_LAYER` | `P1` | `proposed:trustforge.paper01_exports` |
| `src/trustforge/paper01_execution_evidence.py` | `canonical_consumer` | `ALREADY_CANONICAL` | `NONE` | `none` |
| `tests/trustforge/test_paper01_execution_evidence.py` | `canonical_consumer` | `ALREADY_CANONICAL` | `NONE` | `none` |
| `tools/aggregate_phase1.py` | `historical_evidence_mutator_or_producer` | `WRAP_WITH_COMPATIBILITY_LAYER` | `P1` | `proposed:trustforge.paper01_exports` |
| `tools/build_paper1_jsonl.py` | `historical_evidence_mutator_or_producer` | `WRAP_WITH_COMPATIBILITY_LAYER` | `P1` | `proposed:trustforge.paper01_exports` |
| `tools/build_phase1_scores.py` | `historical_evidence_mutator_or_producer` | `MIGRATE_TO_CANONICAL_INTERFACE` | `P0` | `trustforge.paper01_execution_evidence.load_paper01_linkage_study` |
| `tools/check_phase1_integrity.py` | `legacy_or_direct_consumer` | `MIGRATE_TO_CANONICAL_INTERFACE` | `P0` | `trustforge.paper01_execution_evidence.load_paper01_linkage_study` |
| `tools/freeze_phase1_snapshots.py` | `historical_evidence_mutator_or_producer` | `PRESERVE_HISTORICAL` | `NONE` | `none` |

## Interpretation

- `ALREADY_CANONICAL` requires no migration.
- `CONFIGURATION_ONLY` records dependency identity; it is not authorization to rewrite historical configs.
- `GOVERNANCE_TOOL` preserves evidence/audit responsibilities.
- `PRESERVE_HISTORICAL` protects historical execution or evidence-mutating behavior.
- `MIGRATE_TO_CANONICAL_INTERFACE` is a future direct-reader migration candidate.
- `WRAP_WITH_COMPATIBILITY_LAYER` should produce the legacy output contract from canonical linkage.
- `USE_CANONICAL_DERIVED_EXPORT` should remain decoupled from authority selection and consume a canonical-derived compatibility export.
- `REVIEW_BEFORE_MIGRATION` requires a separate safety review before any code change.

This audit is policy only. It does not modify any consumer or historical artifact.
