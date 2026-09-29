# M6.5.1 — Paper 1 Downstream Consumer Inventory

- Audit version: `M6.5.1-v1.0`
- Study: `paper01_benchmark`
- Repository HEAD: `1cc580ec2c47e6108a12fb429adb3bdb8d8f1ff8`
- Inventory records: **71**

## Governance boundary

This audit inventories repository references and consumers only.
It does not migrate code, modify historical artifacts, infer provenance,
or promote any legacy summary/alias to authoritative evidence.

## Classification census

- `canonical_consumer`: 4
- `configuration_dependency`: 22
- `documentation`: 7
- `governance_auditor`: 1
- `historical_evidence_mutator_or_producer`: 14
- `historical_pipeline`: 1
- `legacy_or_direct_consumer`: 17
- `test_or_ci`: 5

## Evidence-category census

- `authoritative_score_table`: 4
- `canonical_linkage_interface`: 4
- `consolidated_jsonl`: 21
- `historical_paper1_root`: 27
- `latest_alias`: 17
- `manifest_evidence`: 22
- `timestamped_summary`: 22

## Inventory

| Path | Classification | Migration policy | Categories | Access |
|---|---|---|---|---|
| `.github/workflows/ci.yml` | `test_or_ci` | `review_only` | `consolidated_jsonl` | `reference` |
| `Makefile` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `consolidated_jsonl, historical_paper1_root, latest_alias, timestamped_summary` | `reference, write` |
| `README.md` | `documentation` | `review_only` | `consolidated_jsonl, timestamped_summary` | `reference` |
| `Runbook.md` | `documentation` | `review_only` | `consolidated_jsonl, timestamped_summary` | `reference` |
| `configs/paper1_cicmaldroid.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `configs/paper1_config.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `configs/paper1_final.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `configs/paper1_final_rerun.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `configs/paper1_real.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `configs/paper1_smoke.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `docs/REPRODUCIBILITY_CONTRACT.md` | `documentation` | `review_only` | `latest_alias` | `reference` |
| `docs/STORAGE_AND_PATHS.md` | `documentation` | `review_only` | `historical_paper1_root, latest_alias` | `reference` |
| `docs/paper01_execution_evidence_linkage_schema.md` | `documentation` | `review_only` | `authoritative_score_table, latest_alias, manifest_evidence, timestamped_summary` | `reference` |
| `eval/runner.py` | `historical_pipeline` | `protected_do_not_migrate_in_m6_5_1` | `latest_alias, manifest_evidence, timestamped_summary` | `reference, write` |
| `papers/paper2_conditional_generation_done_right/configs/paper2_cicmaldroid_fakeclass_seed42.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `papers/paper2_conditional_generation_done_right/configs/paper2_cicmaldroid_fakeclass_seed43.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `papers/paper2_conditional_generation_done_right/configs/paper2_cicmaldroid_fakeclass_seed44.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `papers/paper2_conditional_generation_done_right/configs/paper2_cicmaldroid_fakeclass_smoke_seed42.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `papers/paper2_conditional_generation_done_right/notes/paper2_cicmaldroid_seed42_findings.md` | `documentation` | `review_only` | `historical_paper1_root` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_class_repair_topk5000_b2000_s42.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root, manifest_evidence` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_class_repair_topk5000_b2000_s43.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root, manifest_evidence` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_class_repair_topk5000_b2000_s44.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root, manifest_evidence` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_confidence_ranked_topk1000_b2000_s42.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root, manifest_evidence` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_confidence_ranked_topk1000_b2000_s43.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root, manifest_evidence` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_confidence_ranked_topk1000_b2000_s44.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root, manifest_evidence` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_keep_all_b2000_s42.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root, manifest_evidence` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_keep_all_b2000_s43.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root, manifest_evidence` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_policy_keep_all_b2000_s44.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root, manifest_evidence` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_realonly_balanced_s42.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_realonly_balanced_s43.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `papers/paper4_selective_synth_policies/configs/paper4_cic_realonly_balanced_s44.yaml` | `configuration_dependency` | `review_dependency_no_automatic_migration` | `historical_paper1_root` | `reference` |
| `scripts/backfill_kid_and_downstream.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `latest_alias, manifest_evidence, timestamped_summary` | `read, reference, write` |
| `scripts/backfill_ms_ssim_per_class.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `manifest_evidence, timestamped_summary` | `read, reference, write` |
| `scripts/build_jsonl.sh` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl, timestamped_summary` | `reference` |
| `scripts/eval_write_summary.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `timestamped_summary` | `reference, write` |
| `scripts/generate_paper01_execution_evidence_linkage.py` | `canonical_consumer` | `already_canonical` | `canonical_linkage_interface, latest_alias, manifest_evidence` | `reference, write` |
| `scripts/inventory_paper01_execution_evidence.py` | `governance_auditor` | `preserve_governance_tool` | `authoritative_score_table, latest_alias, manifest_evidence` | `read, reference, write` |
| `scripts/jsonl_to_csv.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/metrics/aggregate.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `timestamped_summary` | `read` |
| `scripts/metrics/local_fid_kid.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `latest_alias, manifest_evidence, timestamped_summary` | `read, reference, write` |
| `scripts/metrics/print_cfid_table.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `timestamped_summary` | `reference` |
| `scripts/normalize_summaries.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `timestamped_summary` | `read, reference, write` |
| `scripts/phase1_html.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `timestamped_summary` | `read, write` |
| `scripts/phase1_report.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `timestamped_summary` | `read, write` |
| `scripts/plots/_common.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/plots/core/calibration_curves.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/plots/core/pareto_downstream_vs_similarity.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/plots/core/per_class_delta_f1.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/plots/diversity/ms_ssim_hist.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/plots/diversity/nn_distance_distrib.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/plots/hparams/ablation_bars.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/plots/hparams/parallel_coords.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/plots/imbalance/class_counts_before_after.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl` | `reference` |
| `scripts/plots/imbalance/simple_stats_sanity.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl, manifest_evidence` | `read, reference` |
| `scripts/plots/qual/class_triptychs.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `consolidated_jsonl, manifest_evidence` | `reference` |
| `scripts/summaries_to_jsonl.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `consolidated_jsonl, latest_alias, timestamped_summary` | `reference, write` |
| `scripts/trustforge_doctor.py` | `canonical_consumer` | `already_canonical` | `canonical_linkage_interface` | `reference` |
| `scripts/tuning_dashboard.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `manifest_evidence, timestamped_summary` | `read, reference` |
| `scripts/utils/backfill_counts.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `consolidated_jsonl, manifest_evidence` | `reference, write` |
| `src/trustforge/paper01_execution_evidence.py` | `canonical_consumer` | `already_canonical` | `canonical_linkage_interface, latest_alias` | `reference` |
| `studies/paper01_benchmark/README.md` | `documentation` | `review_only` | `authoritative_score_table, historical_paper1_root` | `reference` |
| `tests/test_smoke.py` | `test_or_ci` | `review_only` | `manifest_evidence, timestamped_summary` | `read, reference` |
| `tests/trustforge/test_paper01_execution_evidence.py` | `canonical_consumer` | `already_canonical` | `canonical_linkage_interface, latest_alias` | `reference, write` |
| `tests/trustforge/test_paper01_execution_evidence_inventory.py` | `test_or_ci` | `review_only` | `latest_alias, timestamped_summary` | `reference, write` |
| `tests/trustforge/test_paper01_execution_evidence_linkage_generator.py` | `test_or_ci` | `review_only` | `latest_alias` | `reference` |
| `tests/trustforge/test_paper01_execution_evidence_linkage_schema.py` | `test_or_ci` | `review_only` | `authoritative_score_table, historical_paper1_root, latest_alias, manifest_evidence, timestamped_summary` | `reference` |
| `tools/aggregate_phase1.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `consolidated_jsonl, timestamped_summary` | `reference, write` |
| `tools/build_paper1_jsonl.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `consolidated_jsonl` | `reference, write` |
| `tools/build_phase1_scores.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `latest_alias` | `reference, write` |
| `tools/check_phase1_integrity.py` | `legacy_or_direct_consumer` | `candidate_for_future_canonical_migration` | `latest_alias` | `reference` |
| `tools/freeze_phase1_snapshots.py` | `historical_evidence_mutator_or_producer` | `review_before_any_migration` | `timestamped_summary` | `read, write` |

## Interpretation rules

- `canonical_consumer` already references the governed canonical linkage interface.
- `legacy_or_direct_consumer` directly reads or references legacy Paper 1 evidence surfaces.
- `configuration_dependency` declares a Paper 1 storage/evidence dependency but is not executable consumer code.
- `governance_auditor` is TrustForge migration/audit tooling and must not be treated as legacy downstream code.
- `historical_evidence_mutator_or_producer` contains write/mutation signals near Paper 1 evidence references; this is not authorization to change it.
- `historical_pipeline` is explicitly protected from migration in M6.5.1.
- Paper result tables, frozen/raw result data, schema examples, canonical evidence records, and prior audits are excluded from consumer counts.
