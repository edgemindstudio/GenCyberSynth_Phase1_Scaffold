# Paper 4 Journal Evidence Checkpoint

Date: 2026-05-22

Paper:
Selective Synthetic Data Policies for Malware Classification: Allocation, Filtering, and Acceptance Rules Under Utility Constraints

## Completed journal expansion layers

### 1. Quality--utility diagnostics

Commit/tag:
- paper4-journal-quality-utility-20260522

Key files:
- results/journal/scripts/paper4_journal_quality_utility_diagnostics.py
- results/journal/tables/paper4_journal_quality_utility_policy_summary.csv
- results/journal/tables/paper4_journal_proxy_utility_correlations.csv
- results/journal/figures/paper4_journal_kid_vs_delta_macro_f1.pdf
- results/journal/figures/paper4_journal_kid_vs_delta_macro_auprc.pdf
- results/journal/figures/paper4_journal_msssim_vs_delta_macro_f1.pdf
- results/journal/figures/paper4_journal_msssim_vs_delta_macro_auprc.pdf

Main finding:
KID shows moderate association with downstream utility in this result set, while MS-SSIM shows little association. However, proxy diagnostics do not fully explain downstream policy utility because Top-k1000 and Class-repair have similar KID/MS-SSIM values but different Macro-AUPRC gains.

### 2. Class-repair allocation diagnostics

Commit/tag:
- paper4-journal-class-repair-allocation-20260522

Key files:
- results/journal/scripts/paper4_journal_class_repair_allocation_diagnostics.py
- results/journal/tables/paper4_journal_class_repair_allocation_by_seed.csv
- results/journal/tables/paper4_journal_class_repair_allocation_mean_std.csv
- results/journal/tables/paper4_journal_class_repair_allocation_correlation.csv
- results/journal/figures/paper4_journal_class_repair_allocation_heatmap.pdf
- results/journal/figures/paper4_journal_class_repair_allocation_vs_audit_confidence.pdf

Main finding:
Class-repair is a true adaptive allocation policy. Across 27 class-seed pairs, audit confidence is strongly negatively associated with allocated sample count, with Pearson correlation -0.8981 and Spearman correlation -0.7784. This confirms that Class-repair reallocates the fixed 9000-sample budget toward audit-weaker classes rather than behaving like balanced Top-k1000.

### 3. Strict-confidence collapse diagnostics

Commit/tag:
- paper4-journal-strict-conf-collapse-20260522

Key files:
- results/journal/scripts/paper4_journal_strict_conf_collapse_diagnostics.py
- results/journal/tables/paper4_journal_strict_conf_collapse_summary.csv
- results/journal/tables/paper4_journal_strict_conf_collapse_by_class.csv
- results/journal/figures/paper4_journal_strict_conf_class_coverage_collapse.pdf
- results/journal/figures/paper4_journal_strict_conf_acceptance_rate_by_class.pdf

Main finding:
Strict-conf. accepts only 1004 of 18000 samples, for an acceptance rate of 0.055778. Requested-class coverage collapses from 9 classes to 1 class. Only requested class 4 is accepted. This shows that confidence-only filtering can preserve high-confidence agreement while destroying class coverage.

### 4. Top-k budget sensitivity diagnostics

Commit/tag:
- paper4-journal-topk-budget-sensitivity-20260522

Key files:
- results/journal/scripts/paper4_journal_topk_budget_sensitivity.py
- results/journal/tables/paper4_journal_topk_budget_sensitivity_seed_level.csv
- results/journal/tables/paper4_journal_topk_budget_sensitivity_summary.csv
- results/journal/figures/paper4_journal_topk_budget_sensitivity_delta_macro_f1.pdf
- results/journal/figures/paper4_journal_topk_budget_sensitivity_delta_bal_acc.pdf
- results/journal/figures/paper4_journal_topk_budget_sensitivity_delta_macro_auprc.pdf

Main finding:
Top-k500 is too restrictive and gives near-zero or slightly negative Macro-F1 and balanced-accuracy deltas. Top-k1000 restores positive Macro-F1 and balanced-accuracy utility. Class-repair uses the same 9000-sample budget as Top-k1000 but slightly improves Macro-F1 and balanced accuracy and gives a larger Macro-AUPRC gain.

## Current journal-safe claims

- Synthetic sample selection should be treated as a policy-controlled intervention.
- Selective policies are not automatically better than Keep-all.
- Strict confidence filtering can collapse class coverage.
- Balanced Top-k selection is sensitive to retained budget.
- Adaptive allocation changes the selected-set structure under the same total budget.
- Class-repair is the strongest reduced-budget policy in the current USTC/GAN setting, especially for Macro-AUPRC.
- KID and MS-SSIM are useful diagnostics but not sufficient as standalone policy-selection criteria.

## Claims to avoid

- Do not claim Class-repair universally beats Keep-all.
- Do not claim Class-repair is universally best across datasets or generators.
- Do not claim confidence alone measures downstream usefulness.
- Do not claim KID/MS-SSIM are useless.
- Do not claim journal-level generality until second-dataset or multi-generator evidence is added.

## Next recommended journal steps

1. Write expanded journal Results subsections from these diagnostics.
2. Add a compact journal evidence table/figure selection plan.
3. Decide whether to run CICMalDroid2020 policy experiments as second-dataset validation.
4. Decide whether to add one additional generator family, preferably VAE, after second-dataset planning.