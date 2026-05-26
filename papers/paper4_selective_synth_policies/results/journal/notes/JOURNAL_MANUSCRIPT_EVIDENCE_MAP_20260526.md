# Paper 4 Journal Manuscript Evidence Map

Date: 2026-05-26

Paper:
Selective Synthetic Data Policies for Malware Classification: Allocation, Filtering, and Acceptance Rules Under Utility Constraints

## Purpose of this map

This note maps the completed journal evidence package to the future journal manuscript. It identifies what should appear in the main paper, what should move to appendix/supplement, and what claims are safe or unsafe.

The current journal evidence package is GAN-based and spans two datasets:

- USTC-TFC2016
- CICMalDroid2020

The paper should be positioned as a policy-evaluation study, not as a universal new-method-wins paper.

---

# 1. Core journal thesis

The journal paper should argue:

Synthetic malware-image augmentation should be treated as a policy-controlled intervention. The question is not only whether synthetic data can be generated, but which generated samples should be kept, filtered, ranked, or allocated under downstream utility constraints. Across two datasets, policy behavior is dataset-sensitive, metric-sensitive, and budget-sensitive. Confidence-based filtering and adaptive allocation can be useful, but neither is universally beneficial; held-out downstream evaluation remains necessary.

---

# 2. Recommended journal manuscript structure

## Abstract

Should emphasize:

- Synthetic cybersecurity data is not automatically useful.
- This paper studies selection policies over generated malware images.
- Policies include keep-all, strict confidence filtering, balanced top-k selection, and adaptive class-repair allocation.
- Evaluation spans USTC-TFC2016 and CICMalDroid2020.
- Results show policy effects are dataset-sensitive and metric-sensitive.
- Confidence-only filtering can collapse class coverage.
- Adaptive allocation helps in USTC under reduced budget but does not universally dominate on CICMalDroid.

## Introduction

Main motivation:

- Most synthetic-data work asks whether generated data looks realistic or increases dataset size.
- For downstream malware classification, generated data should be treated as an intervention.
- The key policy question is: which synthetic samples should be retained and how should budget be allocated?

Main research questions:

RQ1. How do keep-all, confidence filtering, top-k ranking, and class-repair allocation affect downstream malware classification utility?

RQ2. Can audit-confidence signals identify useful synthetic samples without destroying class coverage?

RQ3. Does adaptive allocation improve reduced-budget synthetic augmentation?

RQ4. Do policy conclusions transfer across malware datasets?

Contributions:

1. A selective synthetic-data policy framework for malware-image augmentation.
2. A comparison of keep-all, strict-confidence, top-k, and class-repair policies.
3. Evidence that strict confidence filtering can collapse class coverage.
4. Evidence that adaptive allocation changes selected-set structure under fixed budget.
5. Two-dataset validation showing policy behavior differs between USTC-TFC2016 and CICMalDroid2020.
6. Diagnostic evidence showing quality/diversity proxies are useful but insufficient as standalone policy selectors.

## Related Work

Organize around:

1. Synthetic malware images and malware classification.
2. Generative models for cybersecurity data augmentation.
3. Synthetic-data utility and downstream evaluation.
4. Dataset shift, class imbalance, and policy-controlled augmentation.
5. Confidence, uncertainty, and selection/filtering strategies.
6. Limitations of quality/diversity proxies for downstream usefulness.

References will be added later.

## Methodology / Experimental Design

Recommended section title:

`Experimental Design`

Reason:
This paper is primarily an empirical policy-evaluation study. “Experimental Design” better captures datasets, policies, budgets, selection rules, evaluation metrics, seeds, and artifact protocol.

Subsections:

1. Datasets and generated sample pools
2. Policy families
3. Audit-confidence scoring
4. Downstream utility evaluation
5. Quality/diversity diagnostics
6. Reproducibility and artifact protocol

---

# 3. Main-paper evidence selection

## Main Table 1: Policy definitions

Include:

| Policy | Description | Budget behavior | Selection signal |
|---|---|---|---|
| Real-only | No synthetic augmentation | 0 | none |
| Keep-all | Use all generated samples | full budget | none |
| Strict-conf. | Accept if audit classifier predicts requested label with confidence threshold | variable/collapsible | max confidence + label agreement |
| Top-k | Keep top-k per class by requested-class confidence | balanced reduced budget | requested-class confidence |
| Class-repair | Allocate more budget to audit-weaker classes, then rank within class | adaptive reduced budget | median requested-class confidence + within-class confidence |

This table belongs in Methodology.

## Main Table 2: USTC policy summary

Use existing table:

- `results/tables/paper4_policy_mean_std.md`
- `results/tables/paper4_policy_mean_std.csv`

Main paper should show compact columns only:

| Policy | Samples | Δ Macro-F1 | Δ Balanced Acc. | Δ Macro-AUPRC |
|---|---:|---:|---:|---:|

Suggested rows:

- Keep-all b2000
- Strict-conf.
- Top-k500
- Top-k1000
- Class-repair

Optional include Baseline b2000 if needed for context.

Key USTC message:

- Keep-all b2000 is strongest on average for Macro-F1 and balanced accuracy.
- Top-k1000 is a useful reduced-budget policy.
- Class-repair gives similar reduced-budget Macro-F1 and balanced-accuracy behavior and stronger Macro-AUPRC than Top-k1000.
- Strict-conf. is diagnostic, not practical.

## Main Table 3: CICMalDroid policy summary

Use:

- `results/journal/tables/paper4_journal_cicmaldroid_policy_mean_std.csv`
- `results/journal/tables/paper4_journal_cicmaldroid_policy_mean_std.md`

Main paper compact columns:

| Policy | Samples | Δ Macro-F1 | Δ Balanced Acc. | Δ Macro-AUPRC | Δ ECE | Δ Brier |
|---|---:|---:|---:|---:|---:|---:|

Rows:

- CIC Keep-all b2000
- CIC Top-k1000
- CIC Class-repair

Key CICMalDroid message:

- All policies have negative average balanced-accuracy deltas.
- Top-k1000 is generally more favorable than Class-repair on Macro-F1 and balanced accuracy.
- Keep-all improves Macro-AUPRC on average but hurts Macro-F1 and balanced accuracy.
- Class-repair improves ECE most strongly but does not improve primary utility metrics on this dataset.

## Main Figure 1: Strict-confidence collapse

Use:

- `results/journal/figures/paper4_journal_strict_conf_class_coverage_collapse.pdf`
- `results/journal/figures/paper4_journal_strict_conf_acceptance_rate_by_class.pdf`

Recommended main-paper use:

Use only one figure in the main paper, preferably class coverage collapse. Put acceptance-rate-by-class in appendix.

Key message:

Strict-conf. accepts only 1004 of 18000 samples and reduces requested-class coverage from 9 classes to 1 class.

## Main Figure 2: Class-repair allocation behavior

Use:

- `results/journal/figures/paper4_journal_class_repair_allocation_vs_audit_confidence.pdf`
- `results/journal/figures/paper4_journal_class_repair_allocation_heatmap.pdf`

Recommended main-paper use:

Use allocation-vs-audit-confidence scatter in main paper. Put heatmap in appendix.

Key message:

Across USTC class-seed pairs, audit confidence is strongly negatively associated with allocated sample count. This validates that Class-repair is a genuine adaptive allocation policy.

## Main Figure 3: Cross-dataset policy comparison

This figure may need to be created.

Recommended new figure:

Grouped bar chart comparing USTC vs CICMalDroid for:

- Keep-all b2000
- Top-k1000
- Class-repair

Metrics:

- Δ Macro-F1
- Δ Balanced Accuracy
- Δ Macro-AUPRC

This should probably become the strongest journal figure because it visually shows dataset sensitivity.

Potential script to create later:

`results/journal/scripts/paper4_journal_cross_dataset_policy_comparison.py`

## Appendix / Supplement evidence

Move these to appendix or supplement:

1. Full seed-level CICMalDroid table
2. Full seed-level USTC table
3. KID/MS-SSIM quality-utility scatterplots
4. MS-SSIM vs utility figures
5. Strict-conf. acceptance-rate-by-class figure
6. Class-repair allocation heatmap
7. Full raw policy diagnostic notes
8. Tables with KID and MS-SSIM columns

---

# 4. Results section evidence plan

Recommended Results subsections:

## 4.1 Overall USTC policy utility

Use:

- `paper4_policy_mean_std.csv`
- `paper4_policy_mean_std.md`

Claims:

- Selective policies are not automatically superior to keep-all.
- Top-k1000 gives positive reduced-budget utility.
- Class-repair is competitive under the same 9000-sample budget and improves Macro-AUPRC relative to Top-k1000.
- Strict-conf. has near-zero downstream value and is diagnostic.

Avoid:

- Do not claim Class-repair beats full keep-all.
- Do not claim adaptive allocation is always best.

## 4.2 Strict-confidence filtering can collapse class coverage

Use:

- `paper4_journal_strict_conf_collapse_summary.csv`
- `paper4_journal_strict_conf_collapse_by_class.csv`
- `paper4_journal_strict_conf_class_coverage_collapse.pdf`

Claims:

- Strict-conf. accepted 1004 of 18000 samples.
- Requested-class coverage collapsed from 9 classes to 1.
- Confidence-only filtering can preserve high-confidence samples while destroying class coverage.

Avoid:

- Do not say confidence filtering is useless.
- Say confidence filtering requires coverage and utility constraints.

## 4.3 Reduced-budget ranking is budget-sensitive

Use:

- `paper4_journal_topk_budget_sensitivity_summary.csv`
- `paper4_journal_topk_budget_sensitivity_delta_macro_f1.pdf`
- `paper4_journal_topk_budget_sensitivity_delta_bal_acc.pdf`
- `paper4_journal_topk_budget_sensitivity_delta_macro_auprc.pdf`

Claims:

- Top-k500 is too restrictive.
- Top-k1000 restores positive Macro-F1 and balanced-accuracy gains.
- Reduced-budget selection depends on retained sample count.

Avoid:

- Do not claim any k is universally optimal.

## 4.4 Adaptive Class-repair allocation behavior

Use:

- `paper4_journal_class_repair_allocation_correlation.csv`
- `paper4_journal_class_repair_allocation_vs_audit_confidence.pdf`

Claims:

- Class-repair reallocates away from audit-confident classes and toward audit-weaker classes.
- Audit confidence has strong negative association with allocated samples.
- Allocation structure can matter under a fixed selected-sample budget.

Avoid:

- Do not imply allocation always improves utility.

## 4.5 Quality--utility diagnostics

Use:

- `paper4_journal_proxy_utility_correlations.csv`
- `paper4_journal_quality_utility_policy_summary.csv`
- KID/MS-SSIM scatterplots

Claims:

- KID shows moderate association with downstream utility in USTC.
- MS-SSIM has little association.
- Similar KID/MS-SSIM values can correspond to different downstream utility.

Avoid:

- Do not claim KID/MS-SSIM are useless.
- Say they are insufficient as standalone selectors.

## 4.6 CICMalDroid second-dataset validation

Use:

- `paper4_journal_cicmaldroid_policy_mean_std.csv`
- `paper4_journal_cicmaldroid_policy_seed_level.csv`

Claims:

- CICMalDroid confirms dataset sensitivity.
- Keep-all and reduced-budget policies do not consistently reverse negative balanced-accuracy effects.
- Top-k1000 is generally more favorable than Class-repair on CICMalDroid.
- Class-repair improves calibration most strongly but is not best on primary utility.

Avoid:

- Do not claim the CICMalDroid result invalidates USTC.
- Frame it as cross-dataset variation, not contradiction.

---

# 5. Discussion section evidence plan

Discussion should synthesize:

1. Synthetic data is an intervention, not just extra data.
2. Confidence signals are useful but incomplete.
3. Coverage constraints are necessary.
4. Allocation can help, but its value is dataset-dependent.
5. Different metrics tell different stories.
6. Quality/diversity proxies are diagnostics, not substitutes for downstream utility.
7. CICMalDroid shows that policy rankings do not automatically transfer across datasets.

Potential discussion paragraph themes:

- Why strict confidence fails: label agreement plus high confidence does not guarantee class coverage.
- Why Top-k helps: it imposes balanced coverage and avoids collapse.
- Why Class-repair helps in USTC but not CICMalDroid: audit-confidence weakness may not align with downstream weakness on every dataset.
- Why Macro-F1, balanced accuracy, Macro-AUPRC, ECE, and Brier can disagree.
- Why journal contribution is evaluation framework + policy insight, not a universal winning policy.

---

# 6. Limitations

Must include:

1. Current two-dataset package is GAN-based.
2. VAE/diffusion policy behavior remains future work unless added later.
3. Audit classifier is primarily CNN-based.
4. Policy decisions depend on audit-confidence quality.
5. CICMalDroid conversion to image-like tensors may affect interpretation.
6. Synthetic data is evaluated for malware classification utility, not deployment-time robustness.
7. Some metrics can improve while primary utility decreases.

---

# 7. Main claims to use

Safe claims:

- Selective synthetic-data policies are dataset-sensitive.
- Confidence-only filtering can collapse class coverage.
- Balanced top-k ranking avoids collapse but is budget-sensitive.
- Class-repair creates genuine adaptive allocation behavior.
- Class-repair can be useful under reduced budget, especially on USTC.
- CICMalDroid shows that adaptive allocation is not universally superior.
- KID/MS-SSIM are useful diagnostics but insufficient as standalone policy selectors.
- Held-out downstream utility evaluation is necessary.

---

# 8. Claims to avoid

Avoid:

- Class-repair is the best policy.
- Selective filtering always improves synthetic augmentation.
- Confidence is a reliable proxy for utility.
- KID or MS-SSIM are useless.
- Results generalize to all malware datasets.
- Results generalize to all generator families.
- CICMalDroid confirms USTC policy ordering.

---

# 9. Decision about VAE

Recommendation:

Do not add VAE immediately unless the journal target demands multi-generator validation or the advisor specifically asks for it.

Reason:

The current paper already has a coherent two-dataset GAN-based story. Adding VAE would expand scope and delay writing. If added, VAE should be framed as generator-family robustness, not as required to make the current core contribution valid.

If VAE is added later, minimum package:

- VAE keep-all b2000 on USTC and/or CICMalDroid
- VAE Top-k reduced-budget policy
- VAE Class-repair policy
- Collector comparing GAN vs VAE policy rankings

---

# 10. Immediate next step after this map

Recommended next action:

Create compact journal-ready tables and cross-dataset figures from the existing evidence.

Suggested scripts:

1. `paper4_journal_make_compact_main_tables.py`
2. `paper4_journal_cross_dataset_policy_comparison.py`

These should produce:

- `paper4_main_table_ustc_policy_summary.tex`
- `paper4_main_table_cic_policy_summary.tex`
- `paper4_cross_dataset_delta_macro_f1.pdf`
- `paper4_cross_dataset_delta_bal_acc.pdf`
- `paper4_cross_dataset_delta_macro_auprc.pdf`

After that, begin drafting the journal Results section.
