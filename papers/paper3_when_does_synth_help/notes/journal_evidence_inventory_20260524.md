# Paper 3 Journal Evidence Inventory

Date: 2026-05-24  
Paper: When Does Synthetic Data Help Malware Classification? Utility, Class Imbalance, and Failure Modes Across Augmentation Regimes

## 1. Current Paper Identity

Paper 3 studies synthetic malware-image augmentation as a downstream intervention rather than only as a generative modeling task. The central question is not simply whether a generator can produce samples, but when synthetic data improves or harms held-out real-data classification performance.

The paper compares augmentation behavior across:

- generator families: GAN, VAE, Diffusion
- synthetic budgets
- balanced all-class augmentation
- minority-heavy c4/c7 minority-only augmentation
- downstream utility metrics
- targeted minority-class diagnostics
- precision--recall tradeoff behavior
- external synthetic class-alignment audit

## 2. Core Frozen Evidence Blocks

### Evidence Block A: Balanced All-Class Augmentation

Purpose:

Evaluate whether synthetic augmentation helps when all classes receive synthetic samples under balanced augmentation.

Frozen evidence:

- `papers/paper3_when_does_synth_help/results/frozen/paper3_gan_vae_diffusion_budget_aggregate_frozen_20260516.csv`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig1_balanced_macro_f1_vs_budget.pdf`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig1_balanced_macro_f1_vs_budget.png`

Claim supported:

Balanced all-class augmentation can help, especially at larger budgets, but the effect depends on generator family and budget.

---

### Evidence Block B: Minority-Heavy c4/c7 Cross-Family Results

Purpose:

Evaluate whether minority-only synthetic augmentation repairs a minority-heavy regime where classes 4 and 7 are reduced to 20 percent of their real-training samples.

Frozen evidence:

- `papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_cross_family_aggregate_frozen_20260517.csv`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_cross_family_aggregate_table.md`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_cross_family_aggregate_table.tex`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig2_minority_heavy_macro_f1_vs_budget.pdf`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig2_minority_heavy_macro_f1_vs_budget.png`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig3_balanced_vs_minority_b2000_macro_f1.pdf`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig3_balanced_vs_minority_b2000_macro_f1.png`

Claim supported:

Minority-only augmentation under c4/c7 imbalance hurts across GAN, VAE, and Diffusion. This shows that targeted augmentation is not automatically beneficial, even when synthetic samples are added to the depleted minority classes.

---

### Evidence Block C: Minority-Class c4/c7 Per-Class Diagnostics

Purpose:

Determine whether macro-level minority-heavy degradation reflects actual failure on the targeted minority classes.

Frozen evidence:

- `papers/paper3_when_does_synth_help/results/raw/paper3_minority_heavy_perclass_c4c7_raw.csv`
- `papers/paper3_when_does_synth_help/results/raw/paper3_minority_heavy_perclass_c4c7_aggregate.csv`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_perclass_c4c7_aggregate_frozen_20260524.csv`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_perclass_c4c7_aggregate_table.md`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_perclass_c4c7_aggregate_table.tex`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig4_minority_c4c7_delta_f1_by_family_budget.pdf`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig4_minority_c4c7_delta_f1_by_family_budget.png`

Claim supported:

The targeted minority classes themselves are not reliably repaired. Class 4 is especially vulnerable, with mostly negative F1 changes across families and budgets. Class 7 is more mixed but does not show reliable F1 improvement.

---

### Evidence Block D: Precision--Recall Tradeoff Diagnosis

Purpose:

Explain how minority-only augmentation fails at the class level.

Frozen evidence:

- `papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_c4c7_precision_recall_tradeoff_frozen_20260524.csv`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_c4c7_precision_recall_tradeoff_table.md`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_c4c7_precision_recall_tradeoff_table.tex`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig5_c4c7_precision_recall_tradeoff.pdf`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig5_c4c7_precision_recall_tradeoff.png`

Main observed patterns:

- precision gain / recall loss: 8 cases
- both precision and recall decline: 6 cases
- recall gain / precision loss: 3 cases
- net F1 decline: 1 case

Claim supported:

The minority-heavy failure is not a simple quantity problem. Synthetic augmentation often shifts the precision--recall balance without improving F1. Class 4 frequently suffers recall loss, while class 7 sometimes trades recall gains for precision loss.

---

### Evidence Block E: Synthetic Class-Alignment Audit

Purpose:

Test whether requested class-4 and class-7 synthetic samples are externally recognized as class 4 and class 7 by an independent real-only audit classifier.

Frozen evidence:

- `papers/paper3_when_does_synth_help/results/raw/paper3_c4c7_synthetic_alignment_raw.csv`
- `papers/paper3_when_does_synth_help/results/raw/paper3_c4c7_synthetic_alignment_pred_hist.csv`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_aggregate_frozen_20260524.csv`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_table.md`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_table.tex`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_summary_20260524.json`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig6_c4c7_synthetic_alignment_audit.pdf`
- `papers/paper3_when_does_synth_help/results/frozen/figures_final/paper3_fig6_c4c7_synthetic_alignment_audit.png`

Observed alignment pattern:

- Diffusion requested c4/c7 samples are externally recognized as class 8.
- GAN requested c4/c7 samples are predominantly externally recognized as class 1.
- VAE requested c4 samples are externally recognized as class 7.
- VAE requested c7 samples align with class 7.

Claim supported:

Minority-only augmentation fails partly because the generated minority samples are often class-misaligned. The synthetic budget increases sample count, but many samples do not behave like the requested minority class under an independent real-only classifier.

## 3. Current Journal-Level Story

The current evidence supports the following narrative:

1. Synthetic data can help under balanced all-class augmentation, but gains depend on family and budget.
2. The same idea fails under a minority-heavy c4/c7 regime when synthetic data is added only to minority classes.
3. The failure is visible not only in macro metrics but also in the targeted minority classes.
4. The failure often appears as precision--recall tradeoff instability.
5. External alignment auditing shows that many synthetic minority samples are not recognized as their requested minority class.
6. Therefore, synthetic augmentation should be treated as a regime-dependent intervention requiring downstream validation and class-faithfulness diagnostics.

## 4. Candidate Figures for Journal Version

Recommended core figures:

- Fig. 1: Balanced all-class Macro-F1 vs budget
- Fig. 2: Minority-heavy Macro-F1 vs budget
- Fig. 3: Balanced vs minority-heavy comparison at high budget
- Fig. 4: Minority-class c4/c7 Delta F1
- Fig. 5: c4/c7 precision--recall tradeoff
- Fig. 6: c4/c7 synthetic class-alignment audit

Possible journal figure reduction:

If space is limited, combine or move some figures to appendix/supplement:

- Keep Fig. 1, Fig. 3, Fig. 4, Fig. 6 in main paper.
- Move Fig. 2 and Fig. 5 to appendix if needed.
- Keep the precision--recall table or summarize it in text.

## 5. Remaining Gaps Before Journal Submission

### Gap 1: Dataset Generalization

Current journal-strength diagnostics are based on USTC-TFC2016. A second dataset would strengthen external validity.

Possible next dataset:

- CICMalDroid2020

Benefit:

Shows whether the regime-dependent failure pattern generalizes beyond one malware-image dataset.

Cost:

Requires reproducing at least the balanced vs minority-heavy evidence and possibly a smaller alignment audit.

---

### Gap 2: Classifier Architecture Sensitivity

Current downstream and audit results depend on the chosen CNN classifiers.

Possible extension:

- repeat selected evidence with a second classifier architecture, e.g., ResNet-style CNN or linear/SVM feature baseline

Benefit:

Reduces concern that the observed failure is an artifact of one classifier.

Cost:

Moderate. May be easier than a full second dataset.

---

### Gap 3: Visual / Qualitative Synthetic Inspection

Current evidence is quantitative. A small qualitative grid of generated c4/c7 samples might help reviewers understand alignment failure.

Benefit:

Supports the alignment audit visually.

Risk:

Qualitative evidence can be subjective and should be secondary.

---

### Gap 4: Calibration / Confidence Analysis

Synthetic augmentation may affect confidence and calibration even when F1 changes are small.

Possible metrics:

- ECE
- Brier score
- confidence distributions for c4/c7

Benefit:

Useful for a top journal if positioned as trustworthiness.

Cost:

Only worthwhile if already available in summaries or easy to extract.

## 6. Recommended Next Research Move

The next best step is to decide between:

### Option A: Second Dataset Extension

Best if targeting a high-impact journal that expects broad generalization.

Minimum version:

- run balanced and minority-heavy comparison on CICMalDroid2020
- include macro-F1, balanced accuracy, macro-AUPRC
- optionally include alignment audit if synthetic labels/classes are compatible

### Option B: Classifier Sensitivity Check

Best if compute/time is limited.

Minimum version:

- repeat key Paper 3 downstream comparisons using one alternative downstream classifier
- focus only on selected budgets, e.g., 100 and 2000
- include c4/c7 per-class F1

### Option C: Manuscript Integration First

Best if the current evidence is already enough for the intended journal/conference target.

Minimum version:

- rewrite Results around the evidence blocks
- add a stronger Discussion section
- add limitations around single dataset and classifier dependence
- position second dataset as future work if not required

## 7. Current Recommendation

For a top journal version, the strongest next move is:

1. Perform a compact classifier-sensitivity check on USTC-TFC2016.
2. Then decide whether a second dataset is necessary.

Reason:

The current paper already has deep single-dataset diagnostics. A reviewer may first question whether the failure depends on the downstream classifier. Addressing classifier sensitivity is cheaper and directly strengthens the trustworthiness of the current conclusions.
