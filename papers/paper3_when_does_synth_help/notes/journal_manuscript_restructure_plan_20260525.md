# Paper 3 Journal Manuscript Restructure Plan

Date: 2026-05-25  
Paper: When Does Synthetic Data Help Malware Classification? Utility, Class Imbalance, and Failure Modes Across Augmentation Regimes

## 1. Journal-Version Identity

The journal version should not be framed as only a budget-comparison paper.

The stronger identity is:

> A regime- and evaluator-aware study of when synthetic malware-image augmentation helps, fails, and why it fails under class imbalance.

The key idea is that synthetic data is not automatically useful because it increases training-set size. It should be treated as a downstream intervention whose value depends on:

- augmentation regime,
- target class structure,
- generator family,
- synthetic budget,
- class-level behavior,
- synthetic class alignment,
- downstream classifier architecture,
- and evaluation metric.

## 2. Possible Journal Title

Recommended title:

**When Does Synthetic Data Help Malware Classification? Regime-Dependent Utility, Minority-Class Failure, and Class-Alignment Diagnostics**

Alternative shorter title:

**When Does Synthetic Data Help Malware Classification? Regime-Dependent Utility and Minority-Class Failure Analysis**

Alternative more diagnostic title:

**Synthetic Malware Augmentation Is Not Automatically Useful: Regime Dependence, Class Misalignment, and Downstream Evaluator Sensitivity**

## 3. Revised Abstract Thesis

The abstract should make five points:

1. Synthetic malware data is attractive for addressing data scarcity, imbalance, and sharing constraints.
2. Synthetic augmentation does not automatically improve downstream classification.
3. This paper compares balanced all-class augmentation and minority-heavy c4/c7 minority-only augmentation across GAN, VAE, and Diffusion families.
4. The results show that augmentation utility is regime-dependent: balanced augmentation can be beneficial or near-neutral, while minority-heavy augmentation does not reliably repair targeted minority classes.
5. Per-class diagnostics, precision--recall tradeoffs, external class-alignment auditing, and ResNet sensitivity analysis show that failure is caused by class-level instability, class misalignment, and evaluator dependence.

Core abstract sentence:

> Across generator families and budgets, we show that synthetic-data utility is conditional rather than automatic: it depends on the augmentation regime, the targeted classes, the downstream metric, the class alignment of generated samples, and the classifier used for evaluation.

## 4. Recommended Journal Paper Structure

### I. Introduction

Purpose:

Introduce synthetic malware augmentation as a useful but risky intervention.

Recommended paragraph flow:

1. Synthetic malware data is attractive for data scarcity, imbalance, and sharing constraints.
2. Utility is not guaranteed because more samples do not necessarily improve downstream decision boundaries.
3. Existing work often emphasizes generation quality or aggregate metrics without sufficiently diagnosing regime-specific and class-level failures.
4. This paper studies when synthetic augmentation helps or fails across balanced and minority-heavy regimes.
5. Main findings: balanced all-class augmentation can be helpful or near-neutral, minority-heavy c4/c7 augmentation does not reliably repair targeted classes, and failure is explained by precision--recall instability and synthetic class misalignment.
6. Contributions.

Recommended contributions:

- We evaluate synthetic malware augmentation across generator families, budgets, and augmentation regimes.
- We compare balanced all-class augmentation against minority-heavy c4/c7 minority-only augmentation.
- We show that minority-only augmentation does not reliably repair targeted minority classes.
- We provide per-class c4/c7 diagnostics and precision--recall tradeoff analysis to explain how failure appears.
- We introduce an external class-alignment audit to test whether generated minority samples are recognized as their requested classes.
- We perform a ResNet-style classifier sensitivity check showing that measured utility is evaluator-dependent.

### II. Related Work

Keep concise but journal-level.

Recommended subsections:

A. Synthetic Data for Malware Classification  
B. Generative Models for Data Augmentation  
C. Class Imbalance and Minority-Class Augmentation  
D. Downstream Utility and Class-Level Evaluation  
E. Class-Faithfulness and Conditioning Diagnostics

Important framing:

Do not make this section too long. Its purpose is to motivate why Paper 3 focuses on downstream utility, class-level diagnostics, and regime dependence.

### III. Experimental Protocol

This section should define the study clearly.

Recommended subsections:

A. Dataset and Malware-Image Representation  
B. Generator Families and Synthetic Budgets  
C. Augmentation Regimes  
D. Downstream Utility Metrics  
E. Per-Class and Precision--Recall Diagnostics  
F. External Synthetic Class-Alignment Audit  
G. Reproducibility and Seed Protocol

Key definitions:

Balanced all-class augmentation:

- all classes receive synthetic samples
- budgets include 25, 100, 500, 2000 where available
- primary focus in journal version can emphasize b2000 while still reporting budget trends

Minority-heavy c4/c7 augmentation:

- classes 4 and 7 are reduced to 20 percent of their original real-training samples
- synthetic samples are added only to classes 4 and 7
- budgets include 100, 500, 2000

Delta definition:

> Δ metric = Real+Synthetic minus Real-only.

### IV. Main Results: Regime-Dependent Utility

This should be the first results section.

Recommended subsections:

A. Balanced All-Class Augmentation Shows Family- and Budget-Dependent Utility  
B. Minority-Heavy c4/c7 Augmentation Does Not Reliably Help  
C. Balanced and Minority-Heavy Regimes Behave Differently

Main figures/tables:

- Fig. 1: Balanced all-class Macro-F1 vs budget
- Fig. 2: Minority-heavy Macro-F1 vs budget
- Fig. 3: Balanced vs minority-heavy b2000 comparison
- Table: balanced GAN/VAE/Diffusion aggregate
- Table: minority-heavy cross-family aggregate

Main claim:

> Synthetic utility depends on regime. Balanced augmentation and minority-heavy targeted augmentation should not be treated as equivalent interventions.

### V. Minority-Class Failure Analysis

This section explains whether the targeted minority classes were repaired.

Recommended subsections:

A. Class-Level ΔF1 for Targeted Minority Classes  
B. Precision--Recall Tradeoff Patterns

Main figures/tables:

- Fig. 4: Minority-class c4/c7 ΔF1 by family and budget
- Fig. 5: c4/c7 precision--recall tradeoff
- Table: c4/c7 per-class diagnostic table
- Table: precision--recall tradeoff table

Main claim:

> The minority-heavy failure is not only a macro-level artifact. The targeted minority classes themselves are not consistently repaired, and failure often appears as precision--recall tradeoff instability.

Important nuance:

- Class 4 is the stronger failure case.
- Class 7 is more mixed but does not show stable improvement.
- Some settings may improve one component, such as precision or recall, while reducing the other.

### VI. Synthetic Class-Alignment Audit

This should be a dedicated journal-level diagnostic section.

Purpose:

Explain why minority-only augmentation may fail.

Main figure/table:

- Fig. 6: c4/c7 synthetic class-alignment audit
- Table: synthetic alignment audit aggregate

Main claim:

> Many synthetic samples requested as class 4 or class 7 are not externally recognized as those classes by an independent real-only audit classifier.

Observed pattern:

- Diffusion c4/c7 samples are externally recognized as class 8.
- GAN c4/c7 samples are predominantly externally recognized as class 1.
- VAE c4 samples are externally recognized as class 7.
- VAE c7 samples align with class 7.

Careful wording:

This should be framed as an audit-classifier diagnostic, not absolute ground truth.

Recommended wording:

> The alignment audit suggests that minority-only augmentation may add samples under minority labels that do not behave like those minority classes under an independent classifier.

### VII. Classifier Sensitivity Analysis

This section should be shorter than the main results.

Purpose:

Show evaluator dependence.

Main table:

- ResNet sensitivity table

Main claim:

> Under a ResNet-style downstream classifier, augmentation effects become weaker and closer to neutral, showing that measured utility depends on the downstream classifier.

Correct interpretation:

Do not claim that ResNet exactly reproduces EvalCNN results.

Recommended wording:

> The ResNet sensitivity analysis does not reproduce the exact magnitude or direction of all EvalCNN deltas. Instead, it shows that synthetic augmentation effects are downstream-classifier sensitive. This reinforces the central claim that synthetic data should be evaluated under the intended downstream model and regime.

### VIII. Discussion

This should synthesize, not repeat tables.

Recommended discussion points:

1. Synthetic data is a conditional intervention.
2. Balanced augmentation and minority-only augmentation are different interventions.
3. Adding synthetic samples to minority classes does not guarantee minority-class repair.
4. Class-level diagnostics are necessary because macro metrics can hide targeted-class behavior.
5. Precision--recall tradeoffs reveal how failure appears.
6. Class-alignment auditing helps explain why failure occurs.
7. ResNet sensitivity shows that utility is evaluator-dependent.

Strong discussion sentence:

> The central lesson is that synthetic malware augmentation should be validated as a downstream intervention, not accepted as automatically beneficial because it increases class counts.

### IX. Threats to Validity / Limitations

Be honest and reviewer-ready.

Mention:

- The main evidence is based on USTC-TFC2016.
- The minority-heavy regime focuses on classes 4 and 7.
- The generator families are limited to GAN, VAE, and Diffusion.
- The ResNet sensitivity check uses one alternative classifier, not an exhaustive classifier benchmark.
- The alignment audit depends on the audit classifier and should be interpreted as diagnostic evidence.
- Results may vary with dataset, feature representation, imbalance severity, generator training quality, classifier architecture, and synthetic filtering policy.

Do not overstate universality.

Recommended limitation sentence:

> Our results should not be interpreted as proving that minority-only augmentation always fails, but rather that its utility is not guaranteed and must be validated under the target data regime, generator, and downstream classifier.

### X. Conclusion

Keep direct.

Recommended conclusion points:

1. Synthetic data can help, but its utility is not automatic.
2. Balanced all-class augmentation and minority-heavy minority-only augmentation behave differently.
3. Minority-heavy c4/c7 augmentation does not reliably repair targeted classes.
4. Failure is explained by class-level F1 degradation, precision--recall instability, and class-alignment problems.
5. ResNet sensitivity shows that measured utility is also downstream-classifier dependent.
6. Synthetic cybersecurity data should be evaluated as a regime-dependent, evaluator-dependent intervention.

## 5. Recommended Main-Paper Figures

Main paper should include:

1. Fig. 1: Balanced all-class Macro-F1 vs budget
2. Fig. 3: Balanced vs minority-heavy b2000 comparison
3. Fig. 4: Minority-class c4/c7 ΔF1
4. Fig. 6: Synthetic class-alignment audit

Optional main paper if space allows:

5. Fig. 5: Precision--recall tradeoff

Possible appendix/supplement:

- Fig. 2: Minority-heavy Macro-F1 vs budget if similar information is already captured in Fig. 3
- additional budget plots
- full predicted-label histograms
- full ResNet raw seed-level table

## 6. Recommended Main-Paper Tables

Main tables:

1. Balanced all-class budget aggregate
2. Minority-heavy cross-family aggregate
3. c4/c7 per-class diagnostic table
4. Synthetic class-alignment audit table
5. ResNet sensitivity table

If space is tight:

- Move precision--recall tradeoff table to appendix.
- Keep only summarized pattern counts in the main Discussion.

## 7. Claim Boundaries

Strong claims supported:

- Synthetic augmentation utility is regime-dependent.
- Minority-only augmentation under c4/c7 does not reliably repair targeted minority classes.
- Class-level diagnostics reveal failures hidden by aggregate metrics.
- Precision--recall tradeoffs help explain how minority-class failure appears.
- Alignment auditing shows that many requested minority samples are externally misaligned.
- Utility measurement is sensitive to downstream classifier architecture.

Claims to avoid:

- Synthetic data always helps.
- Minority-only augmentation always fails in all datasets.
- One generator family is universally best.
- Alignment audit predictions are absolute ground truth.
- ResNet exactly confirms the EvalCNN pattern.

## 8. Journal Contribution Framing

The strongest journal contribution is not only the experimental results. It is the evaluation framework:

> A regime-aware and class-aware evaluation protocol for synthetic malware augmentation that combines downstream utility, class-level diagnostics, precision--recall tradeoff analysis, synthetic class-alignment auditing, and classifier-sensitivity checking.

This makes the paper more than a dataset-specific result.

## 9. Next Writing Step

When drafting begins, write in this order:

1. Experimental Protocol
2. Main Results
3. Minority-Class Failure Analysis
4. Synthetic Class-Alignment Audit
5. Classifier Sensitivity Analysis
6. Discussion
7. Introduction
8. Abstract
9. Related Work
10. Conclusion

Reason:

The evidence is now clear. The paper should be built around the evidence, then the Introduction and Abstract should be written last to accurately reflect the final contribution.

## 10. Current Recommendation

Do not add more experiments immediately.

The next major step should be manuscript integration:

- reorganize Overleaf using this journal structure,
- decide which figures/tables stay in the main paper,
- move extra evidence to appendix/supplement,
- rewrite claims around conditional utility,
- and ensure every strong claim has a corresponding frozen evidence file.