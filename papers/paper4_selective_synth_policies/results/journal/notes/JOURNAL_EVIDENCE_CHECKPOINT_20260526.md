# Paper 4 Journal Evidence Checkpoint

Date: 2026-05-26

Paper:
Selective Synthetic Data Policies for Malware Classification: Allocation, Filtering, and Acceptance Rules Under Utility Constraints

## Completed journal evidence layers

### USTC-TFC2016 evidence

1. Quality--utility diagnostics
- Tag: paper4-journal-quality-utility-20260522
- Main finding: KID shows moderate association with downstream utility, while MS-SSIM shows little association. However, neither proxy fully explains policy utility.

2. Class-repair allocation diagnostics
- Tag: paper4-journal-class-repair-allocation-20260522
- Main finding: Class-repair is a true adaptive allocation policy. Audit confidence is strongly negatively associated with allocated sample count.

3. Strict-confidence collapse diagnostics
- Tag: paper4-journal-strict-conf-collapse-20260522
- Main finding: Strict confidence filtering accepts only 1004 of 18000 samples and collapses requested-class coverage from 9 classes to 1 class.

4. Top-k budget sensitivity diagnostics
- Tag: paper4-journal-topk-budget-sensitivity-20260522
- Main finding: Top-k500 is too restrictive, Top-k1000 restores positive Macro-F1 and balanced-accuracy utility, and Class-repair improves Macro-AUPRC under the same 9000-sample budget.

### CICMalDroid2020 evidence

5. CICMalDroid keep-all validation
- Tag: paper4-journal-cicmaldroid-keepall-20260525
- Main finding: Keep-all b2000 decreases Macro-F1 and balanced accuracy on average, while Macro-AUPRC and calibration-oriented metrics improve on average.

6. CICMalDroid selective-policy validation
- Tag: paper4-journal-cicmaldroid-policies-20260526
- Main finding: Reduced-budget policies do not consistently reverse CICMalDroid utility harm. Top-k1000 is generally more favorable than Class-repair for Macro-F1 and balanced accuracy on CICMalDroid, while Class-repair improves calibration most strongly.

## Current cross-dataset interpretation

The two datasets support a stronger journal-level claim: selective synthetic-data policies are not universally beneficial. Their behavior depends on dataset, retained budget, class allocation structure, and downstream metric. USTC-TFC2016 shows that adaptive allocation can be useful under a reduced budget, while CICMalDroid2020 shows that the same policy family may fail to reverse negative balanced-accuracy effects.

## Journal-safe claims

- Synthetic-data selection should be treated as a policy-controlled intervention.
- Confidence filtering alone can destroy class coverage.
- Balanced top-k selection is sensitive to retained budget.
- Adaptive allocation changes selected-set structure under a fixed total budget.
- Class-repair is useful on USTC but is not universally best.
- CICMalDroid validates that policy effects are dataset-sensitive.
- Proxy metrics such as KID and MS-SSIM are useful diagnostics but insufficient as standalone policy selectors.
- Held-out downstream utility remains necessary.

## Claims to avoid

- Do not claim Class-repair universally improves performance.
- Do not claim selective filtering always beats keep-all.
- Do not claim confidence alone is a reliable measure of downstream usefulness.
- Do not claim CICMalDroid confirms the same ordering as USTC.
- Do not claim KID/MS-SSIM are useless; instead say they are insufficient as standalone selectors.

## Remaining possible journal upgrades

1. Add one additional generator family, preferably VAE, if time permits.
2. Add audit-classifier robustness, using alternate audit models from Paper 2 if available.
3. Add compact cross-dataset paper tables for USTC and CICMalDroid.
4. Write the expanded journal Results and Discussion sections.