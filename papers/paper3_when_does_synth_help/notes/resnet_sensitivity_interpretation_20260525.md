# Paper 3 ResNet Classifier-Sensitivity Interpretation

Date: 2026-05-25  
Paper: When Does Synthetic Data Help Malware Classification? Utility, Class Imbalance, and Failure Modes Across Augmentation Regimes

## 1. Purpose of the Sensitivity Check

The ResNet-style classifier-sensitivity analysis was added to test whether the Paper 3 conclusions depend entirely on the original lightweight downstream EvalCNN used in the main evaluation pipeline.

The question is:

> Do the main augmentation conclusions remain meaningful when the downstream classifier architecture changes?

This is important because synthetic-data utility is not an intrinsic property of the generated samples alone. Utility is measured through an interaction between:

- the real training distribution,
- the synthetic intervention,
- the downstream classifier,
- the evaluation metric,
- the class regime,
- and the random seed.

## 2. Experimental Scope

The sensitivity check used:

- Dataset: USTC-TFC2016
- Alternative classifier: ResNet-style CNN
- Families: GAN, VAE, Diffusion
- Regimes:
  - balanced all-class augmentation
  - minority-heavy c4/c7 augmentation
- Budget: 2000 synthetic samples per relevant class
- Seeds: 42, 43, 44
- Metrics:
  - Macro-F1
  - Balanced Accuracy
  - Macro-AUPRC
  - Class-4 F1
  - Class-7 F1

Evidence files:

- `papers/paper3_when_does_synth_help/results/raw/paper3_resnet_sensitivity_raw.csv`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_aggregate_frozen_20260524.csv`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_table.md`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_table.tex`
- `papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_summary_20260524.json`

## 3. Main Observed Pattern

Under the ResNet-style classifier, the augmentation effects shrink toward near-neutral values.

Balanced augmentation at budget 2000 shows slightly negative or near-neutral Macro-F1 deltas:

- Diffusion: -0.0043 ± 0.0020
- GAN: -0.0046 ± 0.0026
- VAE: -0.0007 ± 0.0019

Minority-heavy c4/c7 augmentation at budget 2000 is also near-neutral or mixed:

- Diffusion: 0.0012 ± 0.0034
- GAN: -0.0010 ± 0.0160
- VAE: 0.0010 ± 0.0023

This means the ResNet check does not simply reproduce the exact main EvalCNN pattern. Instead, it shows that the magnitude and direction of synthetic augmentation effects can change under a different downstream classifier.

## 4. Correct Scientific Interpretation

The correct interpretation is not:

> The exact same gains and failures appear under every downstream classifier.

The stronger and more honest interpretation is:

> Synthetic augmentation effects are classifier-sensitive. Under a ResNet-style downstream classifier, the augmentation effects become weaker and closer to neutral, reinforcing the central claim that synthetic-data utility is conditional and must be validated under the intended downstream model.

This strengthens Paper 3 because it shows that synthetic-data usefulness should not be assumed from:

- generator family,
- synthetic sample count,
- balanced class counts,
- or generative plausibility alone.

Instead, usefulness depends on the downstream evaluation setting.

## 5. How This Fits the Paper's Main Argument

The original Paper 3 evidence shows strong regime dependence:

- balanced all-class augmentation can help under some settings,
- minority-heavy c4/c7 augmentation can fail,
- c4/c7 per-class diagnostics show targeted minority classes are not reliably repaired,
- precision--recall tradeoffs explain part of the failure,
- class-alignment audit shows many synthetic minority samples are not externally recognized as the requested class.

The ResNet sensitivity result adds another layer:

> Even when the same synthetic interventions are evaluated with another classifier, the effects are not automatically stable. The downstream model changes the measured utility.

Therefore, the paper's broader thesis becomes stronger:

> Synthetic augmentation is a regime-dependent and evaluator-dependent intervention. It requires direct downstream validation and class-level diagnostics before it can be trusted.

## 6. Recommended Manuscript Framing

In the journal version, the ResNet sensitivity check should be framed as a robustness and limitation analysis, not as the main result.

Recommended wording:

> To test whether the observed utility patterns are tied to a single downstream classifier, we repeated the high-budget balanced and minority-heavy comparisons using a ResNet-style classifier. The resulting deltas were smaller and closer to neutral than those observed with the main evaluator, indicating that augmentation utility is sensitive to downstream architecture. This finding does not contradict the main results; rather, it reinforces the paper's central argument that synthetic data should be treated as an intervention whose value must be validated under the intended classifier and regime.

## 7. Implication for Claims

Use cautious claim language:

Good:

- "The ResNet sensitivity check shows that augmentation effects are downstream-classifier sensitive."
- "The main conclusion should be interpreted as regime- and evaluator-dependent rather than universal."
- "Synthetic augmentation should be validated under the intended downstream classifier."

Avoid:

- "The exact same results hold under ResNet."
- "Balanced augmentation always helps."
- "Minority-heavy augmentation always hurts."
- "Generator family alone determines utility."

## 8. Recommended Placement in Paper

Best location:

- Discussion section, under a subsection such as:
  - `Classifier Sensitivity`
  - `Evaluator Dependence`
  - `Robustness to Downstream Architecture`

Alternative:

- Threats to Validity / Limitations section, with a short mention in Results.

Recommended journal-level use:

- Include the ResNet sensitivity table in the main paper if space allows.
- Otherwise include it in an appendix/supplement and summarize the key result in Discussion.

## 9. Current Conclusion

The ResNet sensitivity analysis should be used to support this final journal-level statement:

> The value of synthetic malware augmentation is not universal. It depends on the augmentation regime, target classes, generator family, synthetic class alignment, and downstream classifier. Therefore, synthetic cybersecurity data should be evaluated as a task-specific intervention rather than accepted as automatically beneficial.