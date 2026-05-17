# Paper 3 Result Block 7: Diffusion Minority-Heavy Class-Imbalance Regime

Frozen on: 2026-05-17

## Design

- Dataset: USTC-TFC2016
- Family: Diffusion
- Regime: minority-heavy real-training imbalance
- Minority classes: 4 and 7
- Minority real-training fraction: 0.2
- Synthetic augmentation policy: minority-only
- Synthetic classes: 4 and 7
- Seeds: 42, 43, 44
- Budgets per minority class: 0, 100, 500, 2000
- Metrics: Δ Macro-F1, Δ Balanced Accuracy, Δ Macro-AUPRC
- Delta definition: Real+Synthetic minus Real-only

## Aggregate result

| Budget/Class | Runs | Δ Macro-F1 | Δ Bal. Acc. | Δ Macro-AUPRC |
|---:|---:|---:|---:|---:|
| 0 | 3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 100 | 3 | -0.0089 ± 0.0073 | -0.0074 ± 0.0069 | -0.0075 ± 0.0032 |
| 500 | 3 | -0.0075 ± 0.0052 | -0.0065 ± 0.0041 | -0.0079 ± 0.0019 |
| 2000 | 3 | -0.0108 ± 0.0106 | -0.0096 ± 0.0095 | -0.0070 ± 0.0016 |

## Main finding

Minority-only Diffusion augmentation does not repair the minority-heavy c4/c7 imbalance. Across three seeds, every tested synthetic budget produces negative mean deltas for Macro-F1, Balanced Accuracy, and Macro-AUPRC.

## Interpretation

This result is important because Diffusion was one of the stronger families under the balanced all-class regime, yet it still fails under the minority-heavy c4/c7 regime. This strengthens the paper’s claim that augmentation utility depends on the interaction between generator family, real-data class structure, and augmentation policy.

## Paper claim supported

Synthetic augmentation is conditional and regime-dependent. Even stronger generator families can hurt downstream utility when the augmentation policy and imbalance regime are unfavorable.