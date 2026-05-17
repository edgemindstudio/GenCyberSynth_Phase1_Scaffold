# Paper 3 Result Block 6: VAE Minority-Heavy Class-Imbalance Regime

Frozen on: 2026-05-17

## Design

- Dataset: USTC-TFC2016
- Family: VAE
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
| 100 | 3 | -0.0085 ± 0.0015 | -0.0062 ± 0.0032 | -0.0088 ± 0.0029 |
| 500 | 3 | -0.0075 ± 0.0028 | -0.0059 ± 0.0037 | -0.0084 ± 0.0040 |
| 2000 | 3 | -0.0072 ± 0.0061 | -0.0066 ± 0.0055 | -0.0074 ± 0.0019 |

## Main finding

Minority-only VAE augmentation does not repair the minority-heavy c4/c7 imbalance. Across three seeds, every tested synthetic budget produces negative mean deltas for Macro-F1, Balanced Accuracy, and Macro-AUPRC.

## Interpretation

This result strengthens Paper 3 because the minority-heavy failure is no longer isolated to GAN. VAE also fails under the same class-restricted augmentation policy, suggesting that the problem is tied to the regime and augmentation intervention, not only to one generator family.

## Paper claim supported

Synthetic augmentation is conditional and regime-dependent. Targeting minority classes alone is not sufficient to guarantee downstream improvement.