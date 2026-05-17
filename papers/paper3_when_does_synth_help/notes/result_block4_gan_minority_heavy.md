# Paper 3 Result Block 4: GAN Minority-Heavy Class-Imbalance Regime

Frozen on: 2026-05-17

## Design

- Dataset: USTC-TFC2016
- Family: GAN
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
| 100 | 3 | -0.0107 ± 0.0040 | -0.0091 ± 0.0047 | -0.0077 ± 0.0025 |
| 500 | 3 | -0.0054 ± 0.0021 | -0.0046 ± 0.0023 | -0.0101 ± 0.0025 |
| 2000 | 3 | -0.0145 ± 0.0180 | -0.0121 ± 0.0160 | -0.0088 ± 0.0060 |

## Main finding

Minority-only GAN augmentation does not automatically repair a minority-heavy class imbalance. In this experiment, all tested synthetic budgets produce negative mean deltas for Macro-F1, Balanced Accuracy, and Macro-AUPRC across three seeds.

## Interpretation

This result strengthens Paper 3 because it gives a clear failure mode: even when synthetic data is targeted only at the minority classes, downstream utility can decrease. The result suggests that class-targeted augmentation is not enough by itself; the quality, diversity, and class alignment of the generated samples still matter.

## Paper claim supported

Synthetic augmentation should be treated as a regime-dependent intervention. Its usefulness depends jointly on the imbalance structure, augmentation policy, generator family, budget, and downstream metric.