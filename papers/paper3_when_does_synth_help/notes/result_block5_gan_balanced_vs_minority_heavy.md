# Paper 3 Result Block 5: Balanced vs Minority-Heavy GAN Comparison

Frozen on: 2026-05-17

## Design

This block compares two GAN augmentation regimes on USTC-TFC2016:

1. Balanced all-class augmentation:
   - Real training data remains balanced.
   - Synthetic samples are generated for all classes.
   - Budgets per class: 0, 25, 100, 500, 2000.
   - Seeds: 42, 43, 44.

2. Minority-heavy c4/c7 augmentation:
   - Real training data is made minority-heavy by reducing classes 4 and 7 to 20% of their original training count.
   - Synthetic samples are generated only for minority classes 4 and 7.
   - Budgets per minority class: 0, 100, 500, 2000.
   - Seeds: 42, 43, 44.

Metrics are reported as Real+Synthetic minus Real-only deltas for Macro-F1, Balanced Accuracy, and Macro-AUPRC.

## Main comparison

| Regime | Budget/Class | Runs | Δ Macro-F1 | Δ Bal. Acc. | Δ Macro-AUPRC |
|---|---:|---:|---:|---:|---:|
| balanced_all_class | 0 | 3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| minority_heavy_c4c7_minority_only | 0 | 3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| balanced_all_class | 25 | 3 | 0.0008 ± 0.0018 | 0.0006 ± 0.0018 | -0.0006 ± 0.0018 |
| balanced_all_class | 100 | 3 | 0.0003 ± 0.0023 | 0.0000 ± 0.0022 | -0.0009 ± 0.0002 |
| minority_heavy_c4c7_minority_only | 100 | 3 | -0.0107 ± 0.0040 | -0.0091 ± 0.0047 | -0.0077 ± 0.0025 |
| balanced_all_class | 500 | 3 | -0.0003 ± 0.0010 | -0.0003 ± 0.0012 | -0.0001 ± 0.0016 |
| minority_heavy_c4c7_minority_only | 500 | 3 | -0.0054 ± 0.0021 | -0.0046 ± 0.0023 | -0.0101 ± 0.0025 |
| balanced_all_class | 2000 | 3 | 0.0064 ± 0.0036 | 0.0061 ± 0.0036 | 0.0036 ± 0.0019 |
| minority_heavy_c4c7_minority_only | 2000 | 3 | -0.0145 ± 0.0180 | -0.0121 ± 0.0160 | -0.0088 ± 0.0060 |

## Main finding

GAN augmentation behaves differently across regimes. Under the balanced all-class regime, the largest budget of 2000 synthetic samples per class produces the clearest positive downstream gains. Under the minority-heavy c4/c7 regime, minority-only GAN augmentation produces negative mean deltas at all tested budgets.

## Interpretation

This comparison is central to Paper 3. It shows that synthetic data utility is not determined by generator family or budget alone. The same family, GAN, can help in one regime and hurt in another. Even targeted minority-only augmentation can reduce downstream utility when the synthetic samples do not improve the decision boundary for the imbalanced classifier.

## Paper claim supported

Synthetic data helps conditionally. Its benefit depends jointly on the real-data class structure, augmentation policy, synthetic budget, generator family, and downstream metric.