Paper 3 frozen evidence block: GAN minority-heavy class-imbalance regime on USTC-TFC2016.

Design:
- Dataset: USTC-TFC2016
- Model family: GAN
- Regime: minority-heavy real-training imbalance
- Minority classes: 4 and 7
- Minority real-training fraction: 0.2
- Synthetic augmentation policy: minority-only
- Synthetic classes: 4 and 7
- Seeds: 42, 43, 44
- Synthetic budgets per minority class: 0, 100, 500, 2000
- Metrics: delta Macro-F1, delta Balanced Accuracy, delta Macro-AUPRC
- Delta definition: Real+Synthetic minus Real-only
- Baseline budget 0 deltas set to 0.0

Frozen files:
- paper3_gan_minority_heavy_results_frozen_20260517.csv
- paper3_gan_minority_heavy_aggregate_frozen_20260517.csv
- paper3_gan_minority_heavy_aggregate_table.md
- paper3_gan_minority_heavy_aggregate_table.tex

Main finding:
Under the minority-heavy c4/c7 imbalance regime, minority-only GAN augmentation does not automatically repair downstream performance. Across three seeds, all tested minority-only GAN budgets have negative mean deltas for Macro-F1, Balanced Accuracy, and Macro-AUPRC.

Paper claim supported:
Synthetic data utility depends not only on budget and generator family, but also on the class-imbalance regime and augmentation policy. Targeting minority classes alone can still hurt downstream utility when synthetic samples are not sufficiently beneficial for the classifier.