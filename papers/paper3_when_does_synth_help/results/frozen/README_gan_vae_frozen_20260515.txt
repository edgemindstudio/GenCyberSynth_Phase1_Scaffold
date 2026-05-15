Paper 3 frozen evidence block: GAN + VAE balanced-budget regime on USTC-TFC2016.

Design:
- Dataset: USTC-TFC2016
- Model families: GAN, VAE
- Seeds: 42, 43, 44
- Synthetic budgets per class: 0, 25, 100, 500, 2000
- Metrics: delta Macro-F1, delta Balanced Accuracy, delta Macro-AUPRC
- Delta definition: Real+Synthetic minus Real-only
- Baseline budget 0 deltas set to 0.0

Frozen files:
- paper3_gan_vae_budget_results_frozen_20260515.csv
- paper3_gan_vae_budget_aggregate_frozen_20260515.csv
- paper3_gan_vae_budget_aggregate_table.md
- paper3_gan_vae_budget_aggregate_table.tex

Main finding:
Budget sensitivity is family-dependent. GAN shows its clearest improvement at the largest budget of 2000 synthetic samples per class, while VAE shows smaller, less monotonic effects and a negative dip at the medium budget of 500 samples per class for Macro-F1 and balanced accuracy.

Paper claim supported:
Synthetic data does not help simply because it is added. Its downstream utility depends on both generator family and augmentation budget.