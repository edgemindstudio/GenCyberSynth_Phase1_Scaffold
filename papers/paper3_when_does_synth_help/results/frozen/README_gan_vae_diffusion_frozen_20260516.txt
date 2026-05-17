Paper 3 frozen evidence block: GAN + VAE + Diffusion balanced-budget regime on USTC-TFC2016.

Design:
- Dataset: USTC-TFC2016
- Model families: GAN, VAE, Diffusion
- Seeds: 42, 43, 44
- Synthetic budgets per class: 0, 25, 100, 500, 2000
- Regime: balanced augmentation
- Metrics: delta Macro-F1, delta Balanced Accuracy, delta Macro-AUPRC
- Delta definition: Real+Synthetic minus Real-only
- Baseline budget 0 deltas set to 0.0

Frozen files:
- paper3_gan_vae_diffusion_budget_results_frozen_20260516.csv
- paper3_gan_vae_diffusion_budget_aggregate_frozen_20260516.csv
- paper3_gan_vae_diffusion_budget_aggregate_table.md
- paper3_gan_vae_diffusion_budget_aggregate_table.tex

Main finding:
Budget sensitivity is family-dependent. GAN and Diffusion show their clearest downstream gains at the largest tested budget of 2000 synthetic samples per class. VAE shows smaller and less monotonic effects, including a negative dip at 500 samples per class for Macro-F1 and balanced accuracy.

Paper claim supported:
Synthetic data utility is not automatic. It depends jointly on synthetic budget, generator family, and downstream evaluation metric.