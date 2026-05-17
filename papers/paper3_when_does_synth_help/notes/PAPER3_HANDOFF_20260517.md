# Paper 3 Handoff Report — 2026-05-17

## Paper title

When Does Synthetic Data Help Malware Classification? Utility, Class Imbalance, and Failure Modes Across Augmentation Regimes

## Current branch

wip-metrics-fix

## Latest protected commit

66885f7 Draft Paper 3 results section skeleton

## Core research question

This paper asks when synthetic data helps malware classification and when it hurts. The current evidence focuses on downstream utility as a function of generator family, synthetic budget, and real-data class regime.

## Frozen evidence blocks

### Block 1: GAN balanced budget sensitivity

- Dataset: USTC-TFC2016
- Family: GAN
- Regime: balanced all-class augmentation
- Seeds: 42, 43, 44
- Budgets per class: 0, 25, 100, 500, 2000
- Main finding: small and medium budgets are unstable or near-neutral; b2000 gives the clearest positive downstream gain.

### Block 2: GAN + VAE balanced budget sensitivity

- Dataset: USTC-TFC2016
- Families: GAN, VAE
- Regime: balanced all-class augmentation
- Main finding: GAN responds more strongly at high budget; VAE effects are smaller and less monotonic.

### Block 3: GAN + VAE + Diffusion balanced budget sensitivity

- Dataset: USTC-TFC2016
- Families: GAN, VAE, Diffusion
- Regime: balanced all-class augmentation
- Main finding: GAN and Diffusion show clearer gains at b2000; VAE remains weaker.

### Block 4: GAN minority-heavy class-imbalance regime

- Dataset: USTC-TFC2016
- Family: GAN
- Regime: minority-heavy c4/c7
- Minority classes: 4 and 7
- Minority real-training fraction: 0.2
- Synthetic policy: minority-only augmentation
- Budgets per minority class: 0, 100, 500, 2000
- Seeds: 42, 43, 44
- Main finding: all tested minority-only GAN budgets have negative mean deltas for Macro-F1, Balanced Accuracy, and Macro-AUPRC.

### Block 5: GAN balanced vs minority-heavy comparison

- Main finding: the same GAN family helps under balanced high-budget augmentation but hurts under the minority-heavy c4/c7 regime.
- Interpretation: synthetic data utility depends on class structure and augmentation policy, not only on generator family or budget.

## Key frozen files

### Balanced multi-family block

- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_vae_diffusion_budget_results_frozen_20260516.csv
- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_vae_diffusion_budget_aggregate_frozen_20260516.csv
- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_vae_diffusion_budget_aggregate_table.md
- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_vae_diffusion_budget_aggregate_table.tex

### GAN minority-heavy block

- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_minority_heavy_results_frozen_20260517.csv
- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_minority_heavy_aggregate_frozen_20260517.csv
- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_minority_heavy_aggregate_table.md
- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_minority_heavy_aggregate_table.tex

### GAN balanced vs minority-heavy comparison

- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_balanced_vs_minority_heavy_comparison_20260517.csv
- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_balanced_vs_minority_heavy_comparison_table.md
- papers/paper3_when_does_synth_help/results/frozen/paper3_gan_balanced_vs_minority_heavy_comparison_table.tex

## Notes created

- notes/result_block1_gan_budget_sensitivity.md
- notes/result_block2_gan_vae_budget_sensitivity.md
- notes/result_block3_gan_vae_diffusion_budget_sensitivity.md
- notes/result_block4_gan_minority_heavy.md
- notes/result_block5_gan_balanced_vs_minority_heavy.md
- notes/results_section_skeleton_v1.md

## Code infrastructure added

### eval/runner.py

Added Paper 3 real-training subsampling support through `real_train_subsample`, allowing minority-heavy regimes to reduce selected real training classes before downstream utility evaluation.

### gan/sample.py

Added class-restricted GAN synthesis through:

```yaml
synth:
  class_ids: [4, 7]