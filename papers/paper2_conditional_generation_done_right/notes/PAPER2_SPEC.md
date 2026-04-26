# Paper 2 Scientific Specification

## Working Title

Conditional Generation Done Right: Class-Conditional Evaluation, Conditioning Audits, and Minority-Class Failure Analysis for Synthetic Cybersecurity Images

## Dissertation Role

This paper extends Paper 1 by moving from global cross-family synthetic image evaluation to class-conditional evaluation. Paper 1 established that global generative metrics are not sufficient as standalone model-selection criteria. Paper 2 investigates whether conditional generators respect requested class labels and whether class-level or minority-class failures are hidden by global averages.

## Core Claim

A class-conditional cybersecurity image generator should not be evaluated only by global image-quality metrics. It must also be evaluated for label consistency, class-level fidelity, class-level diversity, downstream utility, and minority-class behavior.

## Primary Research Questions

1. Do conditional generators produce samples that match the requested class label?
2. Which classes are most vulnerable to conditional generation failure?
3. Do minority classes show lower fidelity, lower diversity, or weaker downstream utility?
4. Can global generative metrics hide severe class-level failures?
5. Does class-conditional augmentation improve minority-class malware detection compared with real-only baselines?

## Planned Evidence Blocks

### 1. Conditioning Audit

Generate synthetic samples using requested labels, then evaluate whether a frozen real-trained classifier predicts the same labels.

Expected outputs:

- Requested-label versus predicted-label confusion matrix
- Overall conditioning accuracy
- Per-class conditioning accuracy
- Class leakage rate
- Minority-class conditioning failure rate

### 2. Per-Class Generative Quality

Evaluate fidelity and diversity at the class level instead of relying only on global metrics.

Expected outputs:

- Per-class KID or equivalent fidelity score
- Per-class diversity summary
- Per-class nearest-neighbor or memorization audit, if feasible
- Global-versus-class-level comparison table

### 3. Downstream Utility

Compare real-only training against real-plus-synthetic training.

Expected outputs:

- Per-class precision
- Per-class recall
- Per-class F1
- Macro-F1
- Balanced accuracy
- Macro-AUPRC
- Minority-class recall and F1 deltas

### 4. Failure Analysis

Identify classes and model families where conditional generation fails.

Expected outputs:

- Failure taxonomy
- Class-level degradation table
- Examples of label leakage or class confusion
- Minority-class failure report

## Initial Model Scope

Start with a small, stable set of conditional generators before expanding.

Recommended initial models:

1. Conditional GAN / CDGAN
2. Conditional VAE / cVAE
3. Conditional diffusion model, if implementation cost is manageable

Additional families may be added only after the audit pipeline is stable.

## Dataset Scope

Primary dataset:

- USTC-TFC2016 malware image dataset
- Shape: 40 x 40 x 1
- Classes: 9

Optional second dataset:

- CICMalDroid 2020 image-style dataset
- Shape: 12 x 12 x 1
- Classes: 5

The first implementation should stabilize on the USTC-TFC2016 anchor dataset before expanding to CICMalDroid.

## Reproducibility Rules

1. Use fixed seeds.
2. Use frozen train/validation/test splits.
3. Store every run config.
4. Store every synthetic manifest.
5. Store per-class sample counts.
6. Store audit outputs separately from downstream classifier outputs.
7. Do not write Slurm logs to the repository root.
8. Do not commit raw Slurm `.out` or `.err` files unless selected examples are required for evidence.

## Paper 2 Success Criteria

Paper 2 is successful if it produces evidence that shows whether conditional generation failures are visible only when evaluation is performed at the class level.

The strongest outcome would be a clear demonstration that global synthetic-quality metrics can look acceptable while conditional label consistency, minority-class fidelity, or downstream per-class utility fails.