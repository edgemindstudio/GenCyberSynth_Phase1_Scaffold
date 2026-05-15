# Paper 3 Result Block 2: GAN + VAE Budget Sensitivity

Frozen on: 2026-05-15

## Design

- Dataset: USTC-TFC2016
- Families: GAN and VAE
- Regime: balanced augmentation
- Seeds: 42, 43, 44
- Budgets per class: 0, 25, 100, 500, 2000
- Metrics: Δ Macro-F1, Δ Balanced Accuracy, Δ Macro-AUPRC
- Delta definition: Real+Synthetic minus Real-only

## Main table

GAN:
- Budget 25: Δ Macro-F1 = 0.0008 ± 0.0018, Δ Bal. Acc. = 0.0006 ± 0.0018, Δ Macro-AUPRC = -0.0006 ± 0.0018
- Budget 100: Δ Macro-F1 = 0.0003 ± 0.0023, Δ Bal. Acc. = 0.0000 ± 0.0022, Δ Macro-AUPRC = -0.0009 ± 0.0002
- Budget 500: Δ Macro-F1 = -0.0003 ± 0.0010, Δ Bal. Acc. = -0.0003 ± 0.0012, Δ Macro-AUPRC = -0.0001 ± 0.0016
- Budget 2000: Δ Macro-F1 = 0.0064 ± 0.0036, Δ Bal. Acc. = 0.0061 ± 0.0036, Δ Macro-AUPRC = 0.0036 ± 0.0019

VAE:
- Budget 25: Δ Macro-F1 = 0.0014 ± 0.0029, Δ Bal. Acc. = 0.0012 ± 0.0026, Δ Macro-AUPRC = 0.0000 ± 0.0028
- Budget 100: Δ Macro-F1 = 0.0005 ± 0.0014, Δ Bal. Acc. = 0.0002 ± 0.0014, Δ Macro-AUPRC = 0.0001 ± 0.0011
- Budget 500: Δ Macro-F1 = -0.0018 ± 0.0026, Δ Bal. Acc. = -0.0015 ± 0.0028, Δ Macro-AUPRC = 0.0002 ± 0.0020
- Budget 2000: Δ Macro-F1 = 0.0012 ± 0.0010, Δ Bal. Acc. = 0.0011 ± 0.0008, Δ Macro-AUPRC = 0.0004 ± 0.0010

## Interpretation

Budget sensitivity is family-dependent. GAN shows weak or unstable behavior at small and medium budgets but a clear positive improvement at 2000 samples per class. VAE shows smaller and less monotonic effects, with a negative dip at 500 samples per class for Macro-F1 and balanced accuracy.

## Paper claim supported

Synthetic data utility depends jointly on generator family and synthetic budget. Synthetic augmentation should be treated as a regime-dependent intervention, not as a universally beneficial preprocessing step.