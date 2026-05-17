# Paper 3 Result Block 3: GAN + VAE + Diffusion Budget Sensitivity

Frozen on: 2026-05-16

## Design

- Dataset: USTC-TFC2016
- Families: GAN, VAE, Diffusion
- Regime: balanced augmentation
- Seeds: 42, 43, 44
- Budgets per class: 0, 25, 100, 500, 2000
- Metrics: Δ Macro-F1, Δ Balanced Accuracy, Δ Macro-AUPRC
- Delta definition: Real+Synthetic minus Real-only

## Main finding

Budget sensitivity is family-dependent.

GAN and Diffusion show their clearest improvements at the largest budget of 2000 synthetic samples per class.

At 2000 samples/class:
- GAN: Δ Macro-F1 = 0.0064 ± 0.0036, Δ Balanced Accuracy = 0.0061 ± 0.0036, Δ Macro-AUPRC = 0.0036 ± 0.0019
- Diffusion: Δ Macro-F1 = 0.0060 ± 0.0039, Δ Balanced Accuracy = 0.0059 ± 0.0041, Δ Macro-AUPRC = 0.0039 ± 0.0025
- VAE: Δ Macro-F1 = 0.0012 ± 0.0010, Δ Balanced Accuracy = 0.0011 ± 0.0008, Δ Macro-AUPRC = 0.0004 ± 0.0010

## Interpretation

Small and medium budgets are unstable or near-neutral across families. The strongest utility gains appear at the largest tested budget. However, this gain is not uniform across model families. GAN and Diffusion respond more strongly to high-budget augmentation, while VAE remains weaker and less monotonic, including a negative dip at 500 samples per class for Macro-F1 and balanced accuracy.

## Paper claim supported

Synthetic augmentation should be treated as a regime-dependent intervention. Its usefulness depends jointly on budget, generator family, and downstream metric, rather than simply on the presence of additional synthetic samples.