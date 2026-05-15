# Paper 3 Result Block 1: GAN Budget Sensitivity

Evidence frozen on: 2026-05-14

## Design

- Dataset: USTC-TFC2016
- Model family: GAN
- Regime: balanced augmentation
- Seeds: 42, 43, 44
- Budgets per class: 0, 25, 100, 500, 2000
- Delta definition: Real+Synthetic minus Real-only

## Main aggregate result

Budget/Class | Runs | Δ Macro-F1 | Δ Bal. Acc. | Δ Macro-AUPRC
0 | 3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000
25 | 3 | 0.0008 ± 0.0018 | 0.0006 ± 0.0018 | -0.0006 ± 0.0018
100 | 3 | 0.0003 ± 0.0023 | 0.0000 ± 0.0022 | -0.0009 ± 0.0002
500 | 3 | -0.0003 ± 0.0010 | -0.0003 ± 0.0012 | -0.0001 ± 0.0016
2000 | 3 | 0.0064 ± 0.0036 | 0.0061 ± 0.0036 | 0.0036 ± 0.0019

## Interpretation

GAN augmentation is budget-sensitive. Small and medium budgets are unstable, near-neutral, or slightly harmful depending on the metric and seed. 
The largest tested budget, 2000 synthetic samples per class, gives the clearest positive downstream improvement across Macro-F1, balanced accuracy, and Macro-AUPRC.

## Paper claim supported

Synthetic data does not help simply because it is added. Its utility depends on the augmentation budget and evaluation regime.