# Paper 2 Evidence Tracker

## Evidence Blocks

| Evidence block | Config | Seed | Epochs | Audit tag | Conditioning accuracy | Leakage rate | Predicted-label collapse | Interpretation |
|---|---:|---:|---:|---|---:|---:|---|---|
| Baseline smoke | paper2_smoke | 42 | 2 | v2 / baseline smoke | 0.1111 | 0.8889 | mostly class 7 | Label injection alone fails |
| ACGAN smoke | paper2_acgan_smoke | 42 | 1 | paper2_acgan_smoke | ~0.12 | ~0.88 | collapsed | App pipeline works, not scientific |
| ACGAN full pre-routing | paper2_acgan_auxloss | 42 | 50 | paper2_acgan_auxloss_seed42 | 0.1111 | 0.8889 | class 8 | Ambiguous because checkpoints were shared |
| ACGAN full clean v2 | paper2_acgan_auxloss | 42 | 50 | paper2_acgan_auxloss_seed42_v2 | 0.1111 | 0.8889 | class 7 | Internal class loss did not guarantee external fidelity |

## Current Interpretation

The current evidence suggests that both label injection and internal auxiliary class loss are insufficient by themselves. External class-conditioning audits are necessary because internal GAN objectives can report strong class performance while generated samples remain semantically collapsed under an independent classifier.