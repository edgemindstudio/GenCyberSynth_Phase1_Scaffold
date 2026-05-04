# Paper 2 ACGAN Seed42 External Audit

## Setup

This run evaluates the Paper 2 ACGAN auxiliary-loss intervention after fixing config-scoped checkpoint routing.

Config:

- `paper2_acgan_auxloss.yaml`
- Seed: 42
- Epochs: 50
- Synthetic budget audited: 90 samples total

Checkpoint routing was config-scoped:

```text
/home/bruno.fonkeng/gencys/artifacts_paper2/gan/checkpoints/paper2_acgan_auxloss/seed42
```
Synthetic manifest audited:
```text
/home/bruno.fonkeng/gencys/artifacts_paper2/gan/synthetic/paper2_acgan_auxloss/seed42/manifest.json
```

## Training Observation

The ACGAN training objective appeared to converge internally. By epoch 50, the auxiliary class losses were very small:
```text
D_cls ≈ 0.0008
G_cls ≈ 0.0110
```
## External Conditioning Audit Result

The external real-only CNN audit still detected severe conditioning collapse:

- Total synthetic audited: `90`
- Overall conditioning accuracy: `0.1111`
- Conditioning failure rate: `0.8889`
- Leakage count: `80 / 90`
- Leakage rate: `0.8889`

Predicted-label histogram:
```text
predicted_label,count,fraction
0,0,0.0
1,0,0.0
2,0,0.0
3,0,0.0
4,0,0.0
5,0,0.0
6,0,0.0
7,90,1.0
8,0,0.0
```

## Interpretation

The ACGAN auxiliary-loss intervention did not repair class-conditioning under the external audit for seed 42. Although the internal discriminator class head reported very low class losses, the external real-only classifier judged all generated samples as class 7.

This suggests that internal auxiliary classifier loss can be satisfied without producing externally class-faithful synthetic malware images. Therefore, Paper 2 should emphasize the importance of external conditioning audits rather than relying only on internal GAN losses or discriminator auxiliary accuracy.

## Paper 2 Takeaway

The result strengthens the central Paper 2 claim:
```text
Conditioning mechanisms must be externally audited. 
Low internal class loss does not guarantee class-faithful synthetic generation.
```
