# Paper 2 Fake-Class-Head ACGAN Seed42 External Audit

## Setup

This run evaluates the Paper 2 fake-class-head ACGAN intervention.

Config:

- `paper2_acgan_fakeclass.yaml`
- Seed: 42
- Epochs: 50
- Synthetic budget audited: 225 samples total
- `train_fake_class_head: true`

Checkpoint routing was config-scoped:

```text
/home/bruno.fonkeng/gencys/artifacts_paper2/gan/checkpoints/paper2_acgan_fakeclass/seed42
```

Synthetic manifest audited:
```text
/home/bruno.fonkeng/gencys/artifacts_paper2/gan/synthetic/paper2_acgan_fakeclass/seed42/manifest.json
```

## Training Observation

The fake-class-head ACGAN run completed 50 epochs. Internal class losses became very small by the end of training:
```text
D_cls ≈ 0.0008
G_cls ≈ 0.0056
```
This suggests that the internal auxiliary classification objective was successfully optimized.

## Downstream Utility Result
The evaluation reported small positive downstream utility deltas:
```text
delta_macro_f1 = +0.002533858951926038
delta_bal_acc = +0.002061589805833264
delta_macro_auprc = +0.00009866930940127805
```
This suggests that the synthetic samples may still provide limited augmentation utility.

## External Conditioning Audit Result

The external real-only CNN audit still detected severe conditioning collapse:

- Total synthetic audited: `225`
- Overall conditioning accuracy: `0.1111`
- Conditioning failure rate: `0.8889`
- Leakage count: `200 / 225`
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
7,225,1.0
8,0,0.0
```

## Interpretation

Training the discriminator class head on fake samples did not repair external class faithfulness for seed 42. Although the internal class losses became very small, the independent real-only classifier judged all generated samples as class 7.

This strengthens the Paper 2 claim that internal GAN conditioning losses are insufficient evidence of class-faithful generation. A model can improve downstream utility slightly while still failing class-conditional semantic fidelity.

## Paper 2 Takeaway

The result supports the central Paper 2 argument:

```text
Synthetic cybersecurity image generators require external 
conditioning audits because internal conditioning objectives 
can look successful while generated samples remain class-collapsed 
under an independent classifier.
```