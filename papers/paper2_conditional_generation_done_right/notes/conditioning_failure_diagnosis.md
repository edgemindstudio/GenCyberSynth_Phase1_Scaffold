# Paper 2 Conditioning Failure Diagnosis

## Evidence Block 1: Audit Result

The baseline conditional GAN was audited using a real-only CNN classifier trained on real USTC-TFC2016 malware images.

Key audit results:

- Total synthetic samples audited: 225
- Overall conditioning accuracy: 0.1111
- Conditioning failure rate / leakage rate: 0.8889
- Predicted-label collapse: 213/225 samples predicted as class 7, 12/225 predicted as class 4, and 0 samples predicted as the remaining classes.
- Real-only audit classifier accuracy:
  - Validation accuracy: approximately 0.9446
  - Test accuracy: approximately 0.9426

## Code Inspection

The baseline cDCGAN passes labels into both the generator and discriminator.

Generator:

```python
z_in = layers.Input(shape=(latent_dim,), name="z")
y_in = layers.Input(shape=(num_classes,), name="y_onehot")
y_map = layers.Dense(5 * 5 * 1, use_bias=False)(y_in)
x = layers.Concatenate(axis=-1)([x, y_map])