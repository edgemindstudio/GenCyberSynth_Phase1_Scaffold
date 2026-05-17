# When Does Synthetic Data Help Malware Classification?
## Utility, Class Imbalance, and Failure Modes Across Augmentation Regimes

## Abstract

Synthetic malware images are often used to augment limited training data, but additional synthetic samples do not automatically improve downstream classification. This paper studies when synthetic data helps malware classification and when it hurts. We evaluate synthetic augmentation on USTC-TFC2016 under two regimes: a balanced all-class augmentation regime and a minority-heavy class-imbalance regime involving classes 4 and 7. Across GAN, VAE, and Diffusion generators, we measure downstream utility using Macro-F1, Balanced Accuracy, and Macro-AUPRC, reporting deltas between Real+Synthetic and Real-only training across three random seeds.

In the balanced all-class regime, augmentation effects are budget- and family-dependent. GAN and Diffusion show their clearest gains at the largest tested budget of 2000 synthetic samples per class, while VAE produces weaker and less monotonic effects. In contrast, under the minority-heavy c4/c7 regime, minority-only augmentation hurts downstream performance across GAN, VAE, and Diffusion for all tested budgets. These results show that synthetic data utility is conditional on generator family, budget, real-data class structure, augmentation policy, and downstream metric. The findings caution against treating synthetic augmentation as automatically beneficial and support direct task-level evaluation before using synthetic data for malware classification.

## 1. Introduction

Synthetic data generation has become an attractive strategy for malware classification because real malware datasets can be limited, imbalanced, sensitive, or difficult to expand. Generative models such as GANs, VAEs, and Diffusion models can produce additional image-like malware samples, which may appear useful for augmenting classifier training. However, the presence of more training samples does not guarantee better downstream performance.

The central question of this paper is: when does synthetic data help malware classification, and when does it hurt?

This question matters because synthetic augmentation is often treated as a simple intervention: generate more samples, add them to the training set, and expect better classification performance. In practice, the effect of synthetic data may depend on several interacting factors, including the generator family, the synthetic budget, the class distribution of the real data, the augmentation policy, and the metric used to evaluate downstream utility.

We study this question using USTC-TFC2016 malware images and evaluate three generator families: GAN, VAE, and Diffusion. We compare two augmentation regimes. The first is a balanced all-class regime, where synthetic samples are added evenly across all classes. The second is a minority-heavy c4/c7 regime, where real training samples from classes 4 and 7 are reduced to 20% of their original count and synthetic samples are added only to those minority classes.

Our results show a clear contrast. Under balanced all-class augmentation, GAN and Diffusion provide their strongest improvements at the largest tested budget of 2000 samples per class, while VAE remains weaker. Under minority-heavy c4/c7 augmentation, however, GAN, VAE, and Diffusion all produce negative mean deltas across all tested budgets and main metrics. This demonstrates that targeted minority-only augmentation is not automatically beneficial.

The contributions of this paper are:

1. A controlled downstream utility study of synthetic malware image augmentation across generator families, budgets, and class regimes.
2. A balanced all-class budget analysis showing that utility gains are family- and budget-dependent.
3. A minority-heavy class-imbalance analysis showing that minority-only augmentation can hurt across GAN, VAE, and Diffusion.
4. A regime-level interpretation showing that synthetic data utility depends on the interaction among generator family, budget, class structure, augmentation policy, and metric.

## 2. Experimental Design

### 2.1 Dataset

We use USTC-TFC2016 represented as grayscale malware images with shape 40 × 40 × 1 and 9 class labels. The downstream classifier is evaluated on held-out real test data. All reported results use three seeds: 42, 43, and 44.

### 2.2 Generator families

We evaluate three synthetic data generator families:

- GAN
- VAE
- Diffusion

Each family is evaluated under a common downstream utility protocol. The purpose is not only to compare image generation quality, but to determine whether the generated samples improve classifier performance when added to real training data.

### 2.3 Balanced all-class augmentation regime

In the balanced regime, the real training set remains balanced and synthetic samples are added uniformly across all classes. We evaluate budgets of 0, 25, 100, 500, and 2000 synthetic samples per class.

The budget 0 condition is the Real-only baseline. For augmentation budgets, the downstream utility delta is computed as:

Real+Synthetic minus Real-only.

### 2.4 Minority-heavy c4/c7 regime

In the minority-heavy regime, real training samples from classes 4 and 7 are reduced to 20% of their original training count. Validation and test sets remain unchanged. Synthetic samples are then added only to classes 4 and 7.

We evaluate budgets of 0, 100, 500, and 2000 synthetic samples per minority class.

This design tests whether targeted minority-only augmentation can repair a controlled class-imbalance condition.

### 2.5 Metrics

We report three primary downstream utility metrics:

- Macro-F1
- Balanced Accuracy
- Macro-AUPRC

All values are reported as deltas relative to Real-only training. Positive values indicate that synthetic augmentation improved downstream performance; negative values indicate that augmentation hurt downstream performance.

## 3. Results

### 3.1 Balanced all-class augmentation is family- and budget-dependent

The balanced all-class results show that synthetic augmentation is not uniformly beneficial. At small and medium budgets, effects are near-neutral or unstable across families. The clearest positive gains appear at 2000 synthetic samples per class, especially for GAN and Diffusion.

At 2000 samples per class, GAN improves Macro-F1 by 0.0064 ± 0.0036, Balanced Accuracy by 0.0061 ± 0.0036, and Macro-AUPRC by 0.0036 ± 0.0019. Diffusion shows a similar pattern, improving Macro-F1 by 0.0060 ± 0.0039, Balanced Accuracy by 0.0059 ± 0.0041, and Macro-AUPRC by 0.0039 ± 0.0025. VAE produces weaker gains, with Macro-F1 improving by only 0.0012 ± 0.0010 at the same budget.

These results suggest that synthetic data can help, but the effect depends on both the generator family and the synthetic budget.

### 3.2 VAE is weaker and less monotonic under balanced augmentation

Compared with GAN and Diffusion, VAE shows weaker and less monotonic behavior. It has small positive gains at some budgets but also a negative dip at 500 samples per class for Macro-F1 and Balanced Accuracy.

This indicates that not all generator families contribute equally useful synthetic samples, even when evaluated under the same dataset, budget schedule, and downstream classifier protocol.

### 3.3 Minority-heavy GAN augmentation fails despite targeting minority classes

Under the minority-heavy c4/c7 regime, GAN minority-only augmentation produces negative mean deltas across all tested budgets.

At 100 samples per minority class, GAN changes Macro-F1 by -0.0107 ± 0.0040, Balanced Accuracy by -0.0091 ± 0.0047, and Macro-AUPRC by -0.0077 ± 0.0025. At 2000 samples per minority class, GAN remains negative, with Macro-F1 changing by -0.0145 ± 0.0180.

This result shows that targeting synthetic samples at minority classes does not automatically repair class imbalance.

### 3.4 VAE minority-heavy augmentation also fails

VAE shows the same negative direction under the minority-heavy regime. Across all tested budgets, VAE minority-only augmentation produces negative mean deltas for Macro-F1, Balanced Accuracy, and Macro-AUPRC.

At 100 samples per minority class, VAE changes Macro-F1 by -0.0085 ± 0.0015, Balanced Accuracy by -0.0062 ± 0.0032, and Macro-AUPRC by -0.0088 ± 0.0029. At 2000 samples per minority class, VAE remains negative across all three metrics.

This shows that the minority-heavy failure is not isolated to GAN.

### 3.5 Diffusion minority-heavy augmentation also fails

Diffusion is one of the stronger families under the balanced all-class regime, but it also fails under the minority-heavy c4/c7 regime. At 100 samples per minority class, Diffusion changes Macro-F1 by -0.0089 ± 0.0073, Balanced Accuracy by -0.0074 ± 0.0069, and Macro-AUPRC by -0.0075 ± 0.0032. At 2000 samples per minority class, Diffusion remains negative.

This result is especially important because it shows that even a family that helps under balanced high-budget augmentation can hurt under a different real-data class structure and augmentation policy.

### 3.6 Cross-family minority-heavy comparison

The minority-heavy cross-family comparison shows a consistent pattern: GAN, VAE, and Diffusion all produce negative mean deltas under minority-only augmentation.

This supports the main claim of the paper: synthetic augmentation is regime-dependent. The usefulness of synthetic data depends not only on the generator family, but also on the real-data class structure and the augmentation policy.

## 4. Discussion

### 4.1 Synthetic data utility is conditional, not automatic

The results show that synthetic data can help under some conditions and hurt under others. In the balanced all-class regime, GAN and Diffusion show useful gains at high budget. In the minority-heavy regime, all three tested families hurt downstream performance despite being targeted at the minority classes.

Therefore, the presence of synthetic data should not be treated as evidence of utility. Synthetic samples must be evaluated through downstream task performance.

### 4.2 Budget matters, but larger budgets are not universally better

In the balanced regime, the largest budget produces the clearest gains for GAN and Diffusion. However, in the minority-heavy regime, larger budgets do not solve the problem. This shows that budget interacts with regime and augmentation policy.

A larger number of synthetic samples can help only when those samples provide useful decision-boundary information to the downstream classifier.

### 4.3 Minority-only augmentation can hurt

The minority-heavy results show that adding synthetic samples only to underrepresented classes can still reduce performance. This may happen if synthetic samples are low-quality, insufficiently diverse, poorly aligned with the intended class, or distributionally mismatched with the real minority samples.

This finding is important because class-targeted augmentation is often assumed to be beneficial. Our results show that targeted augmentation should still be validated empirically.

### 4.4 Generator family alone does not determine utility

Diffusion helps under balanced high-budget augmentation but hurts under minority-heavy augmentation. This means that generator family alone is not enough to predict downstream benefit. The same family can help in one regime and hurt in another.

## 5. Threats to Validity and Limitations

This study focuses on USTC-TFC2016 and a controlled malware-image representation. Results may differ on other datasets, feature representations, or downstream classifier architectures. The minority-heavy regime focuses specifically on classes 4 and 7 and uses a 20% real-training fraction. Other imbalance structures may lead to different outcomes.

The study evaluates three generator families and a fixed set of budgets. Additional generator variants, quality filters, conditioning audits, or selective policies may change the observed utility. However, the current evidence is sufficient to show that synthetic augmentation should not be assumed beneficial without downstream validation.

## 6. Conclusion

This paper studied when synthetic data helps malware classification and when it hurts. Under balanced all-class augmentation, GAN and Diffusion show their clearest gains at the largest tested budget, while VAE is weaker and less monotonic. Under a minority-heavy c4/c7 regime, however, minority-only augmentation hurts across GAN, VAE, and Diffusion.

The main conclusion is that synthetic data utility is conditional. It depends on the interaction among generator family, synthetic budget, real-data class structure, augmentation policy, and downstream metric. Synthetic augmentation should therefore be treated as an intervention that requires direct downstream validation, not as an automatically beneficial data expansion strategy.