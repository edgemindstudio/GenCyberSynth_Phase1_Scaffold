# Paper 3 Results Section Skeleton v1

## 4. Results

### 4.1 Balanced-budget augmentation shows family-dependent utility

We first evaluate synthetic augmentation under a balanced all-class regime on USTC-TFC2016. In this setting, all real classes remain balanced and synthetic samples are added uniformly across classes. Across GAN, VAE, and Diffusion, the effect of synthetic augmentation is budget- and family-dependent.

At small and medium budgets, downstream gains are generally weak, unstable, or near-neutral. The clearest positive gains appear at the largest tested budget of 2000 synthetic samples per class. At this budget, GAN and Diffusion show the strongest improvements, while VAE shows smaller gains.

This result indicates that synthetic data utility is not automatic. Adding synthetic samples does not consistently improve downstream malware classification unless the budget and generator family produce samples that help the classifier.

### 4.2 GAN budget sensitivity under balanced augmentation

For GAN under the balanced all-class regime, the largest budget produces the clearest downstream improvement. At 2000 synthetic samples per class, GAN improves Macro-F1, Balanced Accuracy, and Macro-AUPRC on average across three seeds.

However, smaller budgets do not show a consistent positive pattern. This suggests that GAN augmentation may require sufficient synthetic volume before producing measurable downstream benefit under this dataset and evaluator configuration.

### 4.3 VAE shows weaker and less monotonic augmentation effects

Compared with GAN, VAE produces smaller and less monotonic gains. In the balanced regime, VAE improves only weakly at the largest budget and shows a negative dip at 500 samples per class for Macro-F1 and Balanced Accuracy.

This suggests that model family matters. A generator family may produce synthetic samples that are valid enough to evaluate but not useful enough to improve downstream decision boundaries.

### 4.4 Diffusion shows high-budget gains similar to GAN

Diffusion follows a pattern closer to GAN than VAE. Small and medium budgets are near-neutral or unstable, while the largest budget produces clearer positive downstream gains.

This supports the broader claim that budget sensitivity is family-dependent: different generator families respond differently to the same synthetic budget schedule.

### 4.5 Minority-heavy imbalance changes the utility of GAN augmentation

We next evaluate GAN under a minority-heavy real-training imbalance. Classes 4 and 7 are reduced to 20% of their original training counts, while the validation and test sets remain unchanged. Synthetic samples are then added only to the minority classes 4 and 7.

Unlike the balanced regime, minority-only GAN augmentation does not repair performance. Across three seeds, all tested minority-only budgets produce negative mean deltas for Macro-F1, Balanced Accuracy, and Macro-AUPRC.

This result provides a clear failure mode: targeting synthetic data at minority classes is not sufficient by itself. If the synthetic samples do not improve the classifier’s representation of those classes, augmentation can hurt downstream utility.

### 4.6 Balanced versus minority-heavy comparison

The contrast between balanced and minority-heavy GAN regimes is central to this paper. Under balanced all-class augmentation, GAN at 2000 samples per class improves downstream performance. Under minority-heavy c4/c7 augmentation, GAN at 2000 samples per minority class hurts downstream performance on average.

Therefore, the usefulness of synthetic data depends not only on the generator family and budget, but also on the real-data class structure and augmentation policy. The same generator family can help in one regime and hurt in another.

## 5. Main empirical takeaways

1. Synthetic data utility is conditional, not automatic.
2. Budget matters, but larger budgets are not universally beneficial.
3. Generator family matters: GAN and Diffusion are stronger than VAE under the balanced regime.
4. Real-data class structure matters: balanced and minority-heavy regimes lead to different outcomes.
5. Targeted minority-only augmentation can still hurt if generated samples do not improve downstream decision boundaries.