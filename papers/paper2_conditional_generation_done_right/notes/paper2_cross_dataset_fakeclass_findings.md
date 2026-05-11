# Paper 2 Cross-Dataset Fake-Class ACGAN Findings



## Purpose



This note summarizes the cross-dataset fake-class-head ACGAN external audit results for Paper 2. The goal was to test whether the conditioning-failure finding persists beyond USTC-TFC2016 and appears on a second malware-image dataset, CICMalDroid2020.



## Datasets



### USTC-TFC2016



- Image shape: 40 x 40 x 1

- Number of classes: 9

- Audit budget: 100 synthetic samples per class

- Total audited per seed: 900



### CICMalDroid2020



- Image shape: 12 x 12 x 1

- Number of classes: 5

- Audit budget: 100 synthetic samples per class

- Total audited per seed: 500



## USTC-TFC2016 fake-class-head ACGAN results



- Seed 42: conditioning accuracy 0.1111, leakage 0.8889, full collapse to class 7

- Seed 43: conditioning accuracy 0.0444, leakage 0.9556, collapse into classes 2, 5, and 8

- Seed 44: conditioning accuracy 0.1111, leakage 0.8889, full collapse to class 8



## CICMalDroid2020 fake-class-head ACGAN results



- Seed 42: conditioning accuracy 0.2440, leakage 0.7560, collapse mostly into classes 1 and 3

- Seed 43: conditioning accuracy 0.2000, leakage 0.8000, full collapse to class 2

- Seed 44: conditioning accuracy 0.2080, leakage 0.7920, collapse mostly into class 1, with smaller class 0 and class 3 components



## Interpretation



The fake-class-head ACGAN exhibits severe external conditioning failure on both datasets. The failure is strongest on USTC-TFC2016, where outputs often collapse almost completely to one externally predicted class. On CICMalDroid2020, the failure is less extreme but still severe, with leakage between 0.756 and 0.800 across seeds.



The results support a dataset-dependent conditioning-failure claim. The exact collapse class and severity vary by dataset and seed, but the central failure mode persists: generated samples are not reliably recognized as their requested classes by a dataset-specific real-only audit classifier.



## Journal-paper implication



This cross-dataset evidence strengthens Paper 2 substantially. The paper can now argue that external conditioning audits reveal label-fidelity failures that are not limited to one dataset, one random seed, one audit budget, or one audit classifier.
