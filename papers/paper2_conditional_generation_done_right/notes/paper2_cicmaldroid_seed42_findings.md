# Paper 2 CICMalDroid2020 Seed42 External Audit Finding



## Purpose



This note records the first full second-dataset experiment for Paper 2. The goal was to test whether fake-class-head ACGAN conditioning failure also appears on CICMalDroid2020, rather than only on USTC-TFC2016.



## Dataset and configuration



- Dataset: CICMalDroid2020

- Prepared data directory: `/home/bruno.fonkeng/gencys/data/CICMalDroid2020_pipeline_paper1`

- Image shape: 12 x 12 x 1

- Number of classes: 5

- Seed: 42

- Training epochs: 50

- Synthetic budget: 100 samples per class

- Total synthetic samples audited: 500

- Audit classifier: dataset-specific real-only CNN trained on CICMalDroid2020



## External audit result



- Conditioning accuracy: 0.244

- Conditioning failure rate: 0.756

- Leakage count: 378 / 500

- Leakage rate: 0.756



## Predicted-label histogram



- Class 0: 0 / 500

- Class 1: 301 / 500

- Class 2: 9 / 500

- Class 3: 190 / 500

- Class 4: 0 / 500



## Interpretation



The CICMalDroid2020 seed42 result shows substantial external conditioning failure. The failure is less extreme than the USTC-TFC2016 fake-class-head ACGAN results, where outputs often collapsed completely to one externally predicted class. However, the CICMalDroid result still shows strong collapse into a small subset of classes, mainly classes 1 and 3, with no samples externally predicted as classes 0 or 4.



This supports the journal-level claim that conditioning failure is not limited to USTC-TFC2016. The severity and collapse pattern are dataset-dependent, but the central failure mode persists on a second malware-image dataset.



## Journal-paper implication



This result strengthens Paper 2 by showing that external conditioning audits reveal label-fidelity failure across datasets. The paper should frame this as dataset-dependent conditioning failure: severe and near-complete collapse on USTC-TFC2016, and substantial but less complete collapse on CICMalDroid2020.
