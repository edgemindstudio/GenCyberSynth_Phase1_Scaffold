# Paper 2 Final Evidence Index



## Core paper claim



Paper 2 studies whether class-conditional malware-image generators actually produce samples that are externally recognizable as their requested classes.



The main finding is that fake-class-head ACGAN outputs exhibit severe external conditioning failure. The failure persists across audit budget, random seed, audit classifier, and dataset, although severity and predicted collapse targets are dataset-dependent.



## Evidence Block 1: USTC-TFC2016 fake-class ACGAN seed stability



Primary table:

- `results/tables/paper2_fakeclass_seed_stability.csv`



Key result:

- Seed 42: severe collapse

- Seed 43: severe collapse

- Seed 44: severe collapse



## Evidence Block 2: USTC-TFC2016 audit-budget stability



Primary table:

- `results/tables/paper2_fakeclass_budget_stability.csv`



Key result:

- Increasing from 25/class to 100/class does not remove the conditioning failure.



## Evidence Block 3: Multi-classifier audit robustness



Primary table:

- `results/tables/paper2_audit_classifier_comparison.csv`



Supporting note:

- `notes/paper2_multiclf_audit_findings.md`



Key result:

- Baseline CNN, ResNet-style CNN, and linear SVM-flat all report severe conditioning failure.



## Evidence Block 4: CICMalDroid2020 second-dataset validation



Primary files:

- `results/tables/paper2_gan_conditioning_summary_paper2_cicmaldroid_fakeclass_seed42.json`

- `results/tables/paper2_gan_conditioning_summary_paper2_cicmaldroid_fakeclass_seed43.json`

- `results/tables/paper2_gan_conditioning_summary_paper2_cicmaldroid_fakeclass_seed44.json`



Supporting note:

- `notes/paper2_cicmaldroid_seed42_findings.md`



Key result:

- CICMalDroid2020 also shows severe conditioning failure, but less extreme than USTC-TFC2016.



## Evidence Block 5: Cross-dataset stability



Primary table:

- `results/tables/paper2_cross_dataset_fakeclass_stability.csv`



Supporting note:

- `notes/paper2_cross_dataset_fakeclass_findings.md`



Key result:

- Conditioning failure persists across USTC-TFC2016 and CICMalDroid2020.



## Paper-writing implication



The Results section should be reorganized around evidence blocks instead of chronological experiments:



1. Baseline external audit result

2. Seed stability

3. Audit-budget stability

4. Multi-classifier robustness

5. Cross-dataset validation

6. Summary of practical implications
