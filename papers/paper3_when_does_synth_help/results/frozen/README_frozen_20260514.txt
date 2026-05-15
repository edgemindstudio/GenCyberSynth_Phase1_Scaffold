
Paper 3 frozen evidence block: GAN balanced-budget regime on USTC-TFC2016.



Design:

- Model: GAN

- Dataset: USTC-TFC2016

- Seeds: 42, 43, 44

- Synthetic budgets per class: 0, 25, 100, 500, 2000

- Metrics: delta Macro-F1, delta Balanced Accuracy, delta Macro-AUPRC

- Delta definition: Real+Synthetic minus Real-only

- Baseline budget 0 deltas set to 0.0



Frozen files:

- paper3_gan_budget_results_frozen_20260514.csv

- paper3_gan_budget_aggregate_frozen_20260514.csv

- paper3_delta_macro_f1_vs_budget.png

- paper3_delta_bal_acc_vs_budget.png

- paper3_delta_macro_auprc_vs_budget.png



Main finding:

GAN augmentation is budget-sensitive. Small and medium budgets are unstable or near-neutral, while budget 2000 per class produces the clearest positive downstream gain across seeds.

