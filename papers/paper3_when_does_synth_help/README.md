# Paper 3: When Does Synthetic Data Help Malware Classification?

**Working title:** *When Does Synthetic Data Help Malware Classification? Utility, Class Imbalance, and Failure Modes Across Augmentation Regimes*

## 1. Purpose of Paper 3

Paper 3 extends the GenCyberSynth dissertation line from controlled benchmark comparison into practical augmentation behavior. Paper 1 showed that synthetic augmentation effects are dataset-dependent and that downstream utility must remain the primary evaluation lens. Paper 3 asks a more targeted question:

> Under what conditions does synthetic data help malware classification, and when does it hurt?

This paper should move beyond asking whether one generator family is better than another. Its goal is to study augmentation benefit as a function of:

- synthetic budget,
- random seed stability,
- class imbalance regime,
- dataset structure,
- generator family,
- downstream utility metrics,
- and potentially conditional-quality or classwise failure measures.

The paper is intended to become one of the practically important papers in the dissertation because it answers a question that researchers and practitioners actually face: **how much synthetic data should be added, under what regime, and when should synthetic augmentation be avoided?**

## 2. Dissertation Connection

This paper belongs to the core GenCyberSynth dissertation arc:

> Trustworthy probabilistic and generative machine learning for synthetic cybersecurity data, malware-image classification, and risk-aware security decision-making.

Within the dissertation structure:

- **Paper 1** establishes the repaired, reproducible, seed-aware GenCyberSynth benchmark.
- **Paper 2** studies whether class-conditional generators actually preserve requested labels under external audit.
- **Paper 3** studies when synthetic data helps or hurts downstream malware classification under augmentation regimes.
- **Paper 4** can build from Paper 3 by proposing selective allocation, filtering, or acceptance policies.

Paper 3 is therefore the bridge between benchmark evidence and policy design. It should produce the empirical foundation for later work on selective synthetic-data policies, classwise reliability, robustness, calibration, and uncertainty-aware decisions.

## 3. Main Research Questions

Paper 3 should answer the following research questions:

1. **Budget sensitivity:** How does downstream utility change as the number of synthetic samples per class increases?
2. **Seed stability:** Are augmentation gains consistent across random seeds, or are they unstable?
3. **Metric agreement:** Do Macro-F1, balanced accuracy, and Macro-AUPRC agree about whether synthetic data helps?
4. **Failure modes:** At which budgets or regimes does synthetic augmentation hurt downstream performance?
5. **Family effects:** Do different generator families show different budget-response behavior?
6. **Imbalance effects:** Does synthetic data help more under class-imbalanced regimes than balanced regimes?
7. **Dataset effects:** Do augmentation conclusions transfer across datasets, or are they dataset-specific?

## 4. Current Experimental Scope

The current completed evidence block focuses on:

- **Dataset:** USTC-TFC2016 malware images
- **Image shape:** 40 × 40 × 1 grayscale
- **Model family:** GAN
- **Regime:** balanced augmentation
- **Seeds:** 42, 43, 44
- **Budgets per class:** 0, 25, 100, 500, 2000
- **Primary downstream metrics:**
  - Macro-F1
  - Balanced accuracy
  - Macro-AUPRC
- **Delta definition:**
  - Real+Synthetic minus Real-only

## 5. Current Artifact Root and Paper Module

Repository root:

```text
/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold
```

Paper module:

```text
papers/paper3_when_does_synth_help/
```

Paper 3 artifact root:

```text
/home/bruno.fonkeng/gencys/artifacts_paper3
```

Important result folders:

```text
papers/paper3_when_does_synth_help/results/raw/
papers/paper3_when_does_synth_help/results/figures/
papers/paper3_when_does_synth_help/results/frozen/
papers/paper3_when_does_synth_help/notes/
```

## 6. Important Code and Scripts

### Collection and table-building scripts

```text
papers/paper3_when_does_synth_help/scripts/paper3_collect_all_summaries.py
papers/paper3_when_does_synth_help/scripts/paper3_dedup_results.py
papers/paper3_when_does_synth_help/scripts/paper3_filter_for_paper.py
papers/paper3_when_does_synth_help/scripts/paper3_make_figures_v1.py
```

Purpose:

1. collect all summary JSON files,
2. deduplicate repeated runs,
3. filter to paper-valid seeds and regimes,
4. regenerate figures from the final paper CSV.

### Important runtime/evaluation files

```text
app/main.py
eval/runner.py
eval/runner.patched.py
adapters/gan_adapter.py
gan/train.py
gan/sample.py
gan/pipeline.py
gan/models.py
```

These files control the train → synth → eval workflow, GAN sampling, manifest writing, local image loading, downstream utility evaluation, and summary writing.

## 7. Key Fixes Already Completed

The following issues were fixed during Paper 3 execution:

### 7.1 Manifest path resolution

The evaluator previously failed to load synthetic images correctly when manifest entries used relative paths. The evaluation runner was updated so synthetic paths are resolved correctly against the synthetic root. This fixed the problem where `imgs_for_util` and `labels_for_util` were missing or `None`.

### 7.2 Summary metadata preservation

The evaluation summary now preserves:

- `config_id`,
- `run_meta`,
- `budget_per_class`,
- and `deltas_RS_minus_R`.

This matters because Paper 3 aggregation depends on knowing which budget, seed, and regime each summary belongs to.

### 7.3 Delta writing

The evaluator now writes `deltas_RS_minus_R` both at the top level of the JSON summary and inside the utility block. This allows downstream collection scripts to extract Macro-F1, balanced accuracy, and Macro-AUPRC deltas reliably.

### 7.4 Seed placement

GAN configs must use top-level `SEED: 42`, `SEED: 43`, or `SEED: 44`. Placing the seed only under `synth:` was not sufficient because the GAN adapter reads the top-level `SEED`.

### 7.5 Checkpoint path issue

GAN synthesis expected checkpoint directories such as:

```text
/home/bruno.fonkeng/gencys/artifacts_paper3/gan/checkpoints/seed43
/home/bruno.fonkeng/gencys/artifacts_paper3/gan/checkpoints/seed44
```

but training used config-scoped checkpoint paths. Symlinks or consistent checkpoint path handling were used so sampling could find `G_last.weights.h5` correctly.

## 8. Completed Evidence Block: GAN Budget Sensitivity

The first completed and frozen evidence block is:

> GAN balanced-budget augmentation on USTC-TFC2016 across seeds 42, 43, and 44.

### 8.1 Frozen evidence files

Frozen folder:

```text
papers/paper3_when_does_synth_help/results/frozen/
```

Current frozen files:

```text
paper3_delta_bal_acc_vs_budget.png
paper3_delta_macro_auprc_vs_budget.png
paper3_delta_macro_f1_vs_budget.png
paper3_gan_budget_aggregate_frozen_20260514.csv
paper3_gan_budget_aggregate_table.md
paper3_gan_budget_aggregate_table.tex
paper3_gan_budget_results_frozen_20260514.csv
README_frozen_20260514.txt
```

### 8.2 Seed-level evidence

The clean seed-level table is:

```text
papers/paper3_when_does_synth_help/results/raw/paper3_results_paper_clean.csv
```

It contains:

```text
3 seeds × 5 budgets = 15 rows
```

Expected seed-budget combinations are complete:

```text
seeds: 42, 43, 44
budgets: 0, 25, 100, 500, 2000
missing combinations: none
extra combinations: none
```

### 8.3 Budget-level aggregate result

The budget-level aggregate table is:

```text
papers/paper3_when_does_synth_help/results/raw/paper3_budget_aggregate.csv
papers/paper3_when_does_synth_help/results/frozen/paper3_gan_budget_aggregate_frozen_20260514.csv
```

Current aggregate result:

| Budget/Class | Runs | Δ Macro-F1 | Δ Bal. Acc. | Δ Macro-AUPRC |
|---:|---:|---:|---:|---:|
| 0 | 3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 25 | 3 | 0.0008 ± 0.0018 | 0.0006 ± 0.0018 | -0.0006 ± 0.0018 |
| 100 | 3 | 0.0003 ± 0.0023 | 0.0000 ± 0.0022 | -0.0009 ± 0.0002 |
| 500 | 3 | -0.0003 ± 0.0010 | -0.0003 ± 0.0012 | -0.0001 ± 0.0016 |
| 2000 | 3 | 0.0064 ± 0.0036 | 0.0061 ± 0.0036 | 0.0036 ± 0.0019 |

### 8.4 Main finding from completed block

GAN augmentation is budget-sensitive. Small and medium budgets are unstable, near-neutral, or slightly harmful depending on seed and metric. The largest tested budget, 2000 synthetic samples per class, produces the clearest positive downstream utility gain across Macro-F1, balanced accuracy, and Macro-AUPRC.

This supports one of Paper 3's central claims:

> Synthetic data does not help simply because it is added. Its benefit depends on the augmentation budget and evaluation regime.

## 9. Commands Used to Rebuild the Current Evidence

### 9.1 Collect, deduplicate, filter, and regenerate figures

```bash
ARTS=/home/bruno.fonkeng/gencys/artifacts_paper3
python papers/paper3_when_does_synth_help/scripts/paper3_collect_all_summaries.py --artifacts "$ARTS" --out papers/paper3_when_does_synth_help/results/raw/paper3_results_all.csv
python papers/paper3_when_does_synth_help/scripts/paper3_dedup_results.py --in_csv papers/paper3_when_does_synth_help/results/raw/paper3_results_all.csv --out_csv papers/paper3_when_does_synth_help/results/raw/paper3_results_dedup.csv
python papers/paper3_when_does_synth_help/scripts/paper3_filter_for_paper.py --in_csv papers/paper3_when_does_synth_help/results/raw/paper3_results_dedup.csv --out_csv papers/paper3_when_does_synth_help/results/raw/paper3_results_paper.csv
python papers/paper3_when_does_synth_help/scripts/paper3_make_figures_v1.py --in_csv papers/paper3_when_does_synth_help/results/raw/paper3_results_paper_clean.csv --out_dir papers/paper3_when_does_synth_help/results/figures
```

### 9.2 Clean real-only delta rows

```bash
python - <<'PY'
import pandas as pd
p="papers/paper3_when_does_synth_help/results/raw/paper3_results_paper.csv"
out="papers/paper3_when_does_synth_help/results/raw/paper3_results_paper_clean.csv"
df=pd.read_csv(p)
mask = df["synth_budget_per_class"].astype(int).eq(0)
for col in ["delta_macro_f1", "delta_bal_acc", "delta_macro_auprc"]:
    df.loc[mask, col] = 0.0
df.to_csv(out, index=False)
print("[ok] wrote", out)
print("rows:", len(df))
print("missing delta_macro_f1:", df["delta_macro_f1"].isna().sum())
print("missing delta_bal_acc:", df["delta_bal_acc"].isna().sum())
print("missing delta_macro_auprc:", df["delta_macro_auprc"].isna().sum())
PY
```

### 9.3 Create aggregate table

```bash
python - <<'PY'
import pandas as pd
p="papers/paper3_when_does_synth_help/results/raw/paper3_results_paper_clean.csv"
out="papers/paper3_when_does_synth_help/results/raw/paper3_budget_aggregate.csv"
df=pd.read_csv(p)
agg = df.groupby("synth_budget_per_class").agg(
    runs=("seed", "count"),
    delta_macro_f1_mean=("delta_macro_f1", "mean"),
    delta_macro_f1_std=("delta_macro_f1", "std"),
    delta_bal_acc_mean=("delta_bal_acc", "mean"),
    delta_bal_acc_std=("delta_bal_acc", "std"),
    delta_macro_auprc_mean=("delta_macro_auprc", "mean"),
    delta_macro_auprc_std=("delta_macro_auprc", "std"),
).reset_index().sort_values("synth_budget_per_class")
agg.to_csv(out, index=False)
print("[ok] wrote", out)
print(agg.to_string(index=False))
PY
```

### 9.4 Freeze evidence

```bash
mkdir -p papers/paper3_when_does_synth_help/results/frozen
cp papers/paper3_when_does_synth_help/results/raw/paper3_results_paper_clean.csv papers/paper3_when_does_synth_help/results/frozen/paper3_gan_budget_results_frozen_20260514.csv
cp papers/paper3_when_does_synth_help/results/raw/paper3_budget_aggregate.csv papers/paper3_when_does_synth_help/results/frozen/paper3_gan_budget_aggregate_frozen_20260514.csv
cp papers/paper3_when_does_synth_help/results/figures/paper3_delta_macro_f1_vs_budget.png papers/paper3_when_does_synth_help/results/frozen/
cp papers/paper3_when_does_synth_help/results/figures/paper3_delta_bal_acc_vs_budget.png papers/paper3_when_does_synth_help/results/frozen/
cp papers/paper3_when_does_synth_help/results/figures/paper3_delta_macro_auprc_vs_budget.png papers/paper3_when_does_synth_help/results/frozen/
```

## 10. What Has Been Achieved

### Completed

- Paper 3 direction clarified around augmentation regimes and downstream utility.
- Paper 3 module exists under the GenCyberSynth repository.
- GAN balanced-budget configs exist for budgets 0, 25, 100, 500, and 2000.
- Seeds 42, 43, and 44 have been executed for the GAN balanced-budget regime.
- Evaluation bug involving missing Real+Synthetic utility was fixed.
- Manifest path resolution was fixed.
- `config_id` preservation was fixed.
- `deltas_RS_minus_R` writing was fixed.
- Paper-valid summaries were collected from artifact JSON files.
- Results were deduplicated and filtered to the correct seeds/regimes.
- Clean seed-level CSV was created with no missing delta values.
- Budget-level aggregate table was created.
- Three budget-vs-delta figures were regenerated.
- First evidence block was frozen and documented.
- Paper-ready Markdown and LaTeX table files were created.

### Current completed scientific claim

> Under the balanced USTC-TFC2016 GAN regime, synthetic augmentation is budget-sensitive. Small and medium budgets are unstable or near-neutral, while 2000 synthetic samples per class provides the clearest positive downstream gain across seeds and metrics.

## 11. What Is Yet to Be Achieved

### Immediate next tasks

1. Create a notes file summarizing Result Block 1.
2. Write the first Results subsection: **Budget Sensitivity of GAN Augmentation**.
3. Improve the figure style if needed for publication readiness.
4. Add a table caption and figure captions for the paper.
5. Decide whether the next experiment block should extend across generator families or across imbalance regimes.

### Required future evidence blocks

Paper 3 is not complete with only GAN balanced-budget results. The following blocks are still needed or should be considered:

#### A. More generator families

Repeat the budget-regime study for additional generator families, such as:

- VAE,
- diffusion,
- autoregressive,
- Masked AutoFlow,
- Gaussian Mixture,
- Restricted Boltzmann Machine.

This would answer: **for which families does synthetic data help?**

#### B. Imbalance regimes

Create controlled imbalance regimes and evaluate whether synthetic data helps more under imbalance than under balanced training.

Possible regimes:

- balanced,
- mild imbalance,
- moderate imbalance,
- severe imbalance,
- minority-class-targeted augmentation.

This would answer: **for which imbalance regimes does synthetic data help?**

#### C. Classwise analysis

Aggregate metrics may hide class-level harm. Paper 3 should eventually include:

- per-class F1 deltas,
- per-class recall deltas,
- minority vs majority class effects,
- classes helped vs classes hurt.

This would answer: **which classes benefit and which classes are harmed?**

#### D. Dataset comparison

Paper 1 already showed dataset-dependent augmentation effects. Paper 3 should eventually test whether the same budget/imbalance conclusions hold on a second dataset, such as CICMalDroid2020.

This would answer: **do augmentation rules transfer across datasets?**

#### E. Failure-mode analysis

Paper 3 should identify failure modes such as:

- too little synthetic data has no effect,
- medium budgets may add noise without enough useful coverage,
- high budgets may help only for some families,
- synthetic data may improve Macro-F1 while hurting Macro-AUPRC,
- aggregate gains may hide minority-class harm.

#### F. Statistical and robustness framing

For publication quality, the paper should avoid overclaiming from small deltas. It should report:

- mean ± standard deviation across seeds,
- seed-level scatter plots,
- aggregate budget curves,
- and cautious language around practical rather than absolute significance.

## 12. Recommended Next Step

The recommended next step is to write the first completed Results subsection from the frozen evidence block:

```text
A. Budget Sensitivity of GAN Augmentation
```

This subsection should use:

- `paper3_gan_budget_aggregate_table.tex`,
- `paper3_delta_macro_f1_vs_budget.png`,
- `paper3_delta_bal_acc_vs_budget.png`,
- `paper3_delta_macro_auprc_vs_budget.png`,
- and the frozen CSV files as evidence.

After that, the next experimental decision should be:

> Should Paper 3 next expand across generator families, or should it first create imbalance regimes for GAN?

The stronger paper path is likely:

1. finish GAN budget sensitivity section,
2. add at least one or two additional generator families under the same budget design,
3. then add imbalance regimes or classwise analysis.

This will make Paper 3 more than a single-family study and better align with its title: *When Does Synthetic Data Help Malware Classification?*
