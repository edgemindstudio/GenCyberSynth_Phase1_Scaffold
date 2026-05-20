#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


POLICY_ORDER = [
    "paper4_realonly_balanced",
    "paper4_baseline_b25",
    "paper4_baseline_b500",
    "paper4_baseline_b2000",
    "paper4_policy_keep_all_b25",
    "paper4_policy_keep_all_b500",
    "paper4_policy_keep_all_b2000",
    "paper4_policy_confidence_accept_t080_b2000",
    "paper4_policy_confidence_ranked_topk500_b2000",
    "paper4_policy_confidence_ranked_topk1000_b2000",
]

POLICY_LABELS = {
    "paper4_realonly_balanced": "Real-only",
    "paper4_baseline_b25": "Baseline b25",
    "paper4_baseline_b500": "Baseline b500",
    "paper4_baseline_b2000": "Baseline b2000",
    "paper4_policy_keep_all_b25": "keep_all b25",
    "paper4_policy_keep_all_b500": "keep_all b500",
    "paper4_policy_keep_all_b2000": "keep_all b2000",
    "paper4_policy_confidence_accept_t080_b2000": "confidence_accept t080",
    "paper4_policy_confidence_ranked_topk500_b2000": "confidence_ranked topk500",
    "paper4_policy_confidence_ranked_topk1000_b2000": "confidence_ranked topk1000",
}

METRICS = ["delta_macro_f1", "delta_bal_acc", "delta_macro_auprc"]


def policy_label(config_id: str) -> str:
    return POLICY_LABELS.get(config_id, config_id)


def sort_policy_frame(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {name: i for i, name in enumerate(POLICY_ORDER)}
    out = df.copy()
    out["_order"] = out["config_id"].map(order_map).fillna(999).astype(int)
    sort_cols = ["_order", "config_id"]
    if "seed" in out.columns:
        sort_cols.append("seed")
    out = out.sort_values(sort_cols, kind="stable")
    return out.drop(columns=["_order"])


def make_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config_id, g in sort_policy_frame(df).groupby("config_id", sort=False):
        row = {
            "config_id": config_id,
            "policy_label": policy_label(config_id),
            "n_seeds": int(g["seed"].nunique()),
            "seeds": ",".join(str(int(x)) for x in sorted(g["seed"].dropna().unique())),
            "budget_per_class": int(g["budget_per_class"].dropna().iloc[0]) if g["budget_per_class"].notna().any() else None,
            "num_fake_mean": float(g["num_fake"].mean()) if "num_fake" in g else None,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(g[metric].mean()) if g[metric].notna().any() else None
            row[f"{metric}_std"] = float(g[metric].std()) if g[metric].notna().sum() > 1 else None
            row[f"{metric}_mean_pm_std"] = format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_std(mean, std) -> str:
    if mean is None or pd.isna(mean):
        return ""
    if std is None or pd.isna(std):
        return f"{mean:.6f}"
    return f"{mean:.6f} ± {std:.6f}"


def write_markdown_table(df: pd.DataFrame, out_path: Path):
    cols = [
        "policy_label",
        "n_seeds",
        "budget_per_class",
        "num_fake_mean",
        "delta_macro_f1_mean_pm_std",
        "delta_bal_acc_mean_pm_std",
        "delta_macro_auprc_mean_pm_std",
    ]
    tmp = df[cols].copy()
    tmp = tmp.rename(columns={
        "policy_label": "Policy",
        "n_seeds": "Seeds",
        "budget_per_class": "Nominal/selected budget",
        "num_fake_mean": "Mean # fake",
        "delta_macro_f1_mean_pm_std": "Delta Macro-F1",
        "delta_bal_acc_mean_pm_std": "Delta Balanced Acc.",
        "delta_macro_auprc_mean_pm_std": "Delta Macro-AUPRC",
    })
    tmp = tmp.fillna("")
    headers = list(tmp.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in tmp.iterrows():
        vals = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def make_bar_figure(mean_df: pd.DataFrame, metric: str, out_dir: Path):
    plot_df = mean_df[mean_df[f"{metric}_mean"].notna()].copy()
    plot_df = plot_df[plot_df["config_id"] != "paper4_realonly_balanced"]
    plot_df = sort_policy_frame(plot_df)
    labels = plot_df["policy_label"].tolist()
    values = plot_df[f"{metric}_mean"].tolist()
    errors = plot_df[f"{metric}_std"].fillna(0.0).tolist()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values, yerr=errors, capsize=4)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(metric.replace("delta_", "Δ ").replace("_", " "))
    ax.set_title(metric.replace("delta_", "Policy effect on ").replace("_", " "))
    fig.tight_layout()

    png = out_dir / f"paper4_{metric}_by_policy.png"
    pdf = out_dir / f"paper4_{metric}_by_policy.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print("[ok] wrote", png)
    print("[ok] wrote", pdf)


def make_budget_tradeoff(mean_df: pd.DataFrame, out_dir: Path):
    plot_df = mean_df[mean_df["delta_macro_f1_mean"].notna()].copy()
    plot_df = plot_df[plot_df["config_id"] != "paper4_realonly_balanced"]
    plot_df["num_fake_mean"] = pd.to_numeric(plot_df["num_fake_mean"], errors="coerce")
    plot_df = plot_df.dropna(subset=["num_fake_mean", "delta_macro_f1_mean"])
    plot_df = sort_policy_frame(plot_df)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.scatter(plot_df["num_fake_mean"], plot_df["delta_macro_f1_mean"])
    for _, row in plot_df.iterrows():
        ax.annotate(row["policy_label"], (row["num_fake_mean"], row["delta_macro_f1_mean"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Mean synthetic samples used")
    ax.set_ylabel("Mean Δ Macro-F1")
    ax.set_title("Paper 4 utility-budget tradeoff")
    fig.tight_layout()

    png = out_dir / "paper4_budget_utility_tradeoff_macro_f1.png"
    pdf = out_dir / "paper4_budget_utility_tradeoff_macro_f1.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print("[ok] wrote", png)
    print("[ok] wrote", pdf)


def make_notes(mean_df: pd.DataFrame, out_path: Path):
    def get(config_id, metric):
        rows = mean_df[mean_df["config_id"] == config_id]
        if rows.empty:
            return ""
        return rows.iloc[0].get(f"{metric}_mean_pm_std", "")

    text = f"""# Paper 4 Result Interpretation Notes

## Current evidence state

The current paper-facing result table contains the baseline, keep_all identity-policy, strict confidence-acceptance, and confidence-ranked top-k policy results.

## Main policy findings

1. The keep_all identity policy validates that policy-specific manifests can be routed through the same downstream evaluation pipeline.

2. The strict confidence_accept_t080 policy is diagnostic rather than practical. It exposes that strict same-label confidence filtering can collapse the accepted synthetic set.

3. The confidence_ranked_topk500 policy preserves class balance but is too restrictive on average.

4. The confidence_ranked_topk1000 policy preserves class balance and produces positive macro-F1 and balanced-accuracy gains across seeds while using half the full b2000 synthetic volume.

## Mean ± SD summary

- Baseline b500 Δ Macro-F1: {get("paper4_baseline_b500", "delta_macro_f1")}
- Baseline b2000 Δ Macro-F1: {get("paper4_baseline_b2000", "delta_macro_f1")}
- keep_all b2000 Δ Macro-F1: {get("paper4_policy_keep_all_b2000", "delta_macro_f1")}
- confidence_ranked topk500 Δ Macro-F1: {get("paper4_policy_confidence_ranked_topk500_b2000", "delta_macro_f1")}
- confidence_ranked topk1000 Δ Macro-F1: {get("paper4_policy_confidence_ranked_topk1000_b2000", "delta_macro_f1")}

## Safe paper claim

Selective synthetic-data policies are not automatically superior to keeping all generated samples. Strict confidence filtering can collapse accepted samples. Balanced confidence ranking avoids collapse, but its utility depends on the retained budget. In the current results, topk1000 preserves positive utility with half the full b2000 synthetic volume, while the full keep_all b2000 setting remains strongest on average.

## Claims to avoid

- Do not claim confidence ranking universally improves augmentation.
- Do not claim topk1000 beats full b2000 keep_all.
- Do not claim confidence_accept_t080 is a good augmentation policy.
- Do not claim policy quality can be judged from confidence alone; downstream utility remains necessary.
"""
    out_path.write_text(text)
    print("[ok] wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="papers/paper4_selective_synth_policies/results/raw/paper4_results_paper.csv")
    ap.add_argument("--tables_dir", default="papers/paper4_selective_synth_policies/results/tables")
    ap.add_argument("--figures_dir", default="papers/paper4_selective_synth_policies/results/figures")
    ap.add_argument("--notes_dir", default="papers/paper4_selective_synth_policies/results/notes")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    notes_dir = Path(args.notes_dir)

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    df = sort_policy_frame(df)

    seed_level_cols = [
        "config_id", "seed", "budget_per_class", "num_fake",
        "macro_f1_real_only", "macro_f1_real_plus_synth", "delta_macro_f1",
        "bal_acc_real_only", "bal_acc_real_plus_synth", "delta_bal_acc",
        "macro_auprc_real_only", "macro_auprc_real_plus_synth", "delta_macro_auprc",
        "kid", "ms_ssim",
    ]
    seed_cols = [c for c in seed_level_cols if c in df.columns]
    seed_level = df[seed_cols].copy()
    seed_level.insert(1, "policy_label", seed_level["config_id"].map(policy_label))

    mean_std = make_mean_std(df)

    seed_csv = tables_dir / "paper4_policy_seed_level.csv"
    mean_csv = tables_dir / "paper4_policy_mean_std.csv"
    mean_md = tables_dir / "paper4_policy_mean_std.md"

    seed_level.to_csv(seed_csv, index=False)
    mean_std.to_csv(mean_csv, index=False)
    write_markdown_table(mean_std, mean_md)

    print("[ok] wrote", seed_csv)
    print("[ok] wrote", mean_csv)
    print("[ok] wrote", mean_md)

    for metric in METRICS:
        make_bar_figure(mean_std, metric, figures_dir)
    make_budget_tradeoff(mean_std, figures_dir)

    make_notes(mean_std, notes_dir / "paper4_policy_interpretation_notes.md")


if __name__ == "__main__":
    main()