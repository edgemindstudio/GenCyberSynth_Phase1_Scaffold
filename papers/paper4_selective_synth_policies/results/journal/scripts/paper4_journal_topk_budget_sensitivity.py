#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("papers/paper4_selective_synth_policies")
IN_CSV = ROOT / "results/raw/paper4_results_paper.csv"
OUT_TABLE_DIR = ROOT / "results/journal/tables"
OUT_FIG_DIR = ROOT / "results/journal/figures"
OUT_NOTE_DIR = ROOT / "results/journal/notes"

OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_NOTE_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "paper4_policy_confidence_ranked_topk500_b2000": "Top-k500",
    "paper4_policy_confidence_ranked_topk1000_b2000": "Top-k1000",
    "paper4_policy_class_repair_topk9000_b2000": "Class-repair",
}

ORDER = ["Top-k500", "Top-k1000", "Class-repair"]

def write_markdown(df: pd.DataFrame, path: Path):
    tmp = df.copy().fillna("")
    headers = list(tmp.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")

def main():
    df = pd.read_csv(IN_CSV)
    df = df[df["config_id"].isin(TARGETS)].copy()
    df["policy_label"] = df["config_id"].map(TARGETS)
    df["selected_budget_total"] = df["num_fake"]
    df["_order"] = df["policy_label"].map({k: i for i, k in enumerate(ORDER)})
    df = df.sort_values(["_order", "seed"], kind="stable").drop(columns=["_order"])

    seed_csv = OUT_TABLE_DIR / "paper4_journal_topk_budget_sensitivity_seed_level.csv"
    seed_md = OUT_TABLE_DIR / "paper4_journal_topk_budget_sensitivity_seed_level.md"
    seed_cols = ["policy_label", "seed", "selected_budget_total", "delta_macro_f1", "delta_bal_acc", "delta_macro_auprc", "kid", "ms_ssim"]
    df[seed_cols].to_csv(seed_csv, index=False)
    write_markdown(df[seed_cols].round(6), seed_md)

    summary = df.groupby("policy_label", as_index=False).agg(n_seeds=("seed", "nunique"), selected_budget_total_mean=("selected_budget_total", "mean"), delta_macro_f1_mean=("delta_macro_f1", "mean"), delta_macro_f1_std=("delta_macro_f1", "std"), delta_bal_acc_mean=("delta_bal_acc", "mean"), delta_bal_acc_std=("delta_bal_acc", "std"), delta_macro_auprc_mean=("delta_macro_auprc", "mean"), delta_macro_auprc_std=("delta_macro_auprc", "std"), kid_mean=("kid", "mean"), ms_ssim_mean=("ms_ssim", "mean"))
    summary["_order"] = summary["policy_label"].map({k: i for i, k in enumerate(ORDER)})
    summary = summary.sort_values("_order", kind="stable").drop(columns=["_order"])

    summary_csv = OUT_TABLE_DIR / "paper4_journal_topk_budget_sensitivity_summary.csv"
    summary_md = OUT_TABLE_DIR / "paper4_journal_topk_budget_sensitivity_summary.md"
    summary.to_csv(summary_csv, index=False)
    write_markdown(summary.round(6), summary_md)

    metrics = [("delta_macro_f1_mean", "Mean Delta Macro-F1"), ("delta_bal_acc_mean", "Mean Delta Balanced Accuracy"), ("delta_macro_auprc_mean", "Mean Delta Macro-AUPRC")]
    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.bar(summary["policy_label"], summary[metric])
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Reduced-budget policy")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " under reduced-budget policies")
        fig.tight_layout()
        out_stem = metric.replace("_mean", "")
        fig.savefig(OUT_FIG_DIR / f"paper4_journal_topk_budget_sensitivity_{out_stem}.png", dpi=300)
        fig.savefig(OUT_FIG_DIR / f"paper4_journal_topk_budget_sensitivity_{out_stem}.pdf")
        plt.close(fig)

    note = OUT_NOTE_DIR / "paper4_journal_topk_budget_sensitivity_notes.md"
    note.write_text(
        "# Paper 4 Top-k Budget Sensitivity Diagnostics\n\n"
        "This diagnostic compares Top-k500, Top-k1000, and Class-repair.\n\n"
        "Main interpretation:\n\n"
        "- Top-k500 is the most restrictive balanced confidence-ranked policy and is near-zero or negative on Macro-F1 and balanced accuracy.\n"
        "- Top-k1000 keeps twice as many samples per class as Top-k500 and restores positive Macro-F1 and balanced-accuracy utility.\n"
        "- Class-repair uses the same total 9000-sample budget as Top-k1000, but reallocates across classes and improves Macro-AUPRC.\n"
        "- This supports the journal claim that reduced-budget policy utility depends on both retained sample count and allocation structure.\n"
    )

    print("[ok] wrote", seed_csv)
    print("[ok] wrote", seed_md)
    print("[ok] wrote", summary_csv)
    print("[ok] wrote", summary_md)
    print("[ok] wrote figures to", OUT_FIG_DIR)
    print("[ok] wrote", note)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()