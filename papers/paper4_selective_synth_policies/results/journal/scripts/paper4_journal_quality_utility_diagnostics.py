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

LABELS = {
    "paper4_realonly_balanced": "Real-only",
    "paper4_baseline_b25": "Baseline b25",
    "paper4_baseline_b500": "Baseline b500",
    "paper4_baseline_b2000": "Baseline b2000",
    "paper4_policy_keep_all_b25": "Keep-all b25",
    "paper4_policy_keep_all_b500": "Keep-all b500",
    "paper4_policy_keep_all_b2000": "Keep-all b2000",
    "paper4_policy_confidence_accept_t080_b2000": "Strict-conf.",
    "paper4_policy_confidence_ranked_topk500_b2000": "Top-k500",
    "paper4_policy_confidence_ranked_topk1000_b2000": "Top-k1000",
    "paper4_policy_class_repair_topk9000_b2000": "Class-repair",
}

ORDER = list(LABELS.keys())

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["policy_label"] = df["config_id"].map(LABELS).fillna(df["config_id"])
    df["_order"] = df["config_id"].map({k: i for i, k in enumerate(ORDER)}).fillna(999).astype(int)
    sort_cols = ["_order"]
    if "seed" in df.columns:
        sort_cols.append("seed")
    return df.sort_values(sort_cols, kind="stable").drop(columns=["_order"])

def write_markdown(df: pd.DataFrame, path: Path):
    tmp = df.copy().fillna("")
    headers = list(tmp.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")

def make_scatter(df: pd.DataFrame, x: str, y: str, out_stem: str, xlabel: str, ylabel: str):
    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(plot_df[x], plot_df[y])
    for _, r in plot_df.iterrows():
        ax.annotate(r["policy_label"], (r[x], r[y]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel} vs {ylabel}")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / f"{out_stem}.png", dpi=300)
    fig.savefig(OUT_FIG_DIR / f"{out_stem}.pdf")
    plt.close(fig)

def main():
    df = pd.read_csv(IN_CSV)
    df = add_labels(df)

    policy_summary = df.groupby(["config_id", "policy_label"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean_fake=("num_fake", "mean"),
        delta_macro_f1_mean=("delta_macro_f1", "mean"),
        delta_macro_f1_std=("delta_macro_f1", "std"),
        delta_bal_acc_mean=("delta_bal_acc", "mean"),
        delta_bal_acc_std=("delta_bal_acc", "std"),
        delta_macro_auprc_mean=("delta_macro_auprc", "mean"),
        delta_macro_auprc_std=("delta_macro_auprc", "std"),
        kid_mean=("kid", "mean"),
        kid_std=("kid", "std"),
        ms_ssim_mean=("ms_ssim", "mean"),
        ms_ssim_std=("ms_ssim", "std"),
    )
    policy_summary = add_labels(policy_summary)
    out_csv = OUT_TABLE_DIR / "paper4_journal_quality_utility_policy_summary.csv"
    out_md = OUT_TABLE_DIR / "paper4_journal_quality_utility_policy_summary.md"
    policy_summary.to_csv(out_csv, index=False)
    write_markdown(policy_summary[["policy_label", "n_seeds", "mean_fake", "delta_macro_f1_mean", "delta_bal_acc_mean", "delta_macro_auprc_mean", "kid_mean", "ms_ssim_mean"]].round(6), out_md)

    corr_rows = []
    corr_df = df.dropna(subset=["kid", "ms_ssim", "delta_macro_f1", "delta_bal_acc", "delta_macro_auprc"]).copy()
    for proxy in ["kid", "ms_ssim"]:
        for metric in ["delta_macro_f1", "delta_bal_acc", "delta_macro_auprc"]:
            if len(corr_df) >= 3:
                corr_rows.append({"proxy": proxy, "metric": metric, "pearson": corr_df[proxy].corr(corr_df[metric], method="pearson"), "spearman": corr_df[proxy].corr(corr_df[metric], method="spearman"), "n": len(corr_df)})
    corr = pd.DataFrame(corr_rows)
    corr_csv = OUT_TABLE_DIR / "paper4_journal_proxy_utility_correlations.csv"
    corr_md = OUT_TABLE_DIR / "paper4_journal_proxy_utility_correlations.md"
    corr.to_csv(corr_csv, index=False)
    write_markdown(corr.round(4), corr_md)

    make_scatter(policy_summary, "kid_mean", "delta_macro_f1_mean", "paper4_journal_kid_vs_delta_macro_f1", "Mean KID", "Mean Delta Macro-F1")
    make_scatter(policy_summary, "kid_mean", "delta_macro_auprc_mean", "paper4_journal_kid_vs_delta_macro_auprc", "Mean KID", "Mean Delta Macro-AUPRC")
    make_scatter(policy_summary, "ms_ssim_mean", "delta_macro_f1_mean", "paper4_journal_msssim_vs_delta_macro_f1", "Mean MS-SSIM", "Mean Delta Macro-F1")
    make_scatter(policy_summary, "ms_ssim_mean", "delta_macro_auprc_mean", "paper4_journal_msssim_vs_delta_macro_auprc", "Mean MS-SSIM", "Mean Delta Macro-AUPRC")

    note = OUT_NOTE_DIR / "paper4_journal_quality_utility_diagnostics_notes.md"
    note.write_text(
        "# Paper 4 Journal Quality-Utility Diagnostics\n\n"
        "This diagnostic compares downstream policy utility against KID and MS-SSIM values already present in the paper-facing result table.\n\n"
        "Key interpretation to verify from the generated tables:\n\n"
        "- KID and MS-SSIM should be treated as diagnostics, not final policy-selection criteria.\n"
        "- Similar KID/MS-SSIM values can correspond to different downstream utility values.\n"
        "- Class-repair and Top-k1000 are especially important because both use 9000 selected samples but differ in allocation.\n"
        "- Calibration metrics are not included because current Paper 4 summaries do not contain usable ECE/Brier deltas.\n"
    )

    print("[ok] wrote", out_csv)
    print("[ok] wrote", out_md)
    print("[ok] wrote", corr_csv)
    print("[ok] wrote", corr_md)
    print("[ok] wrote figures to", OUT_FIG_DIR)
    print("[ok] wrote", note)

if __name__ == "__main__":
    main()