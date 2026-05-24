from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = Path("papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_perclass_c4c7_aggregate_frozen_20260524.csv")

OUT_DIR = Path("papers/paper3_when_does_synth_help/results/frozen")
FIG_DIR = OUT_DIR / "figures_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "paper3_minority_heavy_c4c7_precision_recall_tradeoff_frozen_20260524.csv"
OUT_MD = OUT_DIR / "paper3_minority_heavy_c4c7_precision_recall_tradeoff_table.md"
OUT_TEX = OUT_DIR / "paper3_minority_heavy_c4c7_precision_recall_tradeoff_table.tex"

OUT_PNG = FIG_DIR / "paper3_fig5_c4c7_precision_recall_tradeoff.png"
OUT_PDF = FIG_DIR / "paper3_fig5_c4c7_precision_recall_tradeoff.pdf"


def classify_tradeoff(dp, dr, df1, eps=0.002):
    if abs(df1) <= eps and abs(dp) <= eps and abs(dr) <= eps:
        return "near neutral"
    if dp > eps and dr > eps and df1 > eps:
        return "both improve"
    if dp < -eps and dr < -eps:
        return "both decline"
    if dp > eps and dr < -eps:
        return "precision gain / recall loss"
    if dp < -eps and dr > eps:
        return "recall gain / precision loss"
    if df1 < -eps:
        return "net F1 decline"
    if df1 > eps:
        return "net F1 improvement"
    return "mixed / near neutral"


def pm(m, sd):
    if pd.isna(m):
        return ""
    if abs(m) < 0.00005:
        m = 0.0
    if pd.isna(sd):
        sd = 0.0
    return f"{m:.4f} ± {sd:.4f}"


df = pd.read_csv(IN_CSV)

df["family"] = df["family"].astype(str).str.upper()
df["budget_per_class"] = df["budget_per_class"].astype(int)
df["class_id"] = df["class_id"].astype(int)

for col in [
    "delta_precision_mean",
    "delta_precision_std",
    "delta_recall_mean",
    "delta_recall_std",
    "delta_f1_mean",
    "delta_f1_std",
]:
    df[col] = df[col].astype(float)

df["tradeoff_pattern"] = [
    classify_tradeoff(dp, dr, df1)
    for dp, dr, df1 in zip(
        df["delta_precision_mean"],
        df["delta_recall_mean"],
        df["delta_f1_mean"],
    )
]

df = df.sort_values(["budget_per_class", "family", "class_id"])
df.to_csv(OUT_CSV, index=False)

table = pd.DataFrame({
    "Family": df["family"],
    "Budget/Class": df["budget_per_class"].astype(int),
    "Class": df["class_id"].astype(int),
    "Runs": df["runs"].astype(int),
    "Δ Precision": [pm(m, s) for m, s in zip(df["delta_precision_mean"], df["delta_precision_std"])],
    "Δ Recall": [pm(m, s) for m, s in zip(df["delta_recall_mean"], df["delta_recall_std"])],
    "Δ F1": [pm(m, s) for m, s in zip(df["delta_f1_mean"], df["delta_f1_std"])],
    "Pattern": df["tradeoff_pattern"],
})

headers = list(table.columns)
rows_md = []
rows_md.append("| " + " | ".join(headers) + " |")
rows_md.append("| " + " | ".join(["---"] * len(headers)) + " |")
for row in table.astype(str).values.tolist():
    rows_md.append("| " + " | ".join(row) + " |")

OUT_MD.write_text("\n".join(rows_md) + "\n")
OUT_TEX.write_text(table.to_latex(index=False, escape=False))

# Figure 5: ΔRecall vs ΔPrecision.
family_order = ["GAN", "VAE", "DIFFUSION"]
budget_order = [100, 500, 2000]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

for ax, cls in zip(axes, [4, 7]):
    sub = df[df["class_id"] == cls].copy()

    for fam in family_order:
        fam_df = sub[sub["family"] == fam].copy()
        fam_df = fam_df.set_index("budget_per_class").reindex(budget_order).reset_index()

        ax.plot(
            fam_df["delta_recall_mean"],
            fam_df["delta_precision_mean"],
            marker="o",
            linewidth=2,
            label=fam,
        )

        for _, row in fam_df.iterrows():
            if pd.notna(row["budget_per_class"]):
                ax.annotate(
                    str(int(row["budget_per_class"])),
                    (row["delta_recall_mean"], row["delta_precision_mean"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.axvline(0.0, linewidth=1, linestyle="--")
    ax.set_title(f"Minority Class {cls}")
    ax.set_xlabel("Δ Recall")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Δ Precision")
axes[1].legend(title="Family", loc="best")

fig.suptitle("Precision–Recall Tradeoffs Under c4/c7 Minority-Only Augmentation", y=1.03)
fig.tight_layout()

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print("[ok] wrote", OUT_CSV)
print("[ok] wrote", OUT_MD)
print("[ok] wrote", OUT_TEX)
print("[ok] wrote", OUT_PNG)
print("[ok] wrote", OUT_PDF)
print()
print(table.to_string(index=False))
print()
print("Pattern counts:")
print(df["tradeoff_pattern"].value_counts().to_string())