from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = Path("papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_perclass_c4c7_aggregate_frozen_20260524.csv")
OUT_DIR = Path("papers/paper3_when_does_synth_help/results/frozen/figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "paper3_fig4_minority_c4c7_delta_f1_by_family_budget.png"
OUT_PDF = OUT_DIR / "paper3_fig4_minority_c4c7_delta_f1_by_family_budget.pdf"

df = pd.read_csv(IN_CSV)

df["family"] = df["family"].astype(str).str.upper()
df["budget_per_class"] = df["budget_per_class"].astype(int)
df["class_id"] = df["class_id"].astype(int)
df["delta_f1_mean"] = df["delta_f1_mean"].astype(float)
df["delta_f1_std"] = df["delta_f1_std"].astype(float)

family_order = ["GAN", "VAE", "DIFFUSION"]
budget_order = [100, 500, 2000]
class_order = [4, 7]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, cls in zip(axes, class_order):
    sub = df[df["class_id"] == cls].copy()

    for fam in family_order:
        fam_df = sub[sub["family"] == fam].copy()
        fam_df = fam_df.set_index("budget_per_class").reindex(budget_order).reset_index()

        ax.errorbar(
            fam_df["budget_per_class"],
            fam_df["delta_f1_mean"],
            yerr=fam_df["delta_f1_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=fam,
        )

    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_title(f"Minority Class {cls}")
    ax.set_xlabel("Synthetic Budget per Minority Class")
    ax.set_xticks(budget_order)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Δ F1 (Real+Synthetic − Real-only)")
axes[1].legend(title="Family", loc="best")

fig.suptitle("Minority-Heavy c4/c7 Regime: Minority-Class ΔF1 Across Family and Budget", y=1.03)
fig.tight_layout()

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print("[ok] wrote", OUT_PNG)
print("[ok] wrote", OUT_PDF)