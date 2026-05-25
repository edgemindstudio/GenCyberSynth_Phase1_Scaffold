#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

REPO_ROOT = Path("/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold")
PARTS_DIR = REPO_ROOT / "papers/paper3_when_does_synth_help/results/raw/resnet_sensitivity_parts"

OUT_RAW = REPO_ROOT / "papers/paper3_when_does_synth_help/results/raw/paper3_resnet_sensitivity_raw.csv"
OUT_AGG = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_aggregate_frozen_20260524.csv"
OUT_MD = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_table.md"
OUT_TEX = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_table.tex"
OUT_SUMMARY = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_summary_20260524.json"

def pm(m, s):
    if pd.isna(m):
        return ""
    if abs(m) < 0.00005:
        m = 0.0
    if pd.isna(s):
        s = 0.0
    return f"{m:.4f} ± {s:.4f}"

def write_md_table(table: pd.DataFrame, path: Path):
    headers = list(table.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")

paths = sorted(PARTS_DIR.glob("resnet_*.csv"))
if not paths:
    raise RuntimeError(f"No part files found in {PARTS_DIR}")

dfs = []
for p in paths:
    df = pd.read_csv(p)
    df["part_path"] = str(p)
    dfs.append(df)

raw = pd.concat(dfs, ignore_index=True)
raw = raw.drop_duplicates(subset=["classifier", "regime", "family", "budget_per_class", "seed"], keep="last")
raw = raw.sort_values(["regime", "family", "budget_per_class", "seed"])

OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
OUT_AGG.parent.mkdir(parents=True, exist_ok=True)
raw.to_csv(OUT_RAW, index=False)

expected = 18
if len(raw) != expected:
    print("[warn] expected", expected, "rows but found", len(raw))
    print(raw.groupby(["regime", "family", "budget_per_class"])["seed"].count())
    raise RuntimeError(f"Incomplete ResNet sensitivity results: {len(raw)} rows")

agg = (
    raw.groupby(["classifier", "regime", "family", "budget_per_class"])
    .agg(
        runs=("seed", "count"),
        delta_macro_f1_mean=("delta_macro_f1", "mean"),
        delta_macro_f1_std=("delta_macro_f1", "std"),
        delta_balanced_accuracy_mean=("delta_balanced_accuracy", "mean"),
        delta_balanced_accuracy_std=("delta_balanced_accuracy", "std"),
        delta_macro_auprc_mean=("delta_macro_auprc", "mean"),
        delta_macro_auprc_std=("delta_macro_auprc", "std"),
        delta_class4_f1_mean=("delta_class4_f1", "mean"),
        delta_class4_f1_std=("delta_class4_f1", "std"),
        delta_class7_f1_mean=("delta_class7_f1", "mean"),
        delta_class7_f1_std=("delta_class7_f1", "std"),
    )
    .reset_index()
    .sort_values(["regime", "budget_per_class", "family"])
)
agg.to_csv(OUT_AGG, index=False)

table = pd.DataFrame({
    "Classifier": agg["classifier"],
    "Regime": agg["regime"],
    "Family": agg["family"],
    "Budget/Class": agg["budget_per_class"].astype(int),
    "Runs": agg["runs"].astype(int),
    "Δ Macro-F1": [pm(m, s) for m, s in zip(agg["delta_macro_f1_mean"], agg["delta_macro_f1_std"])],
    "Δ Bal. Acc.": [pm(m, s) for m, s in zip(agg["delta_balanced_accuracy_mean"], agg["delta_balanced_accuracy_std"])],
    "Δ Macro-AUPRC": [pm(m, s) for m, s in zip(agg["delta_macro_auprc_mean"], agg["delta_macro_auprc_std"])],
    "Δ Class-4 F1": [pm(m, s) for m, s in zip(agg["delta_class4_f1_mean"], agg["delta_class4_f1_std"])],
    "Δ Class-7 F1": [pm(m, s) for m, s in zip(agg["delta_class7_f1_mean"], agg["delta_class7_f1_std"])],
})
write_md_table(table, OUT_MD)
OUT_TEX.write_text(table.to_latex(index=False, escape=False))

summary = {
    "analysis": "paper3_classifier_sensitivity_resnet_collected",
    "expected_rows": expected,
    "actual_rows": int(len(raw)),
    "part_files": [str(p) for p in paths],
    "outputs": {
        "raw_csv": str(OUT_RAW),
        "aggregate_csv": str(OUT_AGG),
        "table_md": str(OUT_MD),
        "table_tex": str(OUT_TEX),
    },
}
OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

print("[ok] wrote", OUT_RAW)
print("[ok] wrote", OUT_AGG)
print("[ok] wrote", OUT_MD)
print("[ok] wrote", OUT_TEX)
print("[ok] wrote", OUT_SUMMARY)
print()
print(table.to_string(index=False))