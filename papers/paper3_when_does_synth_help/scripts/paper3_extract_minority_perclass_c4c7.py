#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd


ARTS = Path("/home/bruno.fonkeng/gencys/artifacts_paper3")
OUT_RAW = Path("papers/paper3_when_does_synth_help/results/raw/paper3_minority_heavy_perclass_c4c7_raw.csv")
OUT_AGG = Path("papers/paper3_when_does_synth_help/results/raw/paper3_minority_heavy_perclass_c4c7_aggregate.csv")
OUT_FROZEN = Path("papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_perclass_c4c7_aggregate_frozen_20260524.csv")
OUT_MD = Path("papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_perclass_c4c7_aggregate_table.md")
OUT_TEX = Path("papers/paper3_when_does_synth_help/results/frozen/paper3_minority_heavy_perclass_c4c7_aggregate_table.tex")


def infer_budget(s, cid):
    rm = s.get("run_meta") or {}
    regime = rm.get("regime") or {}
    for v in [regime.get("synth_budget_per_class"), s.get("budget_per_class"), rm.get("budget_per_class")]:
        try:
            if v is not None:
                return int(v)
        except Exception:
            pass
    m = re.search(r"_b(\d+)", str(cid))
    if m:
        return int(m.group(1))
    return None


def metric(pc, name, class_id):
    vals = pc.get(name)
    if not isinstance(vals, list):
        return None
    if class_id >= len(vals):
        return None
    v = vals[class_id]
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def support_metric(pc, class_id):
    vals = pc.get("support")
    if not isinstance(vals, list):
        return None
    if class_id >= len(vals):
        return None
    v = vals[class_id]
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def pm(m, sd):
    if pd.isna(m):
        return ""
    if abs(m) < 0.00005:
        m = 0.0
    if pd.isna(sd):
        sd = 0.0
    return f"{m:.4f} ± {sd:.4f}"


rows = []
skipped_incomplete = 0

for fam in ["gan", "vae", "diffusion"]:
    summary_dir = ARTS / fam / "summaries"
    for p in sorted(summary_dir.glob("summary_*.json"), key=lambda x: x.stat().st_mtime):
        try:
            s = json.load(open(p))
        except Exception:
            continue

        rm = s.get("run_meta") or {}
        cid = s.get("config_id") or rm.get("config_id") or ""

        if "minority_heavy_c4c7" not in str(cid):
            continue

        seed = rm.get("seed", s.get("seed"))
        try:
            seed = int(seed)
        except Exception:
            continue

        budget = infer_budget(s, cid)
        if budget not in {100, 500, 2000}:
            continue

        ro = s.get("utility_real_only") or {}
        rs = s.get("utility_real_plus_synth") or {}
        ro_pc = ro.get("per_class") or {}
        rs_pc = rs.get("per_class") or {}

        if not isinstance(ro_pc, dict) or not isinstance(rs_pc, dict):
            continue

        for c in [4, 7]:
            ro_precision = metric(ro_pc, "precision", c)
            rs_precision = metric(rs_pc, "precision", c)
            ro_recall = metric(ro_pc, "recall", c)
            rs_recall = metric(rs_pc, "recall", c)
            ro_f1 = metric(ro_pc, "f1", c)
            rs_f1 = metric(rs_pc, "f1", c)
            support = support_metric(ro_pc, c)

            needed = [ro_precision, rs_precision, ro_recall, rs_recall, ro_f1, rs_f1]
            if any(v is None for v in needed):
                skipped_incomplete += 1
                continue

            rows.append({
                "summary_path": str(p),
                "summary_mtime": p.stat().st_mtime,
                "family": fam.upper(),
                "config_id": cid,
                "seed": seed,
                "budget_per_class": budget,
                "class_id": c,
                "real_only_precision": ro_precision,
                "real_plus_synth_precision": rs_precision,
                "delta_precision": rs_precision - ro_precision,
                "real_only_recall": ro_recall,
                "real_plus_synth_recall": rs_recall,
                "delta_recall": rs_recall - ro_recall,
                "real_only_f1": ro_f1,
                "real_plus_synth_f1": rs_f1,
                "delta_f1": rs_f1 - ro_f1,
                "support": support,
            })

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError(f"No valid per-class rows extracted. skipped_incomplete={skipped_incomplete}")

df = df.sort_values("summary_mtime")
df = df.drop_duplicates(subset=["family", "config_id", "seed", "budget_per_class", "class_id"], keep="last")

OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
OUT_FROZEN.parent.mkdir(parents=True, exist_ok=True)

df = df.sort_values(["family", "budget_per_class", "seed", "class_id"])
df.to_csv(OUT_RAW, index=False)

agg = (
    df.groupby(["family", "budget_per_class", "class_id"])
    .agg(
        runs=("seed", "count"),
        delta_precision_mean=("delta_precision", "mean"),
        delta_precision_std=("delta_precision", "std"),
        delta_recall_mean=("delta_recall", "mean"),
        delta_recall_std=("delta_recall", "std"),
        delta_f1_mean=("delta_f1", "mean"),
        delta_f1_std=("delta_f1", "std"),
        real_only_f1_mean=("real_only_f1", "mean"),
        real_plus_synth_f1_mean=("real_plus_synth_f1", "mean"),
        support_mean=("support", "mean"),
    )
    .reset_index()
    .sort_values(["budget_per_class", "family", "class_id"])
)

agg.to_csv(OUT_AGG, index=False)
agg.to_csv(OUT_FROZEN, index=False)

table = pd.DataFrame({
    "Family": agg["family"],
    "Budget/Class": agg["budget_per_class"].astype(int),
    "Class": agg["class_id"].astype(int),
    "Runs": agg["runs"].astype(int),
    "Δ Precision": [pm(m, s) for m, s in zip(agg["delta_precision_mean"], agg["delta_precision_std"])],
    "Δ Recall": [pm(m, s) for m, s in zip(agg["delta_recall_mean"], agg["delta_recall_std"])],
    "Δ F1": [pm(m, s) for m, s in zip(agg["delta_f1_mean"], agg["delta_f1_std"])],
})

# Write Markdown manually to avoid requiring optional pandas dependency: tabulate.
headers = list(table.columns)
rows_md = []
rows_md.append("| " + " | ".join(headers) + " |")
rows_md.append("| " + " | ".join(["---"] * len(headers)) + " |")
for row in table.astype(str).values.tolist():
    rows_md.append("| " + " | ".join(row) + " |")
OUT_MD.write_text("\n".join(rows_md) + "\n")
OUT_TEX.write_text(table.to_latex(index=False, escape=False))

print("[ok] wrote", OUT_RAW)
print("[ok] wrote", OUT_AGG)
print("[ok] wrote", OUT_FROZEN)
print("[ok] wrote", OUT_MD)
print("[ok] wrote", OUT_TEX)
print()
print("raw rows:", len(df))
print("skipped incomplete rows:", skipped_incomplete)
print()
print(df.groupby(["family", "budget_per_class", "class_id"])["seed"].count())
print()
print(table.to_string(index=False))
