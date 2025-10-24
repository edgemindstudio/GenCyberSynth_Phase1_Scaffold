# scripts/plots/imbalance/class_counts_before_after.py
"""
Class counts before/after augmentation (per class, per model).

What
----
Side-by-side bars for REAL vs REAL+SYNTH counts per class. One figure per
model family (or filter to a single model via --model). If your summaries
include per-class counts, this will visualize the balancing policy you used.

Inputs (expected, best-effort)
------------------------------
We scan your consolidated JSONL (one run per line) for any of these per-class
shapes (all are supported; presence of *either* REAL+SYNTH *or* REAL+SYNTH=REAL+SYNTH):
- counts.per_class.real.<cls_id>
- counts.per_class.synth.<cls_id>
- counts.per_class.real_plus_synth.<cls_id>
- counts.real_per_class.<cls_id>            (alias)
- counts.synth_per_class.<cls_id>           (alias)
- counts.real_plus_synth_per_class.<cls_id> (alias)

If only REAL & SYNTH exist, we compute REAL+SYNTH = REAL + SYNTH.
If only REAL+SYNTH exists, we plot that against REAL (if REAL present).

If your JSONL lacks per-class counts entirely, the script exits gracefully
with a helpful message.

Output
------
Per-model PNG written to: artifacts/figures/imbalance/class_counts_<model>.png
(or a single file if --model is given and --outfile is set).

Examples
--------
# Plot all models (one figure each), show up to 12 largest classes by REAL count
python scripts/plots/imbalance/class_counts_before_after.py --max-classes 12

# Focus a single model and normalize bars to proportions
python scripts/plots/imbalance/class_counts_before_after.py \
  --model diffusion --normalize --outfile class_counts_diffusion.png

# Use a different JSONL path
python scripts/plots/imbalance/class_counts_before_after.py \
  --jsonl artifacts/summaries/phase1_summaries.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots._common import read_jsonl, savefig, PLOT_OUT


# -------- utilities -----------------------------------------------------------

# Accept multiple naming conventions
REAL_PREFIXES = (
    "counts.per_class.real.",
    "counts.real_per_class.",
)
SYN_PREFIXES = (
    "counts.per_class.synth.",
    "counts.synth_per_class.",
)
RS_PREFIXES = (
    "counts.per_class.real_plus_synth.",
    "counts.real_plus_synth_per_class.",
)


def _melt_prefixed(df: pd.DataFrame, prefixes: Iterable[str]) -> pd.DataFrame:
    """
    Find columns that start with any prefix and melt into rows:
    columns like 'prefix.<cls>' -> rows with ['class','value'].
    If multiple prefixes match the same class, later prefixes override earlier.
    """
    col_map: Dict[str, str] = {}
    for c in df.columns:
        for p in prefixes:
            if c.startswith(p):
                cls = c[len(p):]
                col_map[cls] = c  # last one wins if duplicates
    if not col_map:
        return pd.DataFrame(columns=["class", "value"])
    # Wide -> long (but only 2 columns of interest)
    parts = []
    for cls, col in sorted(col_map.items(), key=lambda kv: kv[0]):
        vals = pd.to_numeric(df[col], errors="coerce")
        parts.append(pd.DataFrame({"class": cls, "value": vals}))
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def _aggregate_counts(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Return per-model aggregated counts with columns:
    ['class', 'real', 'real_plus_synth'].
    Aggregation is mean over runs (you can change to sum if desired).
    """
    if "model" not in df.columns:
        # If no 'model' column, treat all as one bucket
        df = df.assign(model="(all)")

    results: Dict[str, pd.DataFrame] = {}

    for model, g in df.groupby("model", dropna=False):
        real = _melt_prefixed(g, REAL_PREFIXES)   # class, value
        synth = _melt_prefixed(g, SYN_PREFIXES)   # class, value
        rs = _melt_prefixed(g, RS_PREFIXES)       # class, value

        if real.empty and rs.empty:
            # No usable per-class info
            continue

        # Pivot to mean over runs
        def _mean_by_class(x: pd.DataFrame) -> pd.Series:
            if x.empty:
                return pd.Series(dtype=float)
            return x.groupby("class", as_index=True)["value"].mean()

        real_mean = _mean_by_class(real).rename("real")
        rs_mean = _mean_by_class(rs).rename("real_plus_synth")
        synth_mean = _mean_by_class(synth).rename("synth")

        # If rs missing but we have real + synth, create it
        if rs_mean.empty and (not real_mean.empty or not synth_mean.empty):
            combined = pd.concat([real_mean, synth_mean], axis=1)
            combined = combined.fillna(0.0)
            rs_mean = (combined["real"].fillna(0) + combined["synth"].fillna(0)).rename("real_plus_synth")

        # Join available series
        agg = pd.concat([real_mean, rs_mean], axis=1)
        # If still missing rs or real, keep what's there and drop full-NaN rows
        agg = agg.dropna(how="all")
        if agg.empty:
            continue

        # Classes as str and explicit column
        agg = agg.reset_index().rename(columns={"index": "class"})
        results[str(model)] = agg

    return results


def _limit_classes(agg: pd.DataFrame, max_classes: int, by_col: str = "real") -> pd.DataFrame:
    """Keep up to max_classes sorted by descending column (default: 'real')."""
    if max_classes <= 0 or agg.empty:
        return agg
    if by_col not in agg.columns:
        by_col = agg.columns[1] if len(agg.columns) > 1 else "class"
    return agg.sort_values(by_col, ascending=False).head(max_classes)


def _normalize_rows(agg: pd.DataFrame) -> pd.DataFrame:
    """Normalize counts to proportions per class: divide by REAL+SYNTH if present else by row max."""
    out = agg.copy()
    denom = None
    if "real_plus_synth" in out.columns:
        denom = out["real_plus_synth"].replace(0, np.nan)
    if denom is None or denom.isna().all():
        # fallback: max across available numeric columns in the row
        num_cols = [c for c in out.columns if c not in ("class",)]
        denom = out[num_cols].max(axis=1).replace(0, np.nan)
    for c in ("real", "real_plus_synth"):
        if c in out.columns:
            out[c] = out[c] / denom
    return out


# -------- plotting ------------------------------------------------------------

def _plot_counts_bars(
    agg: pd.DataFrame,
    model: str,
    normalize: bool,
    rotate: int,
    outfile: Optional[str],
) -> None:
    if agg.empty:
        print(f"[skip] no per-class counts for model={model}")
        return

    data = agg.copy()
    if normalize:
        data = _normalize_rows(data)

    # Ensure both series exist; fill missing with zeros for plotting
    if "real" not in data.columns:
        data["real"] = 0.0
    if "real_plus_synth" not in data.columns:
        data["real_plus_synth"] = data["real"]  # fallback: identical

    x = np.arange(len(data))
    w = 0.44

    plt.figure(figsize=(max(8, 0.5 * len(data)), 5.0))
    plt.bar(x - w / 2, data["real"].values, width=w, label="Real")
    plt.bar(x + w / 2, data["real_plus_synth"].values, width=w, label="Real + Synth")

    plt.xticks(x, data["class"].astype(str).values, rotation=rotate, ha="right")
    plt.ylabel("Proportion" if normalize else "Count")
    plt.title(f"Class counts before/after (model: {model})")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    outdir = PLOT_OUT / "imbalance"
    fname = outfile or f"class_counts_{model}.png"
    savefig(outdir / fname)


# -------- CLI ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Class counts before/after augmentation.")
    ap.add_argument("--jsonl", default="artifacts/summaries/phase1_summaries.jsonl",
                    help="Path to consolidated JSONL.")
    ap.add_argument("--model", default=None,
                    help="Filter to a single model family (name as appears in JSONL).")
    ap.add_argument("--max-classes", type=int, default=20,
                    help="Show up to this many classes (sorted by REAL count). Set 0 for all.")
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize bars to per-class proportions instead of raw counts.")
    ap.add_argument("--rotate", type=int, default=25, help="X tick rotation.")
    ap.add_argument("--outfile", default=None,
                    help="Output filename (only used when --model is set).")
    args = ap.parse_args()

    df = read_jsonl(args.jsonl)
    if df.empty:
        return

    by_model = _aggregate_counts(df)
    if not by_model:
        print("[skip] No per-class count fields found in JSONL. "
              "Populate counts.per_class.* in your summaries to enable this figure.")
        return

    if args.model:
        model = args.model
        if model not in by_model:
            print(f"[skip] model '{model}' not found among: {sorted(by_model.keys())}")
            return
        agg = _limit_classes(by_model[model], args.max_classes, by_col="real")
        _plot_counts_bars(agg, model=model, normalize=args.normalize,
                          rotate=args.rotate, outfile=args.outfile)
        return

    # Otherwise: one plot per model
    for model, agg0 in sorted(by_model.items()):
        agg = _limit_classes(agg0, args.max_classes, by_col="real")
        _plot_counts_bars(agg, model=model, normalize=args.normalize,
                          rotate=args.rotate, outfile=None)


if __name__ == "__main__":
    main()
