# scripts/plots/core/calibration_curves.py

#!/usr/bin/env python
"""
Calibration overview (ECE) + optional reliability diagram.

- Reads consolidated JSONL (phase1_summaries.jsonl)
- Picks ECE across schemas:
    utility_real_plus_synth.ece  (preferred)
    utility_real_only.ece
    metrics.ece                  (legacy fallback)
- Always writes a per-model ECE bar plot.
- If per-bin data is present for any run, also draws a reliability diagram for the
  best (lowest ECE) run using these optional fields (either “*_plus_synth” or “*_real_only”):
    <block>_calibration.bin_conf : list[float]  (bin mean confidence)
    <block>_calibration.bin_acc  : list[float]  (bin empirical accuracy)

Outputs:
  artifacts/figures/core/calibration_ece_bars.png
  artifacts/figures/core/calibration_reliability.png   (only if bins available)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots._common import (
    read_jsonl,
    savefig,
    new_figure,
    model_palette,
    PLOT_OUT_DIR,
)

# ----- column candidates ------------------------------------------------------

_ECE_CANDIDATES = [
    "utility_real_plus_synth.ece",   # new, preferred
    "utility_real_only.ece",         # new
    "metrics.ece",                   # legacy catch-all
]

# optional per-bin calibration arrays (either RS or R)
_BIN_CONF_CANDS = [
    "utility_real_plus_synth_calibration.bin_conf",
    "utility_real_only_calibration.bin_conf",
]
_BIN_ACC_CANDS = [
    "utility_real_plus_synth_calibration.bin_acc",
    "utility_real_only_calibration.bin_acc",
]

# ----- helpers ----------------------------------------------------------------

def _pick_first_present(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns and df[c].notna().any():
            return c
    for c in cands:
        if c in df.columns:
            return c
    return None

def _find_bin_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    bc = _pick_first_present(df, _BIN_CONF_CANDS)
    ba = _pick_first_present(df, _BIN_ACC_CANDS)
    if bc and ba:
        return bc, ba
    return None, None

# ----- plots ------------------------------------------------------------------

def plot_ece_bars(df: pd.DataFrame, ece_col: str, *, out: Path) -> None:
    data = df[["model", ece_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[ece_col])
    if data.empty:
        print("[skip] no non-NaN ECE values; skipping ECE bar plot.")
        return

    # Aggregate: best ECE per model (lower is better)
    best = data.groupby("model", as_index=False)[ece_col].min().sort_values(ece_col)
    colors = [model_palette([m])[m] for m in best["model"].astype(str)]

    fig, ax = new_figure(figsize=(8, 4.5))
    ax.barh(best["model"], best[ece_col], color=colors, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("ECE (↓ better)")
    ax.set_title("Expected Calibration Error (best per model)")

    # Annotate bars
    for y, v in enumerate(best[ece_col].values):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=9)

    savefig(out, tight=True, dpi=220)

def plot_reliability(df: pd.DataFrame, ece_col: str, bin_conf_col: str, bin_acc_col: str, *, out: Path) -> None:
    # Choose the single best run (lowest ECE) that has bins available
    sub = df[["model", ece_col, bin_conf_col, bin_acc_col]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[ece_col, bin_conf_col, bin_acc_col])
    if sub.empty:
        print("[info] no runs with per-bin calibration arrays; skipping reliability diagram.")
        return

    # Keep rows where arrays are list-like with same length
    def _valid(row):
        bc, ba = row[bin_conf_col], row[bin_acc_col]
        return isinstance(bc, (list, tuple)) and isinstance(ba, (list, tuple)) and len(bc) == len(ba) and len(bc) > 0
    sub = sub[sub.apply(_valid, axis=1)]
    if sub.empty:
        print("[info] per-bin arrays exist but malformed; skipping reliability diagram.")
        return

    best = sub.sort_values(ece_col, ascending=True).iloc[0]
    bins_conf = np.asarray(best[bin_conf_col], dtype=float)
    bins_acc  = np.asarray(best[bin_acc_col], dtype=float)
    model_name = str(best["model"])
    ece_val = float(best[ece_col])

    fig, ax = new_figure(figsize=(5.2, 5.2))
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#666666", label="Perfect")

    # Step curve (bin means)
    # Draw centers if the inputs are bin centers; fine for summary view.
    ax.plot(bins_conf, bins_acc, marker="o", linewidth=2.0, alpha=0.9, label=f"{model_name} (ECE={ece_val:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram (best ECE run)")
    ax.legend(loc="lower right")

    savefig(out, tight=True, dpi=220)

# ----- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration overview: ECE bars + reliability diagram (optional)")
    ap.add_argument("--jsonl", default=str(PLOT_OUT_DIR.parent / "summaries" / "phase1_summaries.jsonl"),
                    help="Path to consolidated JSONL")
    ap.add_argument("--out-bars", default=str(PLOT_OUT_DIR / "core" / "calibration_ece_bars.png"),
                    help="Output PNG for ECE bars")
    ap.add_argument("--out-reliability", default=str(PLOT_OUT_DIR / "core" / "calibration_reliability.png"),
                    help="Output PNG for reliability diagram (if bin arrays present)")
    args = ap.parse_args()

    df = read_jsonl(args.jsonl)
    if df.empty:
        print("[skip] empty DataFrame; nothing to plot")
        return

    # Ensure model column
    if "model" not in df.columns:
        df["model"] = "NA"

    ece_col = _pick_first_present(df, _ECE_CANDIDATES)
    if ece_col is None:
        print("[skip] no ECE columns found; candidates:",
              ", ".join(_ECE_CANDIDATES))
        return

    print(f"[info] Using ECE column: {ece_col}")
    plot_ece_bars(df, ece_col, out=Path(args.out_bars))

    bc, ba = _find_bin_cols(df)
    if bc and ba:
        print(f"[info] Found per-bin columns: {bc}, {ba}")
        plot_reliability(df, ece_col, bc, ba, out=Path(args.out_reliability))
    else:
        print("[info] No per-bin calibration arrays detected; skipping reliability diagram.")

if __name__ == "__main__":
    main()
