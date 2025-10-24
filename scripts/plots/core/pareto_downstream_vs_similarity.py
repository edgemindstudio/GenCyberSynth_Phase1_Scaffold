# scripts/plots/core/pareto_downstream_vs_similarity.py

#!/usr/bin/env python
"""
Pareto: Downstream (Macro-F1) vs Similarity ((c)FID)

- Auto-detects columns across legacy (metrics.*) and new (generative.*, utility_*.*) schemas.
- Prefers cFID if available; falls back to FID/KID; override with --xcol.
- Prefers Real+Synth Macro-F1; override with --ycol.
- Color by model, optional marker by seed, dashed Pareto frontier.

Input JSONL:
  artifacts/summaries/phase1_summaries.jsonl

Output:
  artifacts/figures/core/pareto_downstream_vs_similarity.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots._common import (
    read_jsonl,
    savefig,
    new_figure,
    model_palette,
    seed_markers,
    pareto_frontier,
    PLOT_OUT_DIR,
)

# ---- candidate columns ------------------------------------------------------

_SIMILARITY_CANDIDATES = [
    # Prefer cFID first, then FID; finally allow KID as last resort.
    "metrics.cfid",                 # legacy shim
    "metrics.cfid_macro",           # legacy
    "generative.cfid_macro",        # new
    "metrics.fid_macro",            # legacy
    "generative.fid_macro",         # new
    "metrics.kid",                  # fallbacks
    "generative.kid",
    "generative.ms_ssim",
    "metrics.ms_ssim",
]

_USEFULNESS_CANDIDATES = [
    # Real+Synth Macro-F1 preferred; accept legacy or new names.
    "metrics.downstream.macro_f1",          # legacy shim
    "metrics.real_plus_synth.macro_f1",     # legacy
    "utility_real_plus_synth.macro_f1",     # new
    # Generic fallbacks if you only stored a single macro_f1
    "metrics.macro_f1",
    "utility.macro_f1",
    "counts.num_fake",
]

# ---- helpers ----------------------------------------------------------------

def _pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # Prefer the first column that exists and has at least one non-NaN
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c
    # If none have values, return the first that merely exists (may be all-NaN)
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _pretty_label(col: str, *, is_x=False) -> str:
    cl = col.lower()
    if "cfid" in cl: return "cFID"
    if "fid"  in cl: return "FID"
    if "kid"  in cl: return "KID"
    if "ms_ssim" in cl or "ms-ssim" in cl:
        return "MS-SSIM (lower ≈ more diverse)" if is_x else "MS-SSIM"
    if col == "counts.num_fake":
        return "# Synthetic Samples"
    if "macro_f1" in cl:
        return "Macro-F1 (Real+Synth)"
    return col


# ---- main plotting ----------------------------------------------------------

def plot_pareto(
    df: pd.DataFrame,
    *,
    xcol: Optional[str] = None,
    ycol: Optional[str] = None,
    prefer_cfid: bool = True,  # retained for CLI parity; auto-pick already prefers cFID
    logx: bool = False,
    annotate_top: int = 0,
    outfile: Path = PLOT_OUT_DIR / "core" / "pareto_downstream_vs_similarity.png",
    transparent: bool = False,
) -> None:
    # Resolve columns (unless user overrides)
    if xcol is None:
        xcol = _pick_first_present(df, _SIMILARITY_CANDIDATES)
    if ycol is None:
        ycol = _pick_first_present(df, _USEFULNESS_CANDIDATES)

    if xcol is None or ycol is None:
        print(f"[skip] required columns not found (xcol={xcol}, ycol={ycol})")
        present_sim = [c for c in _SIMILARITY_CANDIDATES if c in df.columns]
        present_use = [c for c in _USEFULNESS_CANDIDATES if c in df.columns]
        print("  similarity candidates present:", present_sim or "none")
        print("  usefulness candidates present:", present_use or "none")
        print("Tip: override with --xcol and --ycol.")
        return

    print(f"[info] Using x='{xcol}' (lower is better), y='{ycol}' (higher is better)")

    # Ensure model/seed exist (optional)
    if "model" not in df.columns:
        df["model"] = "NA"
    if "seed" not in df.columns:
        df["seed"] = np.nan

    data = df[["model", "seed", xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna(subset=[xcol, ycol])
    if data.empty:
        print("[skip] no finite rows to plot")
        return

    # Color/marker maps
    model_colors = model_palette(data["model"].astype(str).tolist())
    seed_vals = data["seed"].dropna().unique().tolist()
    try:
        seed_vals = [int(s) for s in seed_vals]
    except Exception:
        seed_vals = []
    seed_shapes = seed_markers(seed_vals if seed_vals else [0])

    # Figure
    fig, ax = new_figure(figsize=(7.6, 5.0))

    # Scatter grouped by (model, seed) if seed available
    has_seed = data["seed"].notna().any()
    group_cols = ["model", "seed"] if has_seed else ["model"]
    for keys, g in data.groupby(group_cols, dropna=False):
        if has_seed:
            model, seed = keys
            label = f"{model} • seed {int(seed) if pd.notna(seed) else 'NA'}"
            marker = seed_shapes.get(int(seed) if pd.notna(seed) else 0, "o")
        else:
            model = keys
            label = str(model)
            marker = "o"
        ax.scatter(
            g[xcol], g[ycol],
            s=38,
            c=model_colors.get(str(model), "#7f7f7f"),
            marker=marker,
            alpha=0.85,
            edgecolors="none",
            label=label,
        )

    # Pareto frontier: maximize y, minimize x → feed [y, x] into pareto_frontier
    pts = np.c_[data[ycol].values, data[xcol].values]
    pf_idx = pareto_frontier(pts)
    pf = data.iloc[pf_idx].sort_values(by=xcol, ascending=True)
    if not pf.empty:
        ax.plot(
            pf[xcol].values,
            pf[ycol].values,
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
            color="#444444",
            label="Pareto frontier",
        )
        if annotate_top > 0:
            top = pf.sort_values([ycol, xcol], ascending=[False, True]).head(int(annotate_top))
            for _, r in top.iterrows():
                ax.annotate(str(r["model"]), xy=(r[xcol], r[ycol]), xytext=(4, 4),
                            textcoords="offset points", fontsize=9)

    if logx:
        xmin = max(1e-6, float(np.nanmin(data[xcol].values)))
        ax.set_xscale("log")
        ax.set_xlim(left=xmin)

    # Dedup legend entries
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower left", ncol=2)

    ax.set_xlabel(_pretty_label(xcol, is_x=True))
    ax.set_ylabel(_pretty_label(ycol))
    ax.set_title(f"{_pretty_label(ycol)} vs {_pretty_label(xcol, is_x=True)} (Pareto)")

    savefig(outfile, tight=True, dpi=200 if not transparent else 300, transparent=transparent)

# ---- CLI --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Pareto: Macro-F1 vs (c)FID for all runs/configs")
    ap.add_argument("--jsonl", default=str(PLOT_OUT_DIR.parent / "summaries" / "phase1_summaries.jsonl"),
                    help="Path to consolidated JSONL")
    ap.add_argument("--xcol", default=None, help="Override similarity column (e.g., metrics.cfid_macro)")
    ap.add_argument("--ycol", default=None, help="Override usefulness column (e.g., utility_real_plus_synth.macro_f1)")
    ap.add_argument("--no-cfid", action="store_true", help="Kept for parity; auto-pick already prefers cFID")
    ap.add_argument("--logx", action="store_true", help="Log-scale the x axis")
    ap.add_argument("--annotate-top", type=int, default=0, help="Annotate top-K Pareto points (0=off)")
    ap.add_argument("--out", default=str(PLOT_OUT_DIR / "core" / "pareto_downstream_vs_similarity.png"),
                    help="Output PNG path")
    ap.add_argument("--transparent", action="store_true", help="Save with transparent background")
    args = ap.parse_args()

    df = read_jsonl(args.jsonl)
    if df.empty:
        print("[skip] empty DataFrame; nothing to plot")
        return

    plot_pareto(
        df,
        xcol=args.xcol,
        ycol=args.ycol,
        prefer_cfid=not args.no_cfid,
        logx=args.logx,
        annotate_top=args.annotate_top,
        outfile=Path(args.out),
        transparent=args.transparent,
    )

if __name__ == "__main__":
    main()
