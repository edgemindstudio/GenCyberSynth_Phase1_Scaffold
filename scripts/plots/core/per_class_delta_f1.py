# scripts/plots/core/per_class_delta_f1.py
"""
Per-class ΔF1 (Real+Synth − Real)

What
----
Bar chart of per-class F1 improvement when training/evaluating with synthetic
augmentation. Optionally overlays jittered per-seed points and can emit a
heatmap variant. Aggregates across all runs found in the consolidated JSONL.

Why
---
Answers “who benefited?” — critical for imbalance stories.

Inputs
------
- artifacts/summaries/phase1_summaries.jsonl  (one JSON object per line)
  Expected keys (per your schema):
    - metrics.real_only.per_class_f1.<class_id>
    - metrics.real_plus_synth.per_class_f1.<class_id>
    - seed (optional; inferred from run_id if missing)
    - model (for filtering / heatmap-by-model)

Outputs
-------
- artifacts/figures/core/per_class_delta_f1_bars.png
- artifacts/figures/core/per_class_delta_f1_heatmap.png   (if --heatmap)
- artifacts/figures/core/per_class_delta_f1_model_heatmap.png (if --heatmap-by-model)

Notes
-----
- If per-class F1 columns are not present in the JSONL, the script exits gracefully.
- 95% CIs are bootstrapped across runs (per class).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots._common import (
    read_jsonl,
    savefig,
    bootstrap_ci,
)

# Resolve figure output root from _common (backward compatible names)
try:
    from scripts.plots._common import PLOT_OUT_DIR as _FIG_ROOT
except Exception:
    from scripts.plots._common import PLOT_OUT as _FIG_ROOT  # type: ignore[assignment]
FIG_ROOT = Path(_FIG_ROOT)  # ensure Path
(FIG_ROOT / "core").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_class_columns(df: pd.DataFrame, prefix: str) -> Dict[str, str]:
    """
    Return a mapping {class_id: column_name} for columns that start with `prefix`.
    Example prefix: "metrics.real_only.per_class_f1."
    """
    out: Dict[str, str] = {}
    plen = len(prefix)
    for c in df.columns:
        if c.startswith(prefix):
            cls_id = c[plen:]
            if cls_id:  # non-empty suffix
                out[cls_id] = c
    return out


def _collect_deltas(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Build a long-form DataFrame with one row per (run, class_id):
        ['model', 'seed', 'class', 'delta_f1', 'f1_real', 'f1_rs']
    Returns None if needed columns are missing.
    """
    # Which prefixes exist?
    ro_prefix = "metrics.real_only.per_class_f1."
    rs_prefix = "metrics.real_plus_synth.per_class_f1."

    ro_map = _per_class_columns(df, ro_prefix)
    rs_map = _per_class_columns(df, rs_prefix)

    if not ro_map or not rs_map:
        return None

    # Intersect class ids present in both splits
    class_ids = sorted(set(ro_map.keys()).intersection(rs_map.keys()))
    if not class_ids:
        return None

    # Ensure seed column exists (read_jsonl already tries to infer it)
    if "seed" not in df.columns:
        df = df.copy()
        df["seed"] = np.nan

    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        for cid in class_ids:
            f1_r = r.get(ro_map[cid], np.nan)
            f1_s = r.get(rs_map[cid], np.nan)
            if pd.isna(f1_r) or pd.isna(f1_s):
                continue
            rows.append({
                "model": r.get("model", "unknown"),
                "seed": r.get("seed", np.nan),
                "class": str(cid),
                "f1_real": float(f1_r),
                "f1_rs": float(f1_s),
                "delta_f1": float(f1_s) - float(f1_r),
            })

    if not rows:
        return None
    return pd.DataFrame(rows)


def _jitter_points(ax, y_positions: np.ndarray, x_values: np.ndarray, *, jitter: float = 0.13, alpha: float = 0.5):
    """
    Overlay jittered points (per-seed deltas) at horizontal positions x_values,
    vertical base positions y_positions (one per class).
    """
    if len(x_values) == 0:
        return
    # y_positions is an array of length = #classes (bar centers)
    # x_values is a flattened list of points mapped per class; we will feed this
    # function per-class slices from the caller.
    rng = np.random.default_rng(0)
    y_jitter = rng.uniform(-jitter, +jitter, size=len(x_values))
    ax.scatter(x_values, y_positions + y_jitter, s=12, alpha=alpha, edgecolors="none")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bars_with_ci(
    df_runs: pd.DataFrame,
    *,
    model_filter: Optional[List[str]] = None,
    topk_abs: Optional[int] = None,
    out_path: Path = FIG_ROOT / "core" / "per_class_delta_f1_bars.png",
) -> None:
    """
    Horizontal bar plot of mean ΔF1 per class with 95% bootstrap CI across runs.
    Overlays jittered per-seed deltas for transparency.

    Parameters
    ----------
    df_runs : DataFrame returned by _collect_deltas()
    model_filter : optional list of model names to include
    topk_abs : if set, only plot the Top-K classes by absolute mean ΔF1
    out_path : output PNG path
    """
    data = df_runs.copy()

    if model_filter:
        data = data[data["model"].isin(model_filter)]

    if data.empty:
        print("[skip] No per-class rows to plot after filtering")
        return

    # Aggregate per class across runs
    agg = (
        data.groupby("class", as_index=False)["delta_f1"]
        .mean()  # returns columns: ['class', 'delta_f1']
        .rename(columns={"delta_f1": "mean_delta"})
    )

    # Order by mean delta (ascending = losses at top, gains at bottom)
    agg = agg.sort_values("mean_delta", ascending=True)

    # Optional: restrict to top-K by absolute effect size (for compact papers)
    if topk_abs and topk_abs > 0:
        top_idx = agg["mean_delta"].abs().sort_values(ascending=False).head(topk_abs).index
        agg = agg.loc[top_idx].sort_values("mean_delta", ascending=True)
    else:
        agg = agg.sort_values("mean_delta", ascending=True)

    # Compute 95% CI via bootstrap across runs PER CLASS
    ci_lo: List[float] = []
    ci_hi: List[float] = []
    for cls in agg["class"].tolist():
        v = data.loc[data["class"] == cls, "delta_f1"].values.astype(float)
        m, lo, hi = bootstrap_ci(v, stat=np.mean, iters=2000, alpha=0.05)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.36 * len(agg))))
    y = np.arange(len(agg), dtype=float)
    ax.barh(y, agg["mean_delta"].values, xerr=[ci_lo, ci_hi], alpha=0.9)

    # Overlay jittered per-seed deltas as dots
    # Build a per-class x list in agg order
    xs: List[float] = []
    ys: List[float] = []
    for i, cls in enumerate(agg["class"].tolist()):
        cls_vals = data.loc[data["class"] == cls, "delta_f1"].values.astype(float)
        if len(cls_vals) == 0:
            continue
        _jitter_points(ax, np.full(len(cls_vals), y[i]), cls_vals, jitter=0.14, alpha=0.5)

    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(agg["class"].tolist())
    ax.set_xlabel("ΔF1 = F1(Real+Synth) − F1(Real)")
    ax.set_title("Per-class ΔF1 (bars with 95% CI)")

    savefig(out_path)


def plot_heatmap(
    df_runs: pd.DataFrame,
    *,
    out_path: Path = FIG_ROOT / "core" / "per_class_delta_f1_heatmap.png",
    vmin: float = -0.2,
    vmax: float = +0.2,
) -> None:
    """
    Heatmap of mean ΔF1 per class (aggregated across models/seeds).
    """
    agg = df_runs.groupby("class", as_index=False)["delta_f1"].mean()
    if agg.empty:
        print("[skip] nothing to heatmap")
        return

    classes = agg["class"].astype(str).tolist()
    mat = agg["delta_f1"].to_numpy().reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(4.2, max(3.0, 0.32 * len(classes))))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks([0])
    ax.set_xticklabels(["ΔF1 (R+S − R)"])
    ax.set_title("Per-class ΔF1 (heatmap)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(out_path)


def plot_heatmap_by_model(
    df_runs: pd.DataFrame,
    *,
    out_path: Path = FIG_ROOT / "core" / "per_class_delta_f1_model_heatmap.png",
    vmin: float = -0.2,
    vmax: float = +0.2,
) -> None:
    """
    Heatmap with rows = classes, columns = model families, values = mean ΔF1.
    """
    pivot = (
        df_runs.pivot_table(
            index="class", columns="model", values="delta_f1", aggfunc="mean"
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.empty:
        print("[skip] no data for model heatmap")
        return

    fig, ax = plt.subplots(figsize=(max(4.0, 1.0 + 0.6 * pivot.shape[1]),
                                    max(3.0, 0.28 * pivot.shape[0])))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.astype(str).tolist())
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(str).tolist(), rotation=45, ha="right")
    ax.set_title("Per-class ΔF1 by model (mean across runs)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Per-class ΔF1 plots (bars / heatmaps)")
    ap.add_argument(
        "--jsonl",
        default="artifacts/summaries/phase1_summaries.jsonl",
        help="Path to consolidated JSONL",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Filter to these model families (space-separated)",
    )
    ap.add_argument(
        "--topk-abs",
        type=int,
        default=None,
        help="Plot only Top-K classes by absolute mean ΔF1 (bars only)",
    )
    ap.add_argument(
        "--heatmap",
        action="store_true",
        help="Also emit a class-only heatmap of mean ΔF1",
    )
    ap.add_argument(
        "--heatmap-by-model",
        action="store_true",
        help="Also emit a (class × model) heatmap of mean ΔF1",
    )
    args = ap.parse_args()

    df = read_jsonl(args.jsonl)
    if df.empty:
        print("[skip] JSONL not found or empty")
        return

    runs = _collect_deltas(df)
    if runs is None or runs.empty:
        print("[skip] No per-class F1 fields found; cannot build ΔF1 plots.")
        return

    # Bars (primary)
    plot_bars_with_ci(
        runs,
        model_filter=args.models,
        topk_abs=args.topk_abs,
        out_path=FIG_ROOT / "core" / "per_class_delta_f1_bars.png",
    )

    # Optional heatmaps
    if args.heatmap:
        plot_heatmap(
            runs,
            out_path=FIG_ROOT / "core" / "per_class_delta_f1_heatmap.png",
        )

    if args.heatmap_by_model:
        plot_heatmap_by_model(
            runs,
            out_path=FIG_ROOT / "core" / "per_class_delta_f1_model_heatmap.png",
        )


if __name__ == "__main__":
    main()
