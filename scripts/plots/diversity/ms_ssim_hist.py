# scripts/plots/diversity/ms_ssim_hist.py
"""
MS-SSIM diversity histograms

What
----
Plots the distribution of MS-SSIM values gathered from your consolidated JSONL.
Lower MS-SSIM ⇒ more diversity among generated samples.

Modes
-----
1) Global (default): one histogram over all runs.
2) Per-model (--per-model): overlaid histograms, one per model family.
3) Per-class (--per-class): if per-class MS-SSIM keys exist (e.g.,
   metrics.ms_ssim_per_class.<class_id>), draws a small-multiples grid.

Inputs
------
- artifacts/summaries/phase1_summaries.jsonl

Expected fields
---------------
- metrics.ms_ssim                       (float per run)                 [common]
- metrics.ms_ssim_per_class.<class_id>  (optional floats per run)       [optional]
- model                                  (string)                       [for --per-model]

Outputs
-------
- artifacts/figures/diversity/ms_ssim_hist.png                 (global)
- artifacts/figures/diversity/ms_ssim_hist_per_model.png       (per-model)
- artifacts/figures/diversity/ms_ssim_per_class_grid.png       (per-class grid)

Notes
-----
- MS-SSIM is bounded in [0, 1]. We clamp plotted values to this range defensively.
- No seaborn: pure matplotlib. One chart per figure (no subplots mixing).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots._common import (
    read_jsonl,
    savefig,
)

# Resolve figure output root from _common (supports both names)
try:
    from scripts.plots._common import PLOT_OUT_DIR as _FIG_ROOT
except Exception:
    from scripts.plots._common import PLOT_OUT as _FIG_ROOT  # type: ignore[assignment]
FIG_ROOT = Path(_FIG_ROOT)
(FIG_ROOT / "diversity").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_01(x: np.ndarray) -> np.ndarray:
    """Clamp numeric array to [0,1] and drop NaNs."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return x
    return np.clip(x, 0.0, 1.0)


def _collect_ms_ssim(df: pd.DataFrame) -> np.ndarray:
    col = _pick_ms_ssim_col(df)
    if not col:
        return np.array([], dtype=float)
    return _clamp_01(df[col].to_numpy(dtype=float))


def _collect_ms_ssim_by_model(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    col = _pick_ms_ssim_col(df)
    if not col or "model" not in df.columns:
        return {}
    out: Dict[str, List[float]] = {}
    for _, r in df.iterrows():
        m = str(r.get("model", "unknown"))
        v = r.get(col, np.nan)
        if pd.isna(v):
            continue
        out.setdefault(m, []).append(float(v))
    return {k: _clamp_01(np.array(v, dtype=float)) for k, v in out.items() if v}


def _collect_ms_ssim_per_class(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Search for columns named 'metrics.ms_ssim_per_class.<class_id>'.
    Return {class_id: values[]} aggregated across runs.
    """
    prefix = "metrics.ms_ssim_per_class."
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return {}
    out: Dict[str, List[float]] = {}
    for c in cols:
        cls = c[len(prefix):]
        vals = df[c].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            out.setdefault(cls, []).extend(vals.tolist())
    return {k: _clamp_01(np.array(v, dtype=float)) for k, v in out.items() if len(v) > 0}


def _pick_ms_ssim_col(df: pd.DataFrame) -> str | None:
    for c in ("metrics.ms_ssim", "generative.ms_ssim"):
        if c in df.columns and df[c].notna().any():
            return c
    # if none have non-NaN, still return one if present (all-NaN case)
    for c in ("metrics.ms_ssim", "generative.ms_ssim"):
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_global(values: np.ndarray, bins: int, out: Path) -> None:
    if values.size == 0:
        print("[skip] No metrics.ms_ssim found")
        return
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=bins, alpha=0.9, rwidth=0.92)
    plt.xlim(0, 1)
    plt.xlabel("MS-SSIM (lower = more diverse)")
    plt.ylabel("Count of runs")
    mu = float(np.mean(values))
    plt.title(f"MS-SSIM diversity (all runs) — mean={mu:.3f}, n={values.size}")
    savefig(out)


def plot_per_model(by_model: Dict[str, np.ndarray], bins: int, out: Path) -> None:
    if not by_model:
        print("[skip] No per-model MS-SSIM data (need 'model' & 'metrics.ms_ssim').")
        return

    plt.figure(figsize=(8.5, 5.2))
    # Overlay histograms by model; keep edges off to avoid clutter
    models = sorted(by_model.keys())
    colors = plt.cm.tab10.colors
    for i, m in enumerate(models):
        v = by_model[m]
        if v.size == 0:
            continue
        plt.hist(v, bins=bins, alpha=0.50, label=m, rwidth=0.92,
                 color=colors[i % len(colors)])

    plt.xlim(0, 1)
    plt.xlabel("MS-SSIM (lower = more diverse)")
    plt.ylabel("Count of runs")
    plt.title("MS-SSIM diversity by model")
    plt.legend(title="Model", ncol=min(3, len(models)))
    savefig(out)


def plot_per_class_grid(per_class: Dict[str, np.ndarray], bins: int, out: Path) -> None:
    if not per_class:
        print("[skip] No per-class MS-SSIM columns found (metrics.ms_ssim_per_class.*).")
        return

    classes = sorted(per_class.keys(), key=lambda k: (len(k), k))
    n = len(classes)
    # Grid sizing: up to 4 columns
    cols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for idx, cls in enumerate(classes):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        v = per_class[cls]
        if v.size:
            ax.hist(v, bins=bins, alpha=0.9, rwidth=0.92)
        ax.set_xlim(0, 1)
        ax.set_title(f"class {cls}")
        if r == rows - 1:
            ax.set_xlabel("MS-SSIM")
        if c == 0:
            ax.set_ylabel("Count")

    # Hide any leftover axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle("MS-SSIM per class (lower = more diverse)", y=0.995)
    plt.tight_layout()
    savefig(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="MS-SSIM diversity histograms")
    ap.add_argument(
        "--jsonl",
        default="artifacts/summaries/phase1_summaries.jsonl",
        help="Path to consolidated JSONL",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins",
    )
    ap.add_argument(
        "--per-model",
        action="store_true",
        help="Overlay histograms by model family",
    )
    ap.add_argument(
        "--per-class",
        action="store_true",
        help="If per-class MS-SSIM exists, draw a grid of class-wise histograms",
    )
    args = ap.parse_args()

    df = read_jsonl(args.jsonl)
    if df.empty:
        print("[skip] JSONL not found or empty")
        return

    # Global
    vals = _collect_ms_ssim(df)
    plot_global(vals, args.bins, FIG_ROOT / "diversity" / "ms_ssim_hist.png")

    # Per-model
    if args.per_model:
        by_model = _collect_ms_ssim_by_model(df)
        plot_per_model(by_model, args.bins, FIG_ROOT / "diversity" / "ms_ssim_hist_per_model.png")

    # Per-class grid (if available)
    if args.per_class:
        per_cls = _collect_ms_ssim_per_class(df)
        plot_per_class_grid(per_cls, args.bins, FIG_ROOT / "diversity" / "ms_ssim_per_class_grid.png")


if __name__ == "__main__":
    main()
