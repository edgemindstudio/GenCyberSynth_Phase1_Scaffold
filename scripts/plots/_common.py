# scripts/plots/_common.py
"""
Shared plotting utilities for GenCyberSynth figures.

This module centralizes small-but-important conveniences:
- Safe JSONL loading and flattening into a Pandas DataFrame
- Robust numeric coercion and seed extraction
- Compatibility shim: maps current keys (generative.*, utility_*.*) to legacy
  metrics.* columns expected by some plotting scripts
- Consistent Matplotlib defaults (fonts, dpi, grid)
- Small helpers for saving figures, bootstrapped CIs, palettes, markers

Usage:
    from scripts.plots._common import (
        DEFAULT_JSONL, PLOT_OUT_DIR,
        read_jsonl, savefig, bootstrap_ci, new_figure,
        model_palette, seed_markers, pareto_frontier
    )
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------
# Paths & constants
# ----------------------------

DEFAULT_JSONL = Path("artifacts/summaries/phase1_summaries.jsonl")
PLOT_OUT_DIR = Path("artifacts/figures")
PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUT = PLOT_OUT_DIR  # backward-compat alias

# Global, sensible defaults for Matplotlib across all plots
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "figure.autolayout": False,  # we'll call tight_layout() explicitly
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


# ----------------------------
# Small utilities
# ----------------------------

def _safe(val: Any, *keys: Any, default: Any = None) -> Any:
    """
    Safely index nested dict-like objects, returning `default` on any failure.

    Example:
        _safe(row, "metrics", "real_plus_synth", "macro_f1", default=np.nan)
    """
    cur = val
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(k, None)
        else:
            return default
    return default if cur is None else cur


# Alias for callers that prefer a non-underscore name.
safe_get = _safe


def _coerce_numeric_inplace(df: pd.DataFrame) -> None:
    """
    Best-effort numeric coercion: for any object-typed column, attempt
    to convert to numeric; leave as-is on failure.
    """
    for c in df.columns:
        if df[c].dtype == "object":
            coerced = pd.to_numeric(df[c], errors="coerce")
            # adopt only if conversion clearly helped (avoid turning text into NaNs)
            if coerced.notna().sum() >= max(1, int(0.5 * len(df))):
                df[c] = coerced


def _extract_seed_from_run_id(run_id: Any) -> Optional[int]:
    """
    Try to infer seed from run_id tokens (e.g., 'diffusion_seed43_20250101T000000Z').
    """
    if not isinstance(run_id, str):
        return None
    for token in run_id.replace("-", "_").split("_"):
        if token.startswith("seed"):
            try:
                return int(token.replace("seed", ""))
            except ValueError:
                return None
    return None


# ----------------------------
# Data loading
# ----------------------------

def read_jsonl(jsonl_path: str | Path = DEFAULT_JSONL) -> pd.DataFrame:
    """
    Load a JSONL file (one JSON object per line) and flatten into a DataFrame.

    Columns commonly produced by GenCyberSynth:
      - model, run_id, seed
      - generative.* (fid_macro, cfid_macro, ms_ssim, ...)
      - utility_real_only.*, utility_real_plus_synth.* (macro_f1, ece, brier, ...)
      - counts.*

    Compatibility shim:
      For plotting code that expects legacy columns, we synthesize:
        metrics.cfid              ← generative.cfid_macro  (fallback to generative.fid_macro)
        metrics.fid_macro         ← generative.fid_macro
        metrics.cfid_macro        ← generative.cfid_macro
        metrics.downstream.macro_f1 ← utility_real_plus_synth.macro_f1
        metrics.real_plus_synth.macro_f1 ← utility_real_plus_synth.macro_f1
        metrics.real_only.macro_f1 ← utility_real_only.macro_f1
    """
    p = Path(jsonl_path)
    if not p.exists():
        print(f"[skip] JSONL not found: {p}", file=sys.stderr)
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[warn] bad JSONL line {ln}: {e}", file=sys.stderr)

    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    # Numeric coercion
    _coerce_numeric_inplace(df)

    # Ensure a seed column: prefer 'seed' if present, else infer from run_id
    if "seed" not in df.columns:
        df["seed"] = df.get("run_id", pd.Series([None] * len(df))).map(_extract_seed_from_run_id)

    # Normalize model to string
    if "model" in df.columns:
        df["model"] = df["model"].astype(str)

    # ----------------------------
    # Compatibility shim
    # ----------------------------
    # (c)FID family
    if "metrics.cfid" not in df.columns:
        if "generative.cfid_macro" in df.columns:
            df["metrics.cfid"] = df["generative.cfid_macro"]
        elif "metrics.cfid_macro" in df.columns:
            df["metrics.cfid"] = df["metrics.cfid_macro"]
        elif "generative.fid_macro" in df.columns:
            # fallback: treat FID as cFID when cFID absent (for plotting only)
            df["metrics.cfid"] = df["generative.fid_macro"]

    if "metrics.cfid_macro" not in df.columns and "generative.cfid_macro" in df.columns:
        df["metrics.cfid_macro"] = df["generative.cfid_macro"]

    if "metrics.fid_macro" not in df.columns and "generative.fid_macro" in df.columns:
        df["metrics.fid_macro"] = df["generative.fid_macro"]

    # Downstream macro-F1 (Real+Synth)
    if "metrics.downstream.macro_f1" not in df.columns:
        if "utility_real_plus_synth.macro_f1" in df.columns:
            df["metrics.downstream.macro_f1"] = df["utility_real_plus_synth.macro_f1"]
        elif "metrics.real_plus_synth.macro_f1" in df.columns:
            df["metrics.downstream.macro_f1"] = df["metrics.real_plus_synth.macro_f1"]

    # Mirror some utility fields to legacy locations that other plots may reference
    if "metrics.real_plus_synth.macro_f1" not in df.columns and "utility_real_plus_synth.macro_f1" in df.columns:
        df["metrics.real_plus_synth.macro_f1"] = df["utility_real_plus_synth.macro_f1"]

    if "metrics.real_only.macro_f1" not in df.columns and "utility_real_only.macro_f1" in df.columns:
        df["metrics.real_only.macro_f1"] = df["utility_real_only.macro_f1"]

    # Similarity extras
    if "metrics.ms_ssim" not in df.columns and "generative.ms_ssim" in df.columns:
        df["metrics.ms_ssim"] = df["generative.ms_ssim"]

    if "metrics.kid" not in df.columns and "generative.kid" in df.columns:
        df["metrics.kid"] = df["generative.kid"]

    # Optional: counts mirror (helps some figures)
    if "counts.num_real" not in df.columns and "counts.train_real" in df.columns:
        df["counts.num_real"] = df["counts.train_real"]

    if "counts.num_fake" not in df.columns and "counts.synthetic" in df.columns:
        df["counts.num_fake"] = df["counts.synthetic"]

    # Optional: per-class MS-SSIM alias (generative → metrics)
    gen_prefix = "generative.ms_ssim_per_class."
    met_prefix = "metrics.ms_ssim_per_class."
    if any(c.startswith(gen_prefix) for c in df.columns) and not any(c.startswith(met_prefix) for c in df.columns):
        for c in list(df.columns):
            if c.startswith(gen_prefix):
                df[met_prefix + c[len(gen_prefix):]] = df[c]

    return df


# ----------------------------
# Plot helpers
# ----------------------------

def new_figure(figsize: Tuple[float, float] = (6.0, 4.0)) -> Tuple[plt.Figure, plt.Axes]:
    """Create a new Matplotlib figure with a single Axes and consistent defaults."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def savefig(path: Path | str, *, tight: bool = True, dpi: int = 200, transparent: bool = False) -> None:
    """Save the current Matplotlib figure to `path`, ensuring parent directories exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            plt.tight_layout()
        except Exception:
            pass
    plt.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None, transparent=transparent)
    print(f"[ok] wrote {path}")


def bootstrap_ci(
    values: Sequence[float],
    stat: callable = np.mean,
    iters: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Simple nonparametric bootstrap confidence interval.

    Returns (point_estimate, lo, hi) for the given statistic `stat`.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    boots = np.empty(iters, dtype=float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat(v[idx])

    boots.sort()
    lo = boots[int(alpha / 2 * iters)]
    hi = boots[int((1 - alpha / 2) * iters)]
    return (stat(v), float(lo), float(hi))


def model_palette(models: Iterable[str]) -> Dict[str, str]:
    """
    Deterministic color assignment per model family (colorblind-friendly palette).
    """
    base = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    models = list(dict.fromkeys(models))  # preserve order, unique
    mapping: Dict[str, str] = {}
    for i, m in enumerate(models):
        mapping[m] = base[i % len(base)]
    return mapping


def seed_markers(seeds: Iterable[int]) -> Dict[int, str]:
    """Deterministic marker selection for seeds."""
    base = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    seeds = list(dict.fromkeys(seeds))
    mapping: Dict[int, str] = {}
    for i, s in enumerate(seeds):
        mapping[int(s)] = base[i % len(base)]
    return mapping


def pareto_frontier(points: np.ndarray) -> np.ndarray:
    """
    Compute Pareto frontier indices for points of shape (N, 2),
    where we *maximize* x (usefulness, e.g., Macro-F1) and *minimize* y
    (distance, e.g., cFID). Returns indices of frontier points.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be (N, 2) array")

    # Sort by x desc (better) then y asc (better)
    order = np.lexsort((points[:, 1], -points[:, 0]))
    pts = points[order]
    frontier_idx_local: List[int] = []
    best_y = math.inf
    for i, (_, y) in enumerate(pts):
        if y < best_y:
            frontier_idx_local.append(i)
            best_y = y
    return order[frontier_idx_local]

