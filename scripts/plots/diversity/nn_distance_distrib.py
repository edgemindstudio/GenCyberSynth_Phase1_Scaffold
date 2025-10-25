# scripts/plots/diversity/nn_distance_distrib.py
"""
Nearest-Neighbor distance distributions (feature space)

Visualize the distance from each generated sample to its nearest *real* neighbor
in a fixed feature space (your frozen domain encoder).

Inputs
------
- artifacts/summaries/phase1_summaries.jsonl (or another JSONL via --jsonl)

We support any of these schemas (any subset is OK):
1) Per-sample arrays (preferred for histograms by sample):
   - memorization.nn_dists
   - metrics.nn_dists
   - nn_dists

2) Per-run nested dict (use --which to pick a statistic):
   - diversity.nn_distance: {"mean":..., "min":..., "p50":..., "p90":...}
   - metrics.nn_distance:   {"mean":..., "min":..., "p50":..., "p90":...}
   - nn_distance:           {"mean":..., "min":..., "p50":..., "p90":...}

3) Per-run flat scalars (aliases for --which):
   - memorization.nn_dist_mean / metrics.nn_dist_mean / nn_distance_mean
   - memorization.nn_dist_min  / metrics.nn_dist_min  / nn_distance_min
   - memorization.nn_dist_p50  / metrics.nn_dist_p50  / nn_distance_p50
   - memorization.nn_dist_p90  / metrics.nn_dist_p90  / nn_distance_p90

Outputs
-------
- artifacts/figures/diversity/nn_dist_hist_<which>.png
- artifacts/figures/diversity/nn_dist_hist_<which>_per_model.png    (with --per-model)
- artifacts/figures/diversity/nn_dist_violin_<which>_per_model.png  (with --violin --per-model)

Notes
-----
- Distances are non-negative; we clamp to [0, inf) and drop NaNs.
- Pure matplotlib (no seaborn). One chart per figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots._common import read_jsonl, savefig

# Figure root from _common (supports both names)
try:
    from scripts.plots._common import PLOT_OUT_DIR as _FIG_ROOT
except Exception:
    from scripts.plots._common import PLOT_OUT as _FIG_ROOT  # type: ignore[assignment]
FIG_ROOT = Path(_FIG_ROOT)
(FIG_ROOT / "diversity").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Column aliases / fallbacks
# ---------------------------------------------------------------------------

# Per-sample arrays (preferred)
ARRAY_COLS = [
    "memorization.nn_dists",
    "metrics.nn_dists",
    "nn_dists",
]

# Per-run nested dicts containing keys like {"mean","min","p50","p90"}
DICT_COLS = [
    "diversity.nn_distance",
    "metrics.nn_distance",
    "nn_distance",
]

# Per-run flat scalar aliases for each statistic
ALT_SCALAR_COLS = {
    "mean": [
        "memorization.nn_dist_mean",
        "metrics.nn_dist_mean",
        "nn_distance_mean",
        "diversity.nn_distance.mean",   # in case it's already flattened by loader
        "metrics.nn_distance.mean",
        "nn_distance.mean",
    ],
    "min": [
        "memorization.nn_dist_min",
        "metrics.nn_dist_min",
        "nn_distance_min",
        "diversity.nn_distance.min",
        "metrics.nn_distance.min",
        "nn_distance.min",
    ],
    "p50": [
        "memorization.nn_dist_p50",
        "metrics.nn_dist_p50",
        "nn_distance_p50",
        "diversity.nn_distance.p50",
        "metrics.nn_distance.p50",
        "nn_distance.p50",
        "diversity.nn_distance.median",  # sometimes written as median
        "metrics.nn_distance.median",
        "nn_distance.median",
    ],
    "p90": [
        "memorization.nn_dist_p90",
        "metrics.nn_dist_p90",
        "nn_distance_p90",
        "diversity.nn_distance.p90",
        "metrics.nn_distance.p90",
        "nn_distance.p90",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_1d_array(x) -> np.ndarray:
    """Coerce a JSONL cell to a 1D float array if it looks like a list/array."""
    if isinstance(x, (list, tuple)):
        try:
            arr = np.asarray(x, dtype=float)
            return arr.ravel()
        except Exception:
            return np.array([], dtype=float)
    return np.array([], dtype=float)


def _nonneg_dropna(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return arr
    return arr[arr >= 0.0]


def _collect_scalar(df: pd.DataFrame, col: str) -> np.ndarray:
    """Collect a scalar column across runs (drop NaNs; keep non-negative)."""
    if col not in df.columns:
        return np.array([], dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return _nonneg_dropna(vals)


def _collect_array(df: pd.DataFrame, col: str) -> np.ndarray:
    """Collect and concatenate array-like column across runs."""
    if col not in df.columns:
        return np.array([], dtype=float)
    chunks: List[np.ndarray] = []
    for v in df[col].tolist():
        arr = _to_1d_array(v)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return np.array([], dtype=float)
    return _nonneg_dropna(np.concatenate(chunks, axis=0))


def _collect_from_dict(df: pd.DataFrame, col: str, which: str) -> np.ndarray:
    """
    Collect a statistic from rows where column is a dict, e.g.,
      row[col] == {"mean":..., "min":..., "p50":..., "p90":...}
    """
    if col not in df.columns:
        return np.array([], dtype=float)
    vals: List[float] = []
    for v in df[col].tolist():
        if isinstance(v, dict) and which in v:
            try:
                val = float(v[which])
                if np.isfinite(val) and val >= 0.0:
                    vals.append(val)
            except Exception:
                pass
    if not vals:
        return np.array([], dtype=float)
    return np.asarray(vals, dtype=float)


def _choose_source(df: pd.DataFrame, which: str) -> Tuple[str, np.ndarray]:
    """
    Determine which column to use for the requested 'which' statistic:
      1) Prefer per-sample arrays (…nn_dists): returns all samples across runs.
      2) Else try per-run nested dicts (…nn_distance) extracting 'which'.
      3) Else fall back to scalar columns (…nn_dist_<which>), trying aliases.
    Returns (source_name, values).
    """
    # 1) Per-sample arrays
    for arr_col in ARRAY_COLS:
        if arr_col in df.columns:
            vals = _collect_array(df, arr_col)
            if vals.size:
                return (arr_col, vals)

    # 2) Nested dicts with the requested key
    for dcol in DICT_COLS:
        if dcol in df.columns:
            vals = _collect_from_dict(df, dcol, which)
            if vals.size:
                return (dcol, vals)

    # 3) Scalar aliases
    for col in ALT_SCALAR_COLS.get(which, []):
        if col in df.columns:
            vals = _collect_scalar(df, col)
            if vals.size:
                return (col, vals)

    # Final fallback: return a primary alias even if empty (keeps messaging tidy)
    primary = (ARRAY_COLS[0] if which == "mean" else ALT_SCALAR_COLS[which][0])
    return (primary, np.array([], dtype=float))


def _by_model(df: pd.DataFrame, source_col: str, which: str) -> Dict[str, np.ndarray]:
    """
    Split values by model. Works for arrays, dicts, and scalars.
    Returns {model: 1D array of values}.
    """
    if "model" not in df.columns:
        return {}

    out: Dict[str, List[float]] = {}

    if source_col in ARRAY_COLS:
        # Per-run arrays: iterate rows and label by model (concatenate arrays)
        for _, r in df.iterrows():
            m = str(r.get("model", "unknown"))
            arr = _to_1d_array(r.get(source_col))
            if arr.size:
                out.setdefault(m, []).extend(arr.tolist())
    elif source_col in DICT_COLS:
        # Per-run dicts: take the chosen statistic per run
        for _, r in df.iterrows():
            m = str(r.get("model", "unknown"))
            v = r.get(source_col)
            if isinstance(v, dict) and which in v:
                try:
                    val = float(v[which])
                    if np.isfinite(val) and val >= 0.0:
                        out.setdefault(m, []).append(val)
                except Exception:
                    pass
    else:
        # Scalar per run
        for _, r in df.iterrows():
            m = str(r.get("model", "unknown"))
            v = r.get(source_col, np.nan)
            if pd.isna(v):
                continue
            try:
                val = float(v)
                if np.isfinite(val) and val >= 0.0:
                    out.setdefault(m, []).append(val)
            except Exception:
                pass

    # Finalize
    return {k: _nonneg_dropna(np.array(v, dtype=float)) for k, v in out.items() if len(v) > 0}


def _maybe_clip(values: np.ndarray, clip: float | None) -> np.ndarray:
    """Optionally clip distances to [0, clip]."""
    if values.size == 0 or clip is None:
        return values
    return values[values <= float(clip)]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_hist(values: np.ndarray, which: str, bins: int, out: Path, *, title_suffix: str = "", logx: bool = False) -> None:
    if values.size == 0:
        print(f"[skip] No NN distance values for '{which}'")
        return
    plt.figure(figsize=(7.5, 4.8))
    plt.hist(values, bins=bins, alpha=0.9, rwidth=0.92)
    if logx:
        plt.xscale("log")
    mu, med = float(np.mean(values)), float(np.median(values))
    plt.xlabel("NN distance in feature space (real ↔ synth)")
    plt.ylabel("Count")
    ttl = f"Nearest-Neighbor distances — {which}"
    if title_suffix:
        ttl += f" ({title_suffix})"
    ttl += f" | mean={mu:.3f}, median={med:.3f}, n={values.size}"
    plt.title(ttl)
    plt.grid(alpha=0.25)
    savefig(out)


def plot_hist_per_model(by_model: Dict[str, np.ndarray], which: str, bins: int, out: Path, *, title_suffix: str = "", logx: bool = False) -> None:
    if not by_model:
        print(f"[skip] No per-model NN distance values for '{which}'")
        return
    plt.figure(figsize=(9.5, 5.4))
    colors = plt.cm.tab10.colors
    models = sorted(by_model.keys())
    for i, m in enumerate(models):
        v = by_model[m]
        if v.size == 0:
            continue
        plt.hist(v, bins=bins, alpha=0.5, rwidth=0.92, label=m, color=colors[i % len(colors)])
    if logx:
        plt.xscale("log")
    plt.xlabel("NN distance in feature space (real ↔ synth)")
    plt.ylabel("Count")
    ttl = f"Nearest-Neighbor distances by model — {which}"
    if title_suffix:
        ttl += f" ({title_suffix})"
    plt.title(ttl)
    plt.legend(title="Model", ncol=min(3, len(models)))
    plt.grid(alpha=0.25)
    savefig(out)


def plot_violin_per_model(by_model: Dict[str, np.ndarray], which: str, out: Path, *, title_suffix: str = "") -> None:
    if not by_model:
        print(f"[skip] No per-model NN distance values for '{which}'")
        return
    models = sorted(by_model.keys())
    data = [by_model[m] for m in models]
    plt.figure(figsize=(10, 5.2))
    plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    # X tick labels at 1..N
    plt.xticks(np.arange(1, len(models) + 1), models, rotation=0)
    plt.ylabel("NN distance in feature space (real ↔ synth)")
    ttl = f"Nearest-Neighbor distance distribution by model — {which}"
    if title_suffix:
        ttl += f" ({title_suffix})"
    plt.title(ttl)
    plt.grid(axis="y", alpha=0.25)
    savefig(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Nearest-Neighbor distance distributions")
    ap.add_argument("--jsonl", default="artifacts/summaries/phase1_summaries.jsonl", help="Path to consolidated JSONL file")
    ap.add_argument("--which", choices=["mean", "min", "p50", "p90"], default="mean",
                    help="Which statistic to use when per-sample arrays are not present")
    ap.add_argument("--bins", type=int, default=30, help="Number of histogram bins")
    ap.add_argument("--per-model", action="store_true", help="Overlay per-model histograms")
    ap.add_argument("--violin", action="store_true", help="Per-model violin summary (requires --per-model)")
    ap.add_argument("--logx", action="store_true", help="Log-scale the x axis")
    ap.add_argument("--clip", type=float, default=None, help="Clip distances to [0, CLIP] before plotting")
    args = ap.parse_args()

    df = read_jsonl(args.jsonl)
    if df.empty:
        print("[skip] JSONL not found or empty")
        return

    # Global values + source chosen
    source_col, values = _choose_source(df, which=args.which)
    title_suffix = (
        "arrays" if source_col in ARRAY_COLS
        else ("dict" if source_col in DICT_COLS else source_col.split(".")[-1])
    )
    values = _maybe_clip(values, args.clip)

    # 1) Global histogram
    plot_hist(values, args.which, args.bins,
              FIG_ROOT / "diversity" / f"nn_dist_hist_{args.which}.png",
              title_suffix=title_suffix, logx=args.logx)

    # 2) Per-model overlays and violin
    if args.per_model:
        by_model = _by_model(df, source_col, args.which)
        # clip each model's vector if requested
        by_model = {m: _maybe_clip(v, args.clip) for m, v in by_model.items()}
        plot_hist_per_model(by_model, args.which, args.bins,
                            FIG_ROOT / "diversity" / f"nn_dist_hist_{args.which}_per_model.png",
                            title_suffix=title_suffix, logx=args.logx)
        if args.violin:
            plot_violin_per_model(by_model, args.which,
                                  FIG_ROOT / "diversity" / f"nn_dist_violin_{args.which}_per_model.png",
                                  title_suffix=title_suffix)


if __name__ == "__main__":
    main()
