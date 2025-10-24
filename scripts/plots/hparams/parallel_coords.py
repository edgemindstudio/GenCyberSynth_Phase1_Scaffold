# scripts/plots/hparams/parallel_coords.py
"""
Parallel coordinates for hyperparameter search.

What
----
Draw one polyline per run across a sequence of dimensions that mix
hyperparameters (inputs) and metrics (outputs). Useful to see which
regions of the search space lead to good results.

Key features
------------
- Reads consolidated JSONL: artifacts/summaries/phase1_summaries.jsonl
- Mix any set of columns (hp + metrics) with --cols (numeric or categorical)
- Categorical hparams are ordinal-encoded per column (stable sorted labels)
- Mark which columns are "lower is better" (they will be inverted)
- Per-column min–max normalization to [0, 1] for fair visual comparison
- Color lines by a grouping column (default: model)
- Downsample per group to avoid spaghetti
- Saves to artifacts/figures/hparams/parallel/<outfile>
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots._common import read_jsonl, savefig

# Figure root (support both names used across branches)
try:
    from scripts.plots._common import PLOT_OUT_DIR as _FIG_ROOT
except Exception:
    from scripts.plots._common import PLOT_OUT as _FIG_ROOT  # type: ignore[assignment]
from pathlib import Path

PLOT_ROOT = Path(_FIG_ROOT)
OUT_DIR = PLOT_ROOT / "hparams" / "parallel"


# -------------------------- utilities --------------------------

def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """Return the subset of cols that actually exist in df, preserving order."""
    ok = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[warn] missing columns skipped: {missing}")
    return ok


def _invert_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Invert 'lower is better' columns so that 'higher is better' visually."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            with np.errstate(all="ignore"):
                v = pd.to_numeric(out[c], errors="coerce")
                mx, mn = np.nanmax(v.values), np.nanmin(v.values)
                if np.isfinite(mx) and np.isfinite(mn):
                    out[c] = mx + mn - v
                else:
                    # If it's categorical/nan, skip inversion (already encoded/handled elsewhere)
                    pass
    return out


def _minmax01(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Column-wise min–max to [0,1]; if constant or all-NaN, set to 0.5."""
    out = df.copy()
    for c in cols:
        v = pd.to_numeric(out[c], errors="coerce")
        lo = np.nanmin(v.values)
        hi = np.nanmax(v.values)
        if not np.isfinite(lo) or not np.isfinite(hi):
            out[c] = 0.5
            continue
        if hi - lo < 1e-12:
            out[c] = 0.5
        else:
            out[c] = (v - lo) / (hi - lo)
    return out


def _encode_categoricals(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Ordinal-encode any non-numeric columns among `cols`.
    For each such column, sort unique non-NA labels and map to 0..K-1.
    Returns (df_encoded, mapping_dict) for reference.
    """
    out = df.copy()
    mapping: Dict[str, List[str]] = {}
    for c in cols:
        # Detect non-numeric entries
        as_num = pd.to_numeric(out[c], errors="coerce")
        non_numeric_mask = as_num.isna() & out[c].notna()
        if non_numeric_mask.any():
            labels = sorted(pd.unique(out.loc[non_numeric_mask, c].astype(str)))
            # also include any stringy entries that pandas didn't mark above (defensive)
            labels = sorted(pd.unique(pd.Series(labels, dtype=str)))
            code_map = {lab: i for i, lab in enumerate(labels)}
            mapping[c] = labels
            # map values: strings -> code, numbers stay as numbers
            def _to_code(v):
                if pd.isna(v):
                    return np.nan
                try:
                    return float(v)
                except Exception:
                    return float(code_map.get(str(v), np.nan))
            out[c] = out[c].map(_to_code)
        else:
            # already numeric (or NA-only), keep as-is
            pass
    if mapping:
        for col, labs in mapping.items():
            print(f"[info] encoded categorical '{col}' → ordinal 0..{len(labs)-1}: {labs}")
    return out, mapping


def _downsample_per_group(df: pd.DataFrame, group_col: Optional[str], per_group: int, rng: np.random.Generator) -> pd.DataFrame:
    if group_col is None or group_col not in df.columns or per_group <= 0:
        return df
    parts = []
    for _, g in df.groupby(group_col, dropna=False):
        if len(g) > per_group:
            idx = rng.choice(g.index.values, size=per_group, replace=False)
            parts.append(g.loc[idx])
        else:
            parts.append(g)
    return pd.concat(parts, axis=0).reset_index(drop=True)


# -------------------------- plotting --------------------------

def plot_parallel(
    jsonl_path: str,
    cols: Sequence[str],
    lower_is_cols: Sequence[str],
    group_by: Optional[str],
    per_group: int,
    alpha: float,
    linewidth: float,
    outfile: str,
    title: Optional[str],
    seed: int = 0,
) -> None:
    df = read_jsonl(jsonl_path)
    if df.empty:
        print("[skip] empty DataFrame from JSONL")
        return

    # Keep only requested columns + group
    keep = _ensure_cols(df, cols)
    if not keep:
        print("[skip] none of the requested --cols exist in JSONL")
        return

    need = list(keep)
    if group_by and group_by not in need and group_by in df.columns:
        need.append(group_by)
    df = df[need].copy()

    # Encode categoricals among the selected columns (so we don't drop rows later)
    df, cat_map = _encode_categoricals(df, keep)

    # Coerce selected columns to numeric
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with any NaN across selected columns (group column can be NaN)
    df = df.dropna(subset=keep, how="any")
    if df.empty:
        print("[skip] all rows had NaNs for selected columns after encoding")
        return

    # Invert "lower is better" columns so all columns follow 'higher=better'
    df = _invert_cols(df, lower_is_cols)

    # Normalize each selected column to [0,1]
    df_norm = _minmax01(df, keep)

    # Downsample per group if requested
    rng = np.random.default_rng(seed)
    df_plot = _downsample_per_group(df_norm, group_by, per_group, rng)

    # Colors by group
    if group_by and group_by in df_plot.columns:
        groups = df_plot[group_by].astype(str).fillna("NA").values
        uniq = pd.Index(sorted(pd.unique(groups)))
        color_map = {g: plt.cm.tab10(i % 10) for i, g in enumerate(uniq)}
        colors = [color_map[g] for g in groups]
        legend_items = uniq.tolist()
    else:
        colors = [plt.cm.tab10(0)] * len(df_plot)
        legend_items = []

    x = np.arange(len(keep))  # one axis per column

    plt.figure(figsize=(max(8.5, 1.25 * len(keep)), 6.0))

    # Light vertical axes
    for xi in x:
        plt.axvline(xi, color="k", lw=0.5, alpha=0.2, zorder=0)

    # Plot polylines
    vals = df_plot[keep].values
    for i in range(vals.shape[0]):
        plt.plot(x, vals[i, :], alpha=alpha, lw=linewidth, color=colors[i])

    # Ticks and labels
    plt.xticks(x, keep, rotation=25, ha="right")
    plt.yticks([0.0, 0.5, 1.0], ["low", "mid", "high"])
    plt.ylim(-0.05, 1.05)
    plt.grid(axis="y", alpha=0.2)

    ttl = title or "Parallel Coordinates (normalized, higher=better)"
    if lower_is_cols:
        ttl += f"\nInverted: {', '.join(lower_is_cols)}"
    plt.title(ttl)

    # Legend (group names)
    if legend_items:
        # Show up to 10 entries; if more, show first 10 and add "+N more"
        show = legend_items[:10]
        labels = show[:]
        if len(legend_items) > 10:
            labels.append(f"+{len(legend_items) - 10} more")
            show.append(None)  # dummy for spacing
        proxies = [plt.Line2D([0], [0], color=plt.cm.tab10(i % 10), lw=2) for i, _ in enumerate(show) if _ is not None]
        plt.legend(proxies, labels[:len(proxies)], title=(group_by or "group"), loc="upper right", frameon=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    savefig(OUT_DIR / outfile)


# -------------------------- CLI --------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel coordinates over hyperparameters and metrics.")
    ap.add_argument("--jsonl", default="artifacts/summaries/phase1_summaries.jsonl",
                    help="Path to consolidated JSONL (one run per line).")
    ap.add_argument("--cols", nargs="+", required=True,
                    help="Ordered list of columns to render as axes (mix hparams + metrics; categoricals allowed).")
    ap.add_argument("--lower-is", nargs="*", default=[],
                    help="Subset of --cols where lower is better (will be inverted so higher=better visually).")
    ap.add_argument("--group-by", default="model",
                    help="Column to color by (default: model). Use '' to disable grouping.")
    ap.add_argument("--per-group", type=int, default=100,
                    help="Max lines per group to plot (downsample if exceeded).")
    ap.add_argument("--alpha", type=float, default=0.35, help="Line alpha.")
    ap.add_argument("--linewidth", type=float, default=1.2, help="Line width.")
    ap.add_argument("--outfile", default="parallel_coords.png", help="Output filename.")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for downsampling.")
    args = ap.parse_args()

    group_by = None if args.group_by.strip() == "" else args.group_by

    plot_parallel(
        jsonl_path=args.jsonl,
        cols=args.cols,
        lower_is_cols=args.lower_is,
        group_by=group_by,
        per_group=args.per_group,
        alpha=args.alpha,
        linewidth=args.linewidth,
        outfile=args.outfile,
        title=args.title,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
