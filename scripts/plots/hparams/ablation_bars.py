# scripts/plots/hparams/ablation_bars.py
"""
Ablation bars: Δ metric when toggling one feature / hyperparameter.

What
----
For a chosen hyperparameter/flag column (e.g., "cfg.use_ema" or "hparams.scheduler"),
compute the difference in a target metric between a **variant** and a **baseline**.
Bars show Δ = variant − baseline (with 95% bootstrap CIs) per group.

Typical targets (pick one per run):
  - utility_real_plus_synth.macro_f1        (higher is better)
  - metrics.cfid_macro / generative.cfid_macro (lower is better → --direction lower)
  - metrics.fid_macro / generative.fid_macro   (lower is better → --direction lower)
  - calibration ECE/Brier you may log (lower is better → --direction lower)

Grouping
--------
By default, we compute one bar **per model family** (df["model"]).
Use --group-by "" (empty) to aggregate all rows together,
or set --group-by some_column for a custom grouping.

Assumptions
-----------
- JSONL is one-run-per-line (artifacts/summaries/phase1_summaries.jsonl).
- The ablation column has exactly two values within each group (baseline vs variant).
- Seeds/replicates appear as separate lines and are paired/bootstrapped within groups.

Examples
--------
# EMA on/off → Δ Macro-F1 per model (positive = better with EMA on)
python scripts/plots/hparams/ablation_bars.py \
  --metric utility_real_plus_synth.macro_f1 \
  --ablate-by cfg.use_ema \
  --baseline 0 \
  --variant 1 \
  --direction higher

# Noise schedule ("linear" vs "cosine") → Δ cFID (lower is better)
python scripts/plots/hparams/ablation_bars.py \
  --metric generative.cfid_macro \
  --ablate-by hparams.noise_schedule \
  --baseline linear \
  --variant cosine \
  --direction lower
"""

from __future__ import annotations

import argparse
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots._common import read_jsonl, savefig, bootstrap_ci

# Figure root: support both names used across branches
try:
    from scripts.plots._common import PLOT_OUT_DIR as _FIG_ROOT
except Exception:
    from scripts.plots._common import PLOT_OUT as _FIG_ROOT  # type: ignore[assignment]
from pathlib import Path

PLOT_ROOT = Path(_FIG_ROOT)
OUT_DIR = PLOT_ROOT / "hparams" / "ablation"


# ---------- helpers ----------

def _coerce_cat(x: pd.Series) -> pd.Series:
    """Make ablation keys comparable (strings for robust equality)."""
    def _norm(v: Any) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "1" if bool(v) else "0"
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NA"
        return str(v)
    return x.map(_norm)


def _pick_baseline_variant(
    s: pd.Series,
    user_baseline: Optional[str],
    user_variant: Optional[str],
) -> Tuple[str, str]:
    """Decide baseline & variant labels from the observed values in s."""
    vals = sorted(set(_coerce_cat(s).dropna().tolist()))
    if len(vals) < 2:
        raise ValueError(f"Ablation column needs >=2 distinct values; saw {vals}")
    if user_baseline is not None and user_variant is not None:
        return str(user_baseline), str(user_variant)

    # Common heuristics: prefer ("0","1"), ("off","on"), etc.; else lexicographic
    prefs = [("0", "1"), ("False", "True"), ("false", "true"),
             ("off", "on"), ("linear", "cosine")]
    for b, v in prefs:
        if b in vals and v in vals:
            return b, v
    return vals[0], vals[1]


def _direction_sign(direction: str) -> int:
    """Return +1 for 'higher' and -1 for 'lower' (smaller is better)."""
    d = direction.strip().lower()
    if d in {"higher", "max", "bigger", "greater"}:
        return +1
    if d in {"lower", "min", "smaller", "less"}:
        return -1
    raise ValueError("--direction must be 'higher' or 'lower'")


def _pairwise_deltas(
    df_g: pd.DataFrame,
    metric: str,
    ablate_col: str,
    baseline_label: str,
    variant_label: str,
) -> np.ndarray:
    """
    Per-group deltas: delta = metric(variant) - metric(baseline)

    Pairs rows by simple alignment within each ablation bucket; if unequal sizes,
    align on the min count to avoid imbalance.
    """
    s_ab = _coerce_cat(df_g[ablate_col])
    a = df_g.loc[s_ab == variant_label, metric].astype(float).dropna().values
    b = df_g.loc[s_ab == baseline_label, metric].astype(float).dropna().values
    n = min(len(a), len(b))
    if n == 0:
        return np.array([])
    return a[:n] - b[:n]


def _summarize_group(
    df_g: pd.DataFrame,
    group_key: str,
    metric: str,
    ablate_col: str,
    baseline_label: str,
    variant_label: str,
    sign: int,
    ci_iters: int,
    alpha: float,
) -> Tuple[str, float, float, float]:
    """
    Returns (label, delta_mean_signed, err_lo, err_hi) where
      delta_mean_signed = sign * mean(variant - baseline)
    and CI half-widths are computed by bootstrap.
    """
    deltas = _pairwise_deltas(df_g, metric, ablate_col, baseline_label, variant_label)
    if deltas.size == 0:
        return group_key, np.nan, np.nan, np.nan
    m, lo, hi = bootstrap_ci(deltas, stat=np.mean, iters=ci_iters, alpha=alpha)
    # convert to "improvement" sign convention
    m *= sign
    lo *= sign
    hi *= sign
    return group_key, m, (m - lo), (hi - m)


# ---------- main plotting routine ----------

def plot_ablation_bars(
    jsonl_path: str,
    metric: str,
    ablate_by: str,
    baseline: Optional[str],
    variant: Optional[str],
    direction: str,
    group_by: Optional[str],
    title: Optional[str],
    outfile: str,
    ci_iters: int = 2000,
    alpha: float = 0.05,
) -> None:
    df = read_jsonl(jsonl_path)
    if df.empty:
        print("[skip] empty DataFrame from JSONL")
        return

    if metric not in df.columns:
        print(f"[skip] metric column not found: {metric}")
        return
    if ablate_by not in df.columns:
        print(f"[skip] ablation column not found: {ablate_by}")
        return

    # Decide baseline & variant labels
    baseline_label, variant_label = _pick_baseline_variant(df[ablate_by], baseline, variant)
    if baseline is not None and variant is not None:
        baseline_label, variant_label = str(baseline), str(variant)

    # Determine grouping (default: per model family)
    if group_by is None:
        group_by = "model"
    elif group_by.strip() == "":
        group_by = None

    sign = _direction_sign(direction)

    rows: List[Tuple[str, float, float, float]] = []
    if group_by is None:
        lbl, m, e_lo, e_hi = _summarize_group(
            df_g=df,
            group_key="all",
            metric=metric,
            ablate_col=ablate_by,
            baseline_label=baseline_label,
            variant_label=variant_label,
            sign=sign,
            ci_iters=ci_iters,
            alpha=alpha,
        )
        rows.append((lbl, m, e_lo, e_hi))
    else:
        for key, df_g in df.groupby(group_by, dropna=False):
            key_str = "NA" if key is None or (isinstance(key, float) and np.isnan(key)) else str(key)
            lbl, m, e_lo, e_hi = _summarize_group(
                df_g=df_g,
                group_key=key_str,
                metric=metric,
                ablate_col=ablate_by,
                baseline_label=baseline_label,
                variant_label=variant_label,
                sign=sign,
                ci_iters=ci_iters,
                alpha=alpha,
            )
            rows.append((lbl, m, e_lo, e_hi))

    res = pd.DataFrame(rows, columns=["label", "delta", "err_lo", "err_hi"]).sort_values(
        "delta", ascending=True, na_position="last"
    )

    # Plot
    plt.figure(figsize=(8.2, max(3.0, 0.45 * len(res))))
    y = np.arange(len(res))
    plt.barh(y, res["delta"].values, xerr=[res["err_lo"].values, res["err_hi"].values], alpha=0.9)
    plt.yticks(y, res["label"].values)
    plt.axvline(0, lw=1, color="k")

    # Label depends on direction convention
    if direction.lower().startswith("higher"):
        xlabel = f"Δ {metric} (variant − baseline), positive = improvement"
    else:
        xlabel = f"Δ {metric} (variant − baseline), positive = improvement (lower is better)"
    plt.xlabel(xlabel)

    if title:
        plt.title(title)
    else:
        plt.title(
            f"Ablation: {ablate_by} (baseline='{baseline_label}' → variant='{variant_label}')\n"
            f"Δ computed on '{metric}' ({direction.lower()})"
        )

    plt.grid(alpha=0.25, axis="x")
    savefig(OUT_DIR / outfile)


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation bars: Δ metric when toggling one hyperparameter/feature.")
    ap.add_argument("--jsonl", default="artifacts/summaries/phase1_summaries.jsonl",
                    help="Path to consolidated JSONL (one run per line).")
    ap.add_argument("--metric", required=True,
                    help="Metric column, e.g. 'utility_real_plus_synth.macro_f1', 'generative.cfid_macro'.")
    ap.add_argument("--ablate-by", required=True,
                    help="Column that encodes the ablation switch (e.g., 'cfg.use_ema', 'hparams.noise_schedule').")
    ap.add_argument("--baseline", default=None,
                    help="Baseline label/value (string-bool-ints accepted, coerced to string). If omitted, inferred.")
    ap.add_argument("--variant", default=None,
                    help="Variant label/value to compare against baseline. If omitted, inferred.")
    ap.add_argument("--direction", default="higher", choices=["higher", "lower"],
                    help="Does larger metric mean better (higher) or smaller (lower)? Controls Δ sign convention.")
    ap.add_argument("--group-by", default="model",
                    help="Group to aggregate over (default: 'model'). Use empty string '' for no grouping.")
    ap.add_argument("--title", default=None, help="Optional custom title.")
    ap.add_argument("--outfile", default="ablation_bars.png",
                    help="Output filename (saved under figures/hparams/ablation/).")
    ap.add_argument("--ci-iters", type=int, default=2000, help="Bootstrap iterations.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha for 1−alpha CI.")
    args = ap.parse_args()

    plot_ablation_bars(
        jsonl_path=args.jsonl,
        metric=args.metric,
        ablate_by=args.ablate_by,
        baseline=args.baseline,
        variant=args.variant,
        direction=args.direction,
        group_by=args.group_by,
        title=args.title,
        outfile=args.outfile,
        ci_iters=args.ci_iters,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
