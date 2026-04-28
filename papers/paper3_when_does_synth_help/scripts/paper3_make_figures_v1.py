#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt


def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def to_int(x, default=None):
    if x is None:
        return default
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="papers/paper3_when_does_synth_help/results/raw/paper3_results_paper.csv",
        type=Path,
    )
    ap.add_argument(
        "--out_dir",
        default="papers/paper3_when_does_synth_help/results/figures",
        type=Path,
    )
    args = ap.parse_args()

    rows = read_csv(args.in_csv)
    if not rows:
        raise SystemExit("[error] No rows found in input CSV")

    # Normalize numeric fields
    for r in rows:
        r["budget"] = to_int(r.get("synth_budget_per_class"), default=None)
        r["delta_f1"] = to_float(r.get("delta_macro_f1"))
        r["delta_bal"] = to_float(r.get("delta_bal_acc"))
        r["delta_auprc"] = to_float(r.get("delta_macro_auprc"))

        # For real-only rows, deltas may be blank; define them as 0 for plotting baseline
        if r["budget"] == 0 and r["delta_f1"] is None:
            r["delta_f1"] = 0.0
        if r["budget"] == 0 and r["delta_bal"] is None:
            r["delta_bal"] = 0.0
        if r["budget"] == 0 and r["delta_auprc"] is None:
            r["delta_auprc"] = 0.0

    # Keep rows with known budget + delta values
    rows = [r for r in rows if r["budget"] is not None]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Δ macro-F1 vs budget ---
    xs = [r["budget"] for r in rows if r["delta_f1"] is not None]
    ys = [r["delta_f1"] for r in rows if r["delta_f1"] is not None]

    plt.figure()
    plt.axhline(0.0)
    plt.scatter(xs, ys)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Synthetic budget per class")
    plt.ylabel("Δ Macro-F1 (Real+Synth − Real)")
    plt.title("Paper 3: Δ Macro-F1 vs Budget (current regimes)")
    out1 = args.out_dir / "paper3_delta_macro_f1_vs_budget.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {out1}")

    # --- Plot 2: Δ Balanced Accuracy vs budget ---
    xs = [r["budget"] for r in rows if r["delta_bal"] is not None]
    ys = [r["delta_bal"] for r in rows if r["delta_bal"] is not None]

    plt.figure()
    plt.axhline(0.0)
    plt.scatter(xs, ys)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Synthetic budget per class")
    plt.ylabel("Δ Balanced Acc (Real+Synth − Real)")
    plt.title("Paper 3: Δ Balanced Accuracy vs Budget (current regimes)")
    out2 = args.out_dir / "paper3_delta_bal_acc_vs_budget.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {out2}")

    # --- Plot 3: Δ Macro-AUPRC vs budget ---
    xs = [r["budget"] for r in rows if r["delta_auprc"] is not None]
    ys = [r["delta_auprc"] for r in rows if r["delta_auprc"] is not None]

    plt.figure()
    plt.axhline(0.0)
    plt.scatter(xs, ys)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Synthetic budget per class")
    plt.ylabel("Δ Macro-AUPRC (Real+Synth − Real)")
    plt.title("Paper 3: Δ Macro-AUPRC vs Budget (current regimes)")
    out3 = args.out_dir / "paper3_delta_macro_auprc_vs_budget.png"
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {out3}")


if __name__ == "__main__":
    main()