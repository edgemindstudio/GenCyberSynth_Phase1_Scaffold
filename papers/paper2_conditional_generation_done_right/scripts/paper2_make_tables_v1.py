#!/usr/bin/env python3
"""
Paper 2 - Table builder v1

Reads evaluation summaries (latest.json) and produces:
- per-class cFID table + macro/weighted/worst
- per-class utility deltas table (optional if present)

Outputs go to:
papers/paper2_conditional_generation_done_right/results/tables/
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import csv

PAPER_DIR = Path("papers/paper2_conditional_generation_done_right")
OUT_DIR = PAPER_DIR / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Change these if you want to run on other models later
DEFAULT_MODEL = "gan"
DEFAULT_ARTS = Path("/home/bruno.fonkeng/gencys/artifacts_paper2")

def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r") as f:
        return json.load(f)

def _get(d: Dict[str, Any], keys: List[str], default=None):
    """Try multiple possible key paths."""
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _agg(values: List[Optional[float]], weights: Optional[List[float]] = None) -> Dict[str, Optional[float]]:
    """Return macro, weighted, worst (max) and p95 worst if possible."""
    clean = [(i, v) for i, v in enumerate(values) if v is not None and not math.isnan(v)]
    if not clean:
        return {"macro": None, "weighted": None, "worst": None, "p95": None}

    vals = [v for _, v in clean]
    macro = sum(vals) / len(vals)

    weighted = None
    if weights is not None and len(weights) == len(values):
        wv = [(weights[i], v) for i, v in clean if weights[i] is not None]
        sw = sum(w for w, _ in wv)
        if sw > 0:
            weighted = sum(w * v for w, v in wv) / sw

    worst = max(vals)

    # p95 (approx) on available values
    vals_sorted = sorted(vals)
    idx = int(0.95 * (len(vals_sorted) - 1))
    p95 = vals_sorted[idx]

    return {"macro": macro, "weighted": weighted, "worst": worst, "p95": p95}

def main(model: str = DEFAULT_MODEL, arts_root: Path = DEFAULT_ARTS) -> int:
    latest = arts_root / model / "summaries" / "latest.json"
    if not latest.exists():
        raise FileNotFoundError(f"Missing: {latest}")

    s = _load_json(latest)

    # Your logs show eval has util_bundle with cfid_per_class + cfid_macro.
    # Different runs may store under "metrics" or at top-level; handle both.
    cfid_per_class = (
        _get(s, ["metrics", "cfid_per_class"]) or
        _get(s, ["cfid_per_class"]) or
        _get(s, ["util_bundle", "cfid_per_class"])
    )
    cfid_macro = (
        _safe_float(_get(s, ["metrics", "cfid_macro"])) or
        _safe_float(_get(s, ["cfid_macro"])) or
        _safe_float(_get(s, ["util_bundle", "cfid_macro"]))
    )

    # Try to recover class supports (weights) from real_only per_class support
    supports = _get(s, ["metrics", "real_only", "per_class", "support"]) \
        or _get(s, ["real_only", "per_class", "support"])

    # Normalize per-class to list[float] with class ids 0..K-1 if possible
    per_class_vals: List[Optional[float]] = []
    if isinstance(cfid_per_class, list):
        per_class_vals = [_safe_float(x) for x in cfid_per_class]
    elif isinstance(cfid_per_class, dict):
        # dict with keys "0","1",...
        max_k = max(int(k) for k in cfid_per_class.keys())
        per_class_vals = [None] * (max_k + 1)
        for k, v in cfid_per_class.items():
            per_class_vals[int(k)] = _safe_float(v)
    else:
        per_class_vals = []

    # weights from supports
    weights: Optional[List[float]] = None
    if isinstance(supports, list) and len(supports) == len(per_class_vals):
        weights = [float(x) for x in supports]

    ag = _agg(per_class_vals, weights=weights)

    # Write table
    out_csv = OUT_DIR / f"paper2_{model}_cfid_table.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "macro", "weighted", "worst", "p95", "cfid_macro_reported"])
        w.writerow([model, "cfid", ag["macro"], ag["weighted"], ag["worst"], ag["p95"], cfid_macro])

    out_per_class = OUT_DIR / f"paper2_{model}_cfid_per_class.csv"
    with out_per_class.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "cfid"])
        for i, v in enumerate(per_class_vals):
            w.writerow([i, v])

    print(f"[ok] Wrote: {out_csv}")
    print(f"[ok] Wrote: {out_per_class}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())