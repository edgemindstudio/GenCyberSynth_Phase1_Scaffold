#!/usr/bin/env python3
"""
scripts/jsonl_to_csv.py

Convert consolidated JSONL -> CSV table for README/report previews.

This version exports the "real" metrics you care about:
- generative: FID/cFID/KID (as available)
- downstream: macro_f1 / macro_auprc / balanced_acc
- downstream macro precision/recall ("gen_precision/gen_recall" in your Phase-1 table)
- provenance: metrics_meta fields (runtime vs backfill)

Usage:
  python scripts/jsonl_to_csv.py \
      [--src artifacts/summaries/phase1_summaries.jsonl] \
      [--dst artifacts/phase1_scores.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _dig(d: Dict[str, Any], *path: str) -> Any:
    """Safely dig into nested dicts."""
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _first(*vals):
    """Return first non-None value."""
    for v in vals:
        if v is not None:
            return v
    return None


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _as_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="artifacts/summaries/phase1_summaries.jsonl")
    ap.add_argument("--dst", default="artifacts/phase1_scores.csv")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise SystemExit(f"[ERROR] Missing JSONL: {src}. Run: make summaries-jsonl")

    rows = []
    for i, line in enumerate(src.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"[ERROR] Bad JSON on line {i}: {e}") from e

        metrics = d.get("metrics") or {}
        counts = d.get("counts") or {}
        meta = d.get("metrics_meta") or {}

        # --- run identifiers ---
        run_id = d.get("run_id")
        model = d.get("model")

        # --- counts ---
        num_fake = _first(
            counts.get("num_fake"),
            counts.get("synthetic"),
            d.get("counts.num_fake"),
            _dig(d, "images", "synthetic"),
        )
        num_real = _first(
            counts.get("num_real"),
            counts.get("train_real"),
            d.get("counts.num_real"),
            _dig(d, "images", "train_real"),
        )

        # --- generative metrics ---
        # These may appear in different places depending on the pipeline stage
        fid = _first(
            metrics.get("fid_macro"),
            metrics.get("fid"),
            d.get("metrics.fid_macro"),
            _dig(d, "generative", "fid_macro"),
            _dig(d, "generative", "fid"),
        )

        cfid = _first(
            metrics.get("cfid"),
            metrics.get("cfid_macro"),
            d.get("metrics.cfid"),
            d.get("metrics.cfid_macro"),
            _dig(d, "generative", "cfid_macro"),
        )

        kid = _first(
            metrics.get("kid"),
            d.get("metrics.kid"),
            _dig(d, "generative", "kid"),
        )

        # --- diversity (your CSV column name is ms_ssim historically) ---
        ms_ssim = _first(
            metrics.get("ms_ssim"),
            d.get("metrics.ms_ssim"),
            _dig(d, "generative", "diversity"),  # in your val_common this is "diversity"
        )

        # --- downstream utility metrics ---
        macro_f1 = _first(
            _dig(metrics, "downstream", "macro_f1"),
            d.get("metrics.downstream.macro_f1"),
            _dig(d, "utility_real_plus_synth", "macro_f1"),
        )

        macro_auprc = _first(
            _dig(metrics, "downstream", "macro_auprc"),
            d.get("metrics.downstream.macro_auprc"),
            _dig(d, "utility_real_plus_synth", "macro_auprc"),
        )

        balanced_acc = _first(
            _dig(metrics, "downstream", "balanced_acc"),
            d.get("metrics.downstream.balanced_acc"),
            _dig(d, "utility_real_plus_synth", "balanced_accuracy"),
            _dig(d, "utility_real_plus_synth", "bal_acc"),
        )

        # --- "gen_precision/gen_recall" ---
        # We interpret these as downstream classifier macro precision/recall.
        gen_precision = _first(
            metrics.get("gen_precision"),
            d.get("metrics.gen_precision"),
            _dig(d, "utility_real_plus_synth", "macro_precision"),
            _dig(d, "utility_real_plus_synth", "per_class", "macro_precision"),
        )

        gen_recall = _first(
            metrics.get("gen_recall"),
            d.get("metrics.gen_recall"),
            _dig(d, "utility_real_plus_synth", "macro_recall"),
            _dig(d, "utility_real_plus_synth", "per_class", "macro_recall"),
        )

        # --- provenance/meta ---
        computed_at = _first(
            meta.get("computed_at"),
            d.get("metrics_meta", {}).get("computed_at"),
        )
        metrics_version = _first(
            meta.get("metrics_version"),
            d.get("metrics_meta", {}).get("metrics_version"),
        )
        kid_mode = _first(
            meta.get("kid_mode"),
            d.get("metrics_meta", {}).get("kid_mode"),
        )

        rows.append({
            "run_id": run_id,
            "model": model,
            "num_real": _as_int(num_real),
            "num_fake": _as_int(num_fake),

            "fid": _as_float(fid),
            "cfid": _as_float(cfid),
            "kid": _as_float(kid),
            "ms_ssim": _as_float(ms_ssim),

            "balanced_acc": _as_float(balanced_acc),
            "macro_f1": _as_float(macro_f1),
            "macro_auprc": _as_float(macro_auprc),
            "gen_precision": _as_float(gen_precision),
            "gen_recall": _as_float(gen_recall),

            "computed_at": computed_at,
            "metrics_version": metrics_version,
            "kid_mode": kid_mode,
        })

    if not rows:
        raise SystemExit("[ERROR] No rows parsed from JSONL; is it empty?")

    dst.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {dst} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
