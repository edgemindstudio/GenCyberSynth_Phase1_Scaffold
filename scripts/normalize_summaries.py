#!/usr/bin/env python3
"""
scripts/normalize_summaries.py

Goal
----
Harmonize per-run summary JSON files under:
  artifacts/*/summaries/summary_*.json

into a consistent schema so downstream tooling (build_jsonl, collect_scores,
tables/plots) can reliably read metrics.

Key behaviors
-------------
1) Idempotent + conservative
   - Safe to run multiple times.
   - Never overwrites an existing non-None value.

2) Supports multiple "shapes" written by older/newer pipelines
   - Nested dict style:
       {"metrics": {...}, "counts": {...}, "metrics_meta": {...}, ...}
   - Legacy dotted-key shims:
       {"metrics.kid": 0.1, "metrics.downstream.macro_f1": 0.92, ...}

3) Produces BOTH:
   A) Canonical nested fields (preferred for code)
   B) Top-level convenience fields (great for JSONL/CSV and quick greps):
        kid, balanced_acc, gen_precision, gen_recall,
        computed_at, metrics_version, kid_mode, ...

What this script DOES NOT do
----------------------------
It does not build the consolidated JSONL. That is done by scripts/build_jsonl.sh.
However, since we also write top-level keys into each summary JSON, build_jsonl.sh
can easily pass them through (or simply concatenate) to obtain top-level keys in JSONL.
"""

from __future__ import annotations

import glob
import json
from typing import Any, Dict, Optional


# -----------------------------
# Small utility helpers
# -----------------------------

def ensure_dict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Ensure d[key] exists and is a dict; return it."""
    v = d.get(key)
    if not isinstance(v, dict):
        v = {}
        d[key] = v
    return v


def set_if_missing(dst: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set dst[key] = value only if:
      - value is not None
      - dst[key] is missing OR None
    """
    if value is None:
        return
    if dst.get(key) is None:
        dst[key] = value


def dig(d: Dict[str, Any], *path: str) -> Any:
    """Safely dig nested dicts. Returns None if any component missing."""
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def first_non_none(*vals: Any) -> Any:
    """Return the first non-None value from vals; otherwise None."""
    for v in vals:
        if v is not None:
            return v
    return None


def inflate_dotted(d: Dict[str, Any], dotted_key: str) -> None:
    """
    Inflate a dotted key like "metrics.downstream.macro_f1" into nested dicts.

    If d contains the dotted key at top-level, ensure:
        d["metrics"]["downstream"]["macro_f1"] is set (if missing/None)

    Note: We keep the dotted key (harmless) for backward compatibility.
    """
    if dotted_key not in d:
        return
    value = d.get(dotted_key)

    parts = dotted_key.split(".")
    if not parts:
        return

    root = parts[0]
    cur = ensure_dict(d, root)

    for p in parts[1:-1]:
        cur = ensure_dict(cur, p)

    leaf = parts[-1]
    set_if_missing(cur, leaf, value)


# -----------------------------
# Normalization logic
# -----------------------------

def main() -> None:
    files = sorted(glob.glob("artifacts/*/summaries/summary_*.json"))
    print("Found", len(files), "summaries")

    for p in files:
        with open(p, "r") as f:
            d: Dict[str, Any] = json.load(f)

        # Canonical containers (always dicts)
        metrics = ensure_dict(d, "metrics")
        counts = ensure_dict(d, "counts")
        gen = ensure_dict(d, "generative")
        meta = ensure_dict(d, "metrics_meta")

        # ---------------------------------------------------------
        # 1) Counts normalization (nested + top-level convenience)
        # ---------------------------------------------------------
        num_real = first_non_none(
            counts.get("num_real"),
            counts.get("train_real"),
            d.get("counts.num_real"),
            d.get("num_real"),
            dig(d, "images", "train_real"),
        )
        num_fake = first_non_none(
            counts.get("num_fake"),
            counts.get("synthetic"),
            d.get("counts.num_fake"),
            d.get("num_fake"),
            dig(d, "images", "synthetic"),
        )

        set_if_missing(counts, "num_real", num_real)
        set_if_missing(counts, "num_fake", num_fake)

        # Keep top-level copies (handy for JSONL/CSV)
        set_if_missing(d, "num_real", counts.get("num_real"))
        set_if_missing(d, "num_fake", counts.get("num_fake"))

        # ---------------------------------------------------------
        # 2) Similarity/diversity metric normalization
        # ---------------------------------------------------------
        # Historically your CSV calls this ms_ssim, but in val_common.py it maps from "diversity".
        ms_ssim = first_non_none(
            metrics.get("ms_ssim"),
            d.get("metrics.ms_ssim"),
            d.get("ms_ssim"),
            gen.get("diversity"),
            dig(d, "generative", "diversity"),
        )
        set_if_missing(metrics, "ms_ssim", ms_ssim)
        set_if_missing(d, "ms_ssim", metrics.get("ms_ssim"))  # top-level convenience

        # ---------------------------------------------------------
        # 3) cFID/FID normalization across variants
        # ---------------------------------------------------------
        # cFID can appear in:
        #   - gen.cfid_macro (preferred modern)
        #   - metrics.cfid / metrics.cfid_macro
        #   - dotted: metrics.cfid / metrics.cfid_macro
        cfid = first_non_none(
            gen.get("cfid_macro"),
            metrics.get("cfid"),
            metrics.get("cfid_macro"),
            d.get("metrics.cfid"),
            d.get("metrics.cfid_macro"),
            d.get("generative.cfid_macro"),
            dig(d, "generative", "cfid_macro"),
        )
        if cfid is not None:
            set_if_missing(gen, "cfid_macro", cfid)
            set_if_missing(metrics, "cfid", cfid)
            set_if_missing(metrics, "cfid_macro", cfid)
            # Back-compat: some old scripts read fid_macro as "cfid" in Phase-1 tables
            set_if_missing(metrics, "fid_macro", metrics.get("fid_macro") or cfid)

        # FID can appear in:
        #   - gen.fid_macro / gen.fid
        #   - metrics.fid / metrics.fid_macro
        #   - dotted: metrics.fid_macro
        fid = first_non_none(
            gen.get("fid_macro"),
            gen.get("fid"),
            metrics.get("fid_macro"),
            metrics.get("fid"),
            d.get("metrics.fid_macro"),
            dig(d, "generative", "fid_macro"),
            dig(d, "generative", "fid"),
        )
        # Do NOT overwrite metrics.fid_macro if it exists; keep it conservative.
        if fid is not None:
            set_if_missing(gen, "fid_macro", fid)
            set_if_missing(metrics, "fid_macro", fid)

        # Top-level convenience fields for tables/JSONL
        set_if_missing(d, "cfid", metrics.get("cfid") or metrics.get("cfid_macro") or gen.get("cfid_macro"))
        set_if_missing(d, "fid", metrics.get("fid_macro") or gen.get("fid_macro") or gen.get("fid"))

        # ---------------------------------------------------------
        # 4) Downstream utility: inflate dotted keys + mirror from utility blocks
        # ---------------------------------------------------------
        # Inflate known dotted downstream shims into nested metrics.downstream
        inflate_dotted(d, "metrics.downstream.macro_f1")
        inflate_dotted(d, "metrics.downstream.macro_auprc")
        inflate_dotted(d, "metrics.downstream.balanced_acc")
        inflate_dotted(d, "metrics.downstream.precision")
        inflate_dotted(d, "metrics.downstream.recall")

        downstream = ensure_dict(metrics, "downstream")

        # Prefer utility_real_plus_synth if present (common in your pipeline)
        util_rs = d.get("utility_real_plus_synth")
        if not isinstance(util_rs, dict):
            util_rs = {}

        # Fill downstream core metrics
        set_if_missing(downstream, "macro_f1", first_non_none(
            downstream.get("macro_f1"),
            util_rs.get("macro_f1"),
            d.get("metrics.downstream.macro_f1"),
        ))
        set_if_missing(downstream, "macro_auprc", first_non_none(
            downstream.get("macro_auprc"),
            util_rs.get("macro_auprc"),
            d.get("metrics.downstream.macro_auprc"),
        ))

        # balanced_acc might appear as balanced_acc OR bal_acc OR balanced_accuracy
        bal = first_non_none(
            downstream.get("balanced_acc"),
            util_rs.get("balanced_acc"),
            util_rs.get("bal_acc"),
            util_rs.get("balanced_accuracy"),
            d.get("metrics.downstream.balanced_acc"),
        )
        set_if_missing(downstream, "balanced_acc", bal)

        # ---------------------------------------------------------
        # 5) KID + gen_precision/gen_recall + metrics_meta normalization
        # ---------------------------------------------------------
        # KID may appear in metrics, generative, or dotted legacy.
        kid = first_non_none(
            metrics.get("kid"),
            d.get("metrics.kid"),
            gen.get("kid"),
            dig(d, "generative", "kid"),
        )
        if kid is not None:
            set_if_missing(metrics, "kid", kid)
            set_if_missing(gen, "kid", kid)
            set_if_missing(d, "metrics.kid", kid)  # keep legacy dotted (optional)

        # gen_precision/gen_recall:
        # We interpret these as downstream classifier macro precision/recall.
        # They may appear in:
        #  - metrics.gen_precision / metrics.gen_recall
        #  - dotted: metrics.gen_precision / metrics.gen_recall
        #  - utility_real_plus_synth.macro_precision / macro_recall
        #  - utility_real_plus_synth.per_class.macro_precision / macro_recall (older layout)
        inflate_dotted(d, "metrics.gen_precision")
        inflate_dotted(d, "metrics.gen_recall")

        # Gather macro precision/recall from the best available place
        mp = first_non_none(
            metrics.get("gen_precision"),
            d.get("metrics.gen_precision"),
            util_rs.get("macro_precision"),
            dig(util_rs, "per_class", "macro_precision"),
            downstream.get("precision"),
            d.get("metrics.downstream.precision"),
        )
        mr = first_non_none(
            metrics.get("gen_recall"),
            d.get("metrics.gen_recall"),
            util_rs.get("macro_recall"),
            dig(util_rs, "per_class", "macro_recall"),
            downstream.get("recall"),
            d.get("metrics.downstream.recall"),
        )

        if mp is not None:
            set_if_missing(metrics, "gen_precision", mp)
            set_if_missing(downstream, "precision", mp)
            set_if_missing(d, "metrics.gen_precision", mp)         # dotted convenience
            set_if_missing(d, "metrics.downstream.precision", mp)  # dotted convenience

        if mr is not None:
            set_if_missing(metrics, "gen_recall", mr)
            set_if_missing(downstream, "recall", mr)
            set_if_missing(d, "metrics.gen_recall", mr)            # dotted convenience
            set_if_missing(d, "metrics.downstream.recall", mr)     # dotted convenience

        # Ensure utility_real_plus_synth contains macro_precision/macro_recall too
        # (helps humans + makes schema consistent with your val_common mapper).
        if isinstance(util_rs, dict):
            set_if_missing(util_rs, "macro_precision", mp)
            set_if_missing(util_rs, "macro_recall", mr)
            d["utility_real_plus_synth"] = util_rs

        # metrics_meta normalization:
        # Backfill/runtime stamps live here.
        # Sometimes scripts put these at top-level; we migrate conservatively into metrics_meta.
        set_if_missing(meta, "computed_at", first_non_none(meta.get("computed_at"), d.get("computed_at")))
        set_if_missing(meta, "metrics_version", first_non_none(meta.get("metrics_version"), d.get("metrics_version")))
        set_if_missing(meta, "kid_mode", first_non_none(meta.get("kid_mode"), d.get("kid_mode")))

        # ---------------------------------------------------------
        # 6) Promote top-level convenience keys (what you WANT in JSONL)
        # ---------------------------------------------------------
        # These are intentionally duplicated at top-level for easy JSONL/CSV usage.
        set_if_missing(d, "kid", metrics.get("kid") or gen.get("kid"))
        set_if_missing(d, "balanced_acc", downstream.get("balanced_acc"))
        set_if_missing(d, "macro_f1", downstream.get("macro_f1"))
        set_if_missing(d, "macro_auprc", downstream.get("macro_auprc"))
        set_if_missing(d, "gen_precision", metrics.get("gen_precision") or downstream.get("precision"))
        set_if_missing(d, "gen_recall", metrics.get("gen_recall") or downstream.get("recall"))

        set_if_missing(d, "computed_at", meta.get("computed_at"))
        set_if_missing(d, "metrics_version", meta.get("metrics_version"))
        set_if_missing(d, "kid_mode", meta.get("kid_mode"))

        # Write back (sorted keys keeps diffs stable)
        with open(p, "w") as f:
            json.dump(d, f, indent=2, sort_keys=True)

    print("Normalized", len(files), "summaries")


if __name__ == "__main__":
    main()
