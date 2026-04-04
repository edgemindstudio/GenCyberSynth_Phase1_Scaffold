# scripts/eval_write_summary.py
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


# -----------------------------------------------------------------------------
# Helpers: make values JSON-safe
# -----------------------------------------------------------------------------
def _to_jsonable(v: Any) -> Any:
    """
    Best-effort conversion of values to JSON-safe types.

    Handles:
      - NumPy scalars (np.float32, np.int64, ...) via .item()
      - NaN / Inf -> None (JSON has no NaN/Inf)
    """
    try:
        if hasattr(v, "item"):
            v = v.item()  # e.g. np.float32(1.2) -> 1.2
    except Exception:
        pass

    if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
        return None

    return v


def _get(d: Optional[Mapping[str, Any]], key: str) -> Any:
    """Safe dict get for Mapping/None."""
    return None if d is None else d.get(key)


def _attach_audit_fields(rec: dict[str, Any], run_meta: Optional[Mapping[str, Any]]) -> None:
    """
    Attach provenance/audit metadata to the record we write to disk.

    Why:
      - Reviewers often require proof of:
          * exact config file path used
          * config hash
          * git commit
          * evaluation caps
          * budget_per_class
      - app/main.py attaches cfg["run_meta"] at runtime.
      - eval/runner.py should pass that run_meta into this writer.

    Output:
      - rec["run_meta"] (canonical nested block)
      - plus top-level shims for grep/legacy tools:
          config_path, config_sha1, git_commit, caps, budget_per_class
    """
    if not run_meta:
        return

    # Canonical nested block
    rec["run_meta"] = {str(k): _to_jsonable(v) for k, v in dict(run_meta).items()}

    # Top-level shims (helps older scripts & quick greps)
    rec["config_path"] = _to_jsonable(run_meta.get("config_path"))
    rec["config_sha1"] = _to_jsonable(run_meta.get("config_sha1"))
    rec["git_commit"] = _to_jsonable(run_meta.get("git_commit"))
    rec["caps"] = _to_jsonable(run_meta.get("caps"))
    rec["budget_per_class"] = _to_jsonable(run_meta.get("budget_per_class"))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def write_phase2_summary(
    *,
    out_json: str,
    model: str,
    seed: int,
    generative: Mapping[str, Any] | None,
    util_real: Mapping[str, Any] | None,
    util_rs: Mapping[str, Any] | None,
    counts: Mapping[str, Any] | None,
    run_id: Optional[str] = None,
    # NEW: provenance/audit info (recommended)
    run_meta: Mapping[str, Any] | None = None,
    # Optional per-class F1 dicts like {"0": 0.71, "1": 0.63, ...}
    util_real_per_class: Mapping[str, Any] | None = None,
    util_rs_per_class: Mapping[str, Any] | None = None,
) -> None:
    """
    Append a flattened, plot-friendly JSON line with both legacy and new keys.

    This function writes ONE JSON object as ONE line (JSONL-style) to out_json.

    Inputs
    ------
    out_json:
      File path (usually artifacts/<model>/summaries/summary_YYYYMMDD_HHMMSS.json).
      NOTE: despite the extension ".json", this writer APPENDS a JSON line.
      That matches your Phase-2 "jsonl-like" summary convention.

    model, seed, run_id:
      Run identity fields for grouping and reproducibility.

    generative:
      Quality/diversity metrics (FID/KID/MS-SSIM/CFID etc.), typically computed in runner.py.

    util_real, util_rs:
      Downstream utility results for Real-only and Real+Synth training (macro_f1, etc.).

    counts:
      Helpful sanity fields for number of real and synthetic samples.

    run_meta (NEW):
      Provenance metadata (config_path/hash, git_commit, caps, budget_per_class).
      If provided, it will be written into the record.
    """
    # -------------------------------------------------------------------------
    # Build record (the object we will append)
    # -------------------------------------------------------------------------
    rec: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": str(model),
        "seed": int(seed),
        "run_id": run_id,

        # Preferred structured blocks (phase-2)
        "generative": {
            "fid_macro":  _to_jsonable(_get(generative, "fid_macro")),
            "cfid_macro": _to_jsonable(_get(generative, "cfid_macro")),
            "kid":        _to_jsonable(_get(generative, "kid")),
            "ms_ssim":    _to_jsonable(_get(generative, "ms_ssim")),
        },
        # "utility_real_only": {
        #     "macro_f1": _to_jsonable(_get(util_real, "macro_f1")),
        # },
        # "utility_real_plus_synth": {
        #     "macro_f1": _to_jsonable(_get(util_rs, "macro_f1")),
        # },

        "utility_real_only": {
            "macro_f1": _to_jsonable(_get(util_real, "macro_f1")),
            "macro_auprc": _to_jsonable(_get(util_real, "macro_auprc")),
            "bal_acc": _to_jsonable(_get(util_real, "bal_acc")),
            "balanced_acc": _to_jsonable(_get(util_real, "balanced_acc")),
            "macro_precision": _to_jsonable(_get(util_real, "macro_precision")),
            "macro_recall": _to_jsonable(_get(util_real, "macro_recall")),
        },
        "utility_real_plus_synth": {
            "macro_f1": _to_jsonable(_get(util_rs, "macro_f1")),
            "macro_auprc": _to_jsonable(_get(util_rs, "macro_auprc")),
            "bal_acc": _to_jsonable(_get(util_rs, "bal_acc")),
            "balanced_acc": _to_jsonable(_get(util_rs, "balanced_acc")),
            "macro_precision": _to_jsonable(_get(util_rs, "macro_precision")),
            "macro_recall": _to_jsonable(_get(util_rs, "macro_recall")),
        },

        # Legacy shims (phase-1 compatible)
        "metrics.cfid":                _to_jsonable(_get(generative, "cfid_macro")),
        "metrics.cfid_macro":          _to_jsonable(_get(generative, "cfid_macro")),
        "metrics.fid_macro":           _to_jsonable(_get(generative, "fid_macro")),
        "metrics.kid":                 _to_jsonable(_get(generative, "kid")),
        "metrics.ms_ssim":             _to_jsonable(_get(generative, "ms_ssim")),
        # "metrics.downstream.macro_f1": _to_jsonable(_get(util_rs, "macro_f1")),

        "metrics.downstream.macro_f1": _to_jsonable(_get(util_rs, "macro_f1")),
        "metrics.downstream.macro_auprc": _to_jsonable(_get(util_rs, "macro_auprc")),
        "metrics.downstream.bal_acc": _to_jsonable(_get(util_rs, "bal_acc")),
        "metrics.downstream.balanced_acc": _to_jsonable(_get(util_rs, "balanced_acc")),
        "metrics.downstream.precision": _to_jsonable(_get(util_rs, "macro_precision")),
        "metrics.downstream.recall": _to_jsonable(_get(util_rs, "macro_recall")),
        "metrics.gen_precision": _to_jsonable(_get(util_rs, "macro_precision")),
        "metrics.gen_recall": _to_jsonable(_get(util_rs, "macro_recall")),

        # Counts (optional but helpful)
        "counts.num_real": _to_jsonable(_get(counts, "train_real")),
        "counts.num_fake": _to_jsonable(_get(counts, "synthetic")),
    }

    # -------------------------------------------------------------------------
    # NEW: attach provenance/audit metadata (if provided)
    # -------------------------------------------------------------------------
    _attach_audit_fields(rec, run_meta)

    # -------------------------------------------------------------------------
    # Optional per-class F1 fields (structured + legacy flat)
    # -------------------------------------------------------------------------
    if util_real_per_class:
        rec.setdefault("utility_real_only", {})["per_class_f1"] = {
            str(k): _to_jsonable(v) for k, v in util_real_per_class.items()
        }
        for k, v in util_real_per_class.items():
            rec[f"metrics.real_only.per_class_f1.{str(k)}"] = _to_jsonable(v)

    if util_rs_per_class:
        rec.setdefault("utility_real_plus_synth", {})["per_class_f1"] = {
            str(k): _to_jsonable(v) for k, v in util_rs_per_class.items()
        }
        for k, v in util_rs_per_class.items():
            rec[f"metrics.real_plus_synth.per_class_f1.{str(k)}"] = _to_jsonable(v)

    # -------------------------------------------------------------------------
    # Write: ensure parent exists and append a compact JSON line
    # -------------------------------------------------------------------------
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")

    print(f"[eval] appended summary → {out_path}")
