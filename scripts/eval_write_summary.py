# scripts/eval_write_summary.py
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def _to_jsonable(v: Any) -> Any:
    """Best-effort conversion of numbers (incl. NumPy scalars) to JSON-safe types."""
    try:
        if hasattr(v, "item"):
            v = v.item()  # e.g., np.float32(1.2) -> 1.2
    except Exception:
        pass
    if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _get(d: Optional[Mapping[str, Any]], key: str) -> Any:
    return None if d is None else d.get(key)


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
    # Optional per-class F1 dicts like {"0": 0.71, "1": 0.63, ...}
    util_real_per_class: Mapping[str, Any] | None = None,
    util_rs_per_class: Mapping[str, Any] | None = None,
) -> None:
    """
    Append a flattened, plot-friendly JSON line with both legacy and new keys.

    - Accepts dict-like inputs (plain dicts or Mapping).
    - Cleans NaN/Inf/NumPy scalars into JSON-safe values.
    - Keeps legacy shim keys for backward compatibility with older plots.
    """
    rec: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": str(model),
        "seed": int(seed),
        "run_id": run_id,
        # Preferred, structured blocks (phase-2)
        "generative": {
            "fid_macro":  _to_jsonable(_get(generative, "fid_macro")),
            "cfid_macro": _to_jsonable(_get(generative, "cfid_macro")),
            "kid":        _to_jsonable(_get(generative, "kid")),
            "ms_ssim":    _to_jsonable(_get(generative, "ms_ssim")),
        },
        "utility_real_only": {
            "macro_f1": _to_jsonable(_get(util_real, "macro_f1")),
        },
        "utility_real_plus_synth": {
            "macro_f1": _to_jsonable(_get(util_rs, "macro_f1")),
        },
        # Legacy shims for phase-1 style plots
        "metrics.cfid":                _to_jsonable(_get(generative, "cfid_macro")),
        "metrics.cfid_macro":          _to_jsonable(_get(generative, "cfid_macro")),
        "metrics.fid_macro":           _to_jsonable(_get(generative, "fid_macro")),
        "metrics.kid":                 _to_jsonable(_get(generative, "kid")),
        "metrics.ms_ssim":             _to_jsonable(_get(generative, "ms_ssim")),
        "metrics.downstream.macro_f1": _to_jsonable(_get(util_rs, "macro_f1")),
        # Counts (optional but helpful for sanity checks)
        "counts.num_real": _to_jsonable(_get(counts, "train_real")),
        "counts.num_fake": _to_jsonable(_get(counts, "synthetic")),
    }

    # --- Optional: include per-class F1 (structured + legacy flat keys) ---
    if util_real_per_class:
        # structured
        rec.setdefault("utility_real_only", {})["per_class_f1"] = {
            str(k): _to_jsonable(v) for k, v in util_real_per_class.items()
        }
        # legacy flat
        for k, v in util_real_per_class.items():
            rec[f"metrics.real_only.per_class_f1.{str(k)}"] = _to_jsonable(v)

    if util_rs_per_class:
        # structured
        rec.setdefault("utility_real_plus_synth", {})["per_class_f1"] = {
            str(k): _to_jsonable(v) for k, v in util_rs_per_class.items()
        }
        # legacy flat
        for k, v in util_rs_per_class.items():
            rec[f"metrics.real_plus_synth.per_class_f1.{str(k)}"] = _to_jsonable(v)

    # Ensure parent exists and append compact JSON line
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")

    # Optional: small console breadcrumb
    print(f"[eval] appended summary → {out_path}")
