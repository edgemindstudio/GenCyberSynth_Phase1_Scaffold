# eval/runner.py

"""
Evaluation runner for the GenCyberSynth scaffold.

This module loads synthetic samples produced by a model adapter (via its
manifest), computes a small suite of quality/diversity metrics using
`gcs_core`, and writes a timestamped JSON summary under:

    {artifacts}/{model_name}/summaries/summary_YYYYMMDD_HHMMSS.json

Configuration (from `configs/config.yaml`)
------------------------------------------
evaluator:
  domain_encoder: "malware_encoder_v1"   # optional; if supported by gcs_core
  per_class_cap: 200                     # max samples per class for metric eval
paths:
  artifacts: "artifacts"                 # root output directory

Notes
-----
- This runner is defensive: each metric is computed in isolation; failures
  are caught and reported without aborting the whole evaluation.
- If `gcs_core` is not installed or lacks a metric, the corresponding value
  is set to `None` with an explanatory message in `metrics["_warnings"]`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Tuple

# --- Optional dependency: gcs_core -------------------------------------------
_WARNINGS: list[str] = []
try:
    from gcs_core import val_common, synth_loader  # type: ignore
except Exception as _e:  # pragma: no cover
    val_common = None  # type: ignore
    synth_loader = None  # type: ignore
    _WARNINGS.append(
        "gcs_core import failed; metrics will be skipped. "
        f"ImportError: {type(_e).__name__}: {_e}"
    )


# --- Small utilities ----------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """
    Fetch a nested config value by dotted path, e.g. "evaluator.per_class_cap".
    """
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_metric(name: str, fn, *args, **kwargs):
    """
    Call a metric function safely. On failure, record a warning and return None.
    """
    if fn is None:
        _WARNINGS.append(f"Metric '{name}' unavailable (function not found).")
        return None
    try:
        val = fn(*args, **kwargs)
        # Convert tensors/ndarrays to Python floats if needed
        try:
            return float(val)  # type: ignore[arg-type]
        except Exception:
            return val
    except Exception as e:  # pragma: no cover
        _WARNINGS.append(f"Metric '{name}' failed: {type(e).__name__}: {e}")
        return None


def _maybe_set_domain_encoder(encoder_name: str | None):
    """
    If gcs_core exposes a domain-encoder hook, set it. Otherwise no-op.
    """
    if not encoder_name or val_common is None:
        return
    setter = getattr(val_common, "set_domain_encoder", None)
    _safe_metric("set_domain_encoder", setter, encoder_name)


# --- Public API ---------------------------------------------------------------
def evaluate_model_suite(config: Dict[str, Any], model_name: str, no_synth: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single model family (folder) by loading its synthetic samples and
    computing image-quality/diversity metrics.

    Args:
        config: Parsed configuration mapping.
        model_name: Folder under `{artifacts}/` (e.g., "gan", "vae", "diffusion").
        no_synth: If True, skip loading manifest/samples and only write a stub summary.

    Returns:
        A dictionary payload written to summaries JSON and also returned to caller.
    """
    artifacts_root = _cfg_get(config, "paths.artifacts", "artifacts")
    model_root = os.path.join(artifacts_root, model_name)
    synth_root = os.path.join(model_root, "synthetic")
    summaries_dir = _ensure_dir(os.path.join(model_root, "summaries"))

    man_path = os.path.join(synth_root, "manifest.json")
    have_synth = (not no_synth) and os.path.exists(man_path)

    # Configurable evaluator parameters (with safe defaults)
    per_class_cap = int(_cfg_get(config, "evaluator.per_class_cap", 200))
    domain_encoder = _cfg_get(config, "evaluator.domain_encoder", None)

    metrics: Dict[str, Any] = {}
    # Retain any pre-existing import warnings
    if _WARNINGS:
        metrics["_warnings"] = list(_WARNINGS)

    if have_synth and synth_loader is not None:
        # Load manifest + capped images
        man = synth_loader.load_manifest(man_path)  # type: ignore[attr-defined]
        imgs, labels = synth_loader.load_images(   # type: ignore[attr-defined]
            man, per_class_cap=per_class_cap
        )

        # Optional: select domain encoder if supported
        _maybe_set_domain_encoder(domain_encoder)

        # --- Core metrics (defensive calls) ---
        # CFID (classifier-based FID) — some implementations can be encoder-aware
        cfid_fn = getattr(val_common, "compute_cfid", None) if val_common else None
        metrics["cfid"] = _safe_metric("cfid", cfid_fn, imgs, labels)

        # KID
        kid_fn = getattr(val_common, "compute_kid", None) if val_common else None
        metrics["kid"] = _safe_metric("kid", kid_fn, imgs, labels)

        # Generative precision/recall
        gpr_fn = getattr(val_common, "generative_precision_recall", None) if val_common else None
        gpr_val = _safe_metric("generative_precision_recall", gpr_fn, imgs, labels)
        if isinstance(gpr_val, Tuple) or (isinstance(gpr_val, (list, tuple)) and len(gpr_val) == 2):
            try:
                prec, rec = gpr_val
                metrics["gen_precision"] = float(prec)
                metrics["gen_recall"] = float(rec)
            except Exception:
                metrics["gen_precision"] = None
                metrics["gen_recall"] = None
        else:
            metrics["gen_precision"] = None
            metrics["gen_recall"] = None

        # MS-SSIM (intra-class)
        mss_fn = getattr(val_common, "ms_ssim_intra_class", None) if val_common else None
        metrics["ms_ssim"] = _safe_metric("ms_ssim_intra_class", mss_fn, imgs, labels)

    else:
        metrics["note"] = (
            "No synthetic images found (or --no-synth used); metrics skipped."
        )

    # Placeholder for downstream utility; wire your actual training/eval as needed.
    metrics["downstream"] = {
        "macro_f1": None,
        "macro_auprc": None,
        "balanced_acc": None,
    }

    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "config_used": True,
        "metrics": metrics,
    }

    out_path = os.path.join(summaries_dir, f"summary_{_now_ts()}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] Saved evaluation summary → {out_path}")

    # Maintain/refresh a convenient 'latest.json' copy (best-effort)
    try:  # pragma: no cover
        latest = os.path.join(summaries_dir, "latest.json")
        with open(latest, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    return payload


__all__ = ["evaluate_model_suite"]
