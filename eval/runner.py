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
- Defensive design: each metric is computed in isolation; failures are caught
  and surfaced in metrics["_warnings"] without aborting the run.
- If `gcs_core` is not installed or lacks a metric, corresponding values are
  set to None with an explanatory message.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Optional dependency: gcs_core -------------------------------------------
_WARNINGS: List[str] = []
try:
    from gcs_core import val_common, synth_loader  # type: ignore
except Exception as _e:  # pragma: no cover
    val_common = None  # type: ignore
    synth_loader = None  # type: ignore
    _WARNINGS.append(
        "gcs_core import failed; metrics will be skipped where unavailable. "
        f"ImportError: {type(_e).__name__}: {_e}"
    )


# --- Local fallbacks if gcs_core.* is missing --------------------------------
def _load_manifest_local(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, "r") as f:
        man = json.load(f)
    man.setdefault("paths", [])
    man.setdefault("per_class_counts", {})
    return man


def _read_image(path: Path) -> Optional["np.ndarray"]:
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        return (np.asarray(img).astype("float32") / 255.0)
    except Exception:
        return None


def _load_images_local(
    manifest: Dict[str, Any],
    per_class_cap: int = 200
) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    xs: List[np.ndarray] = []
    ys: List[int] = []
    counts: Dict[int, int] = {}
    for item in manifest.get("paths", []):
        y = int(item["label"])
        if counts.get(y, 0) >= per_class_cap:
            continue
        arr = _read_image(Path(item["path"]))
        if arr is None:
            continue
        xs.append(arr)
        ys.append(y)
        counts[y] = counts.get(y, 0) + 1
    if not xs:
        return np.zeros((0, 0, 0, 0), dtype="float32"), np.zeros((0,), dtype="int32")
    return np.stack(xs, axis=0).astype("float32"), np.asarray(ys, dtype="int32")


# --- Small utilities ----------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch a nested config value by dotted path, e.g. 'evaluator.per_class_cap'."""
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
        try:
            return float(val)  # convert tensors/ndarrays to Python float when possible
        except Exception:
            return val
    except Exception as e:  # pragma: no cover
        _WARNINGS.append(f"Metric '{name}' failed: {type(e).__name__}: {e}")
        return None


def _maybe_set_domain_encoder(encoder_name: Optional[str]):
    """If gcs_core exposes a domain-encoder hook, set it. Otherwise no-op."""
    if not encoder_name or val_common is None:
        return
    setter = getattr(val_common, "set_domain_encoder", None)
    _safe_metric("set_domain_encoder", setter, encoder_name)


# --- Public API ---------------------------------------------------------------
def evaluate_model_suite(
    config: Dict[str, Any],
    model_name: str,
    no_synth: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single model family (folder) by loading its synthetic samples and
    computing image-quality/diversity metrics.

    Args:
        config: Parsed configuration mapping.
        model_name: Folder under `{artifacts}/` (e.g., 'gan', 'vae', 'diffusion').
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
    if _WARNINGS:
        metrics["_warnings"] = list(_WARNINGS)

    if have_synth:
        # --- Manifest ---
        if synth_loader is not None and hasattr(synth_loader, "load_manifest"):
            man = synth_loader.load_manifest(man_path)  # type: ignore[attr-defined]
        else:
            metrics.setdefault("_warnings", []).append(
                "gcs_core.synth_loader.load_manifest not found; using local loader."
            )
            man = _load_manifest_local(man_path)

        # --- Images ---
        if synth_loader is not None and hasattr(synth_loader, "load_images"):
            imgs, labels = synth_loader.load_images(man, per_class_cap=per_class_cap)  # type: ignore[attr-defined]
        else:
            metrics.setdefault("_warnings", []).append(
                "gcs_core.synth_loader.load_images not found; using local loader."
            )
            imgs, labels = _load_images_local(man, per_class_cap=per_class_cap)

        # Optional domain-encoder selection
        _maybe_set_domain_encoder(domain_encoder)

        # If nothing loaded, note it explicitly
        if getattr(imgs, "size", 0) == 0:
            metrics.setdefault("_warnings", []).append(
                "No images loaded from manifest; metrics may be empty."
            )

        # --- Core metrics (defensive calls) ---
        cfid_fn = getattr(val_common, "compute_cfid", None) if val_common else None
        metrics["cfid"] = _safe_metric("cfid", cfid_fn, imgs, labels)

        kid_fn = getattr(val_common, "compute_kid", None) if val_common else None
        metrics["kid"] = _safe_metric("kid", kid_fn, imgs, labels)

        gpr_fn = getattr(val_common, "generative_precision_recall", None) if val_common else None
        gpr_val = _safe_metric("generative_precision_recall", gpr_fn, imgs, labels)
        if isinstance(gpr_val, (tuple, list)) and len(gpr_val) == 2:
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

        mss_fn = getattr(val_common, "ms_ssim_intra_class", None) if val_common else None
        metrics["ms_ssim"] = _safe_metric("ms_ssim_intra_class", mss_fn, imgs, labels)

    else:
        metrics["note"] = "No synthetic images found (or --no-synth used); metrics skipped."

    # Placeholder for downstream utility; wire in your classifier when ready.
    metrics["downstream"] = {"macro_f1": None, "macro_auprc": None, "balanced_acc": None}

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

    # Best-effort “latest.json”
    try:  # pragma: no cover
        with open(os.path.join(summaries_dir, "latest.json"), "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    return payload


__all__ = ["evaluate_model_suite"]
