# eval/runner.py
"""
Evaluation runner for the GenCyberSynth scaffold.

Loads synthetic samples via a model's manifest, computes quality/diversity
metrics using `gcs_core` when available (with robust local fallbacks), and
writes a timestamped JSON summary under:

    {artifacts}/{model_name}/summaries/summary_YYYYMMDD_HHMMSS.json

Also writes:
    {artifacts}/{model_name}/summaries/latest.json

Config (configs/config.yaml)
----------------------------
evaluator:
  domain_encoder: "malware_encoder_v1"   # optional; used if supported by gcs_core
  per_class_cap: 200                     # max samples per class
paths:
  artifacts: "artifacts"                 # root output directory
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from scripts.eval_write_summary import write_phase2_summary

# -----------------------------------------------------------------------------
# Optional dependency: gcs_core
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Local fallbacks if gcs_core.* is missing
# -----------------------------------------------------------------------------
def _load_manifest_local(manifest_path: str) -> Dict[str, Any]:
    """Load a manifest JSON with minimal schema normalization."""
    with open(manifest_path, "r") as f:
        man = json.load(f)
    # Normalize common fields referenced below
    if "paths" not in man and isinstance(man.get("samples"), list):
        # Some adapters emit {"samples":[{"path":..., "label":...}, ...]}
        man["paths"] = man["samples"]
    man.setdefault("paths", [])  # list of {"path": "...", "label": int}
    man.setdefault("per_class_counts", {})
    return man


def _read_image(path: Path, *, min_hw: int = 11, target_hw: tuple[int, int] | None = None) -> Optional["np.ndarray"]:
    """
    Minimal image reader -> float32 HWC in [0,1].
    Ensures each spatial dim >= `min_hw` (default 11) so MS-SSIM won't assert.
    If `target_hw` provided, resizes to exactly that (e.g., (40,40)).
    """
    try:
        from PIL import Image  # type: ignore
        import numpy as np
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if target_hw is not None:
            img = img.resize(target_hw, Image.NEAREST)
        elif min(w, h) < min_hw:
            img = img.resize((max(min_hw, w), max(min_hw, h)), Image.NEAREST)
        arr = np.asarray(img).astype("float32") / 255.0
        return arr
    except Exception:
        return None


def _load_images_local(manifest: Dict[str, Any], per_class_cap: int = 200) -> Tuple["np.ndarray", "np.ndarray"]:
    """Load up to `per_class_cap` images per class from manifest using local IO."""
    import numpy as np
    xs: List[np.ndarray] = []
    ys: List[int] = []
    counts: Dict[int, int] = {}
    # Force to at least 11×11 (and in this project, samples are 40×40, so hit that).
    target_hw = (40, 40)

    for item in manifest.get("paths", []):
        try:
            y = int(item["label"])
        except Exception:
            continue
        if counts.get(y, 0) >= per_class_cap:
            continue
        arr = _read_image(Path(item["path"]), target_hw=target_hw)
        if arr is None:
            continue
        xs.append(arr)
        ys.append(y)
        counts[y] = counts.get(y, 0) + 1

    if not xs:
        return np.zeros((0, 0, 0, 0), dtype="float32"), np.zeros((0,), dtype="int32")
    return np.stack(xs, axis=0).astype("float32"), np.asarray(ys, dtype="int32")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
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


# --- Robust local MS-SSIM with SSIM fallback ---------------------------------
def _ms_ssim_intra_class_local(imgs, labels, max_pairs_per_class: int = 200) -> float | None:
    """
    Robust intra-class diversity proxy.

    Steps:
      - Ensure NHWC float32 in [0,1]
      - Tile 1-channel -> 3 channels
      - Upscale to at least 11x11 (nearest) so TF windows are valid
      - Try MS-SSIM; if it fails for a pair, fall back to SSIM
      - Aggregate across up to `max_pairs_per_class` random pairs per class
    Returns mean similarity in [0,1] (lower => more diverse), or None if no class has >=2 samples.
    """
    try:
        import numpy as np
        import tensorflow as tf
    except Exception:
        return None

    if imgs is None or getattr(imgs, "size", 0) == 0 or labels is None:
        return None

    x = imgs.astype("float32", copy=False)
    if x.max() > 1.5:
        x = x / 255.0
    if x.ndim == 3:  # (N,H,W) -> (N,H,W,1)
        x = x[..., None]
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    # Guarantee H,W >= 11
    H, W = int(x.shape[1]), int(x.shape[2])
    if min(H, W) < 11:
        new_h = max(11, H)
        new_w = max(11, W)
        x = tf.image.resize(tf.convert_to_tensor(x), [new_h, new_w], method="nearest").numpy()
        H, W = new_h, new_w

    # Choose a safe odd filter size (<= min(H,W))
    fs = min(11, H, W)
    if fs % 2 == 0:
        fs -= 1
    fs = max(fs, 3)

    y = np.asarray(labels).astype("int32", copy=False)

    vals: list[float] = []
    rng = np.random.default_rng(42)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < 2:
            continue
        # up to K random unique pairs
        n_pairs = min(max_pairs_per_class, len(idx) * (len(idx) - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng.choice(idx, size=2, replace=False)
            a = tf.convert_to_tensor(x[i:i+1])  # (1,H,W,3)
            b = tf.convert_to_tensor(x[j:j+1])  # (1,H,W,3)
            v = None
            # Try MS-SSIM first
            try:
                v = tf.image.ssim_multiscale(a, b, max_val=1.0, filter_size=fs)  # shape (1,)
                v = float(tf.reduce_mean(v).numpy())
            except Exception:
                # Fallback to plain SSIM
                try:
                    v = tf.image.ssim(a, b, max_val=1.0, filter_size=fs)  # shape (1,)
                    v = float(tf.reduce_mean(v).numpy())
                except Exception:
                    v = None
            if v is not None and np.isfinite(v):
                vals.append(float(v))

    return float(np.mean(vals)) if vals else None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def evaluate_model_suite(
    config: Dict[str, Any],
    model_name: str,
    no_synth: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single model family (folder) by loading its synthetic samples and
    computing image-quality/diversity metrics.
    """
    # Resolve paths
    artifacts_root = _cfg_get(config, "paths.artifacts", "artifacts")
    model_root = os.path.join(artifacts_root, model_name)
    synth_root = os.path.join(model_root, "synthetic")
    summaries_dir = _ensure_dir(os.path.join(model_root, "summaries"))

    man_path = os.path.join(synth_root, "manifest.json")
    have_synth = (not no_synth) and os.path.exists(man_path)

    # Evaluator parameters
    per_class_cap = int(_cfg_get(config, "evaluator.per_class_cap", 200))
    domain_encoder = _cfg_get(config, "evaluator.domain_encoder", None)

    metrics: Dict[str, Any] = {}
    if _WARNINGS:
        metrics["_warnings"] = list(_WARNINGS)

    # ---------------------------------------------------------------------
    # Load manifest & images
    # ---------------------------------------------------------------------
    if have_synth:
        # Manifest
        if synth_loader is not None and hasattr(synth_loader, "load_manifest"):
            man = synth_loader.load_manifest(man_path)  # type: ignore[attr-defined]
        else:
            metrics.setdefault("_warnings", []).append(
                "gcs_core.synth_loader.load_manifest not found; using local loader."
            )
            man = _load_manifest_local(man_path)

        # Images
        if synth_loader is not None and hasattr(synth_loader, "load_images"):
            imgs, labels = synth_loader.load_images(man, per_class_cap=per_class_cap)  # type: ignore[attr-defined]
        else:
            metrics.setdefault("_warnings", []).append(
                "gcs_core.synth_loader.load_images not found; using local loader."
            )
            imgs, labels = _load_images_local(man, per_class_cap=per_class_cap)

        # --- Normalize shapes for metrics (belt & suspenders) --------------
        try:
            import numpy as np
            import tensorflow as tf
            if getattr(imgs, "size", 0) > 0:
                if imgs.ndim == 3:  # (N,H,W) -> (N,H,W,1)
                    imgs = imgs[..., None]
                imgs = imgs.astype("float32", copy=False)
                if imgs.max() > 1.5:
                    imgs /= 255.0
                H, W = int(imgs.shape[1]), int(imgs.shape[2])
                if min(H, W) < 11:
                    imgs = tf.image.resize(tf.convert_to_tensor(imgs),
                                           [max(11, H), max(11, W)],
                                           method="nearest").numpy()
        except Exception:
            pass

        # Optional domain-encoder selection (no-op if unsupported)
        _maybe_set_domain_encoder(domain_encoder)

        # If nothing loaded, note it explicitly
        if getattr(imgs, "size", 0) == 0:
            metrics.setdefault("_warnings", []).append(
                "No images loaded from manifest; metrics may be empty."
            )

        # -----------------------------------------------------------------
        # Core metrics (defensive calls)
        # -----------------------------------------------------------------
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

        # --- MS-SSIM (force robust local implementation) -------------------
        try:
            mss_val = _ms_ssim_intra_class_local(imgs, labels, max_pairs_per_class=200)
            if mss_val is not None:
                metrics.setdefault("_warnings", []).append(
                    "MS-SSIM computed via local fallback (robust)."
                )
            else:
                metrics.setdefault("_warnings", []).append(
                    "Local MS-SSIM returned no value (insufficient pairs per class?)."
                )
        except Exception as e:
            mss_val = None
            metrics.setdefault("_warnings", []).append(
                f"Local MS-SSIM failed: {type(e).__name__}: {e}"
            )
        metrics["ms_ssim"] = mss_val

    else:
        metrics["note"] = "No synthetic images found (or --no-synth used); metrics skipped."

    # Placeholder for downstream utility; wire in your classifier when ready.
    metrics["downstream"] = {"macro_f1": None, "macro_auprc": None, "balanced_acc": None}

    # ---------------------------------------------------------------------
    # Counts: num_real / num_fake (best-effort)
    # ---------------------------------------------------------------------
    counts: Dict[str, Optional[int]] = {"num_real": None, "num_fake": None}
    try:
        import numpy as np
        data_root = _cfg_get(config, "DATA_DIR", _cfg_get(config, "data.root", "USTC-TFC2016_malware"))
        data_dir = Path(data_root)
        real_total = 0
        for fname in ("train_data.npy", "test_data.npy"):
            fpath = data_dir / fname
            if fpath.exists():
                try:
                    real_total += int(np.load(fpath, allow_pickle=False).shape[0])
                except Exception:
                    pass
        counts["num_real"] = real_total
    except Exception:
        counts["num_real"] = None

    try:
        if have_synth and os.path.exists(man_path):
            with open(man_path, "r") as f:
                man_json = json.load(f)
            if isinstance(man_json, dict):
                if isinstance(man_json.get("paths"), list):
                    counts["num_fake"] = len(man_json["paths"])
                elif isinstance(man_json.get("samples"), list):
                    counts["num_fake"] = len(man_json["samples"])
    except Exception:
        counts["num_fake"] = None


    # ---------------------------------------------------------------------
    # Assemble plot-friendly summary & write files (using helper)
    # ---------------------------------------------------------------------
    stamp = _now_ts()
    out_path = os.path.join(summaries_dir, f"summary_{stamp}.json")

    seed_ = int(_cfg_get(config, "seed", 0))

    gen = {
        "fid_macro": None,
        "cfid_macro": metrics.get("cfid"),
        "kid": metrics.get("kid"),
        "ms_ssim": metrics.get("ms_ssim"),
    }
    util_real = {"macro_f1": None}
    util_rs = {"macro_f1": None}

    # Cast counts to plain ints if present
    counts_map = {
        "train_real": (int(counts["num_real"]) if counts.get("num_real") is not None else None),
        "synthetic": (int(counts["num_fake"]) if counts.get("num_fake") is not None else None),
    }

    # Build the exact record we intend to write (so we can return it even if read fails)
    rec = {
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model_name,
        "seed": seed_,
        "run_id": f"{model_name}_{seed_}",
        "generative": {
            "fid_macro": gen["fid_macro"],
            "cfid_macro": gen["cfid_macro"],
            "kid": gen["kid"],
            "ms_ssim": gen["ms_ssim"],
        },
        "utility_real_only": {"macro_f1": util_real["macro_f1"]},
        "utility_real_plus_synth": {"macro_f1": util_rs["macro_f1"]},
        # legacy shims that plots/JSONL expect
        "metrics.cfid": gen["cfid_macro"],
        "metrics.cfid_macro": gen["cfid_macro"],
        "metrics.fid_macro": gen["fid_macro"],
        "metrics.kid": gen["kid"],
        "metrics.ms_ssim": gen["ms_ssim"],
        "metrics.downstream.macro_f1": util_rs["macro_f1"],
        "counts.num_real": counts_map["train_real"],
        "counts.num_fake": counts_map["synthetic"],
    }

    # Write one compact line to the per-model summary file
    write_phase2_summary(
        out_json=out_path,
        model=model_name,
        seed=seed_,
        generative=gen,
        util_real=util_real,
        util_rs=util_rs,
        counts=counts_map,
        run_id=rec["run_id"],
        util_real_per_class=per_class_real,
        util_rs_per_class=per_class_rs,
    )
    print(f"[eval] Saved evaluation summary → {out_path}")

    # Also write a human-readable latest.json (fallback to `rec` if read fails)
    try:
        with open(out_path, "r") as fsrc:
            latest = json.loads(fsrc.read())
    except Exception:
        latest = rec  # safe fallback

    try:
        with open(os.path.join(summaries_dir, "latest.json"), "w") as fdst:
            json.dump(latest, fdst, indent=2)
    except Exception:
        pass

    return latest


__all__ = ["evaluate_model_suite"]
