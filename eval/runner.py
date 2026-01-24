# eval/runner.py
"""
GenCyberSynth – Evaluation Runner
================================

This module evaluates a single model family by:
  1) Loading a synthetic manifest (produced by `app.main synth` via an adapter)
  2) Loading synthetic images (via gcs_core if available, otherwise via local fallbacks)
  3) Computing quality/diversity metrics (KID, cFID, MS-SSIM, etc.)
  4) Writing a timestamped JSON summary:
        {artifacts}/{model}/summaries/summary_YYYYMMDD_HHMMSS.json
     and also updating:
        {artifacts}/{model}/summaries/latest.json

It is designed to be **robust**:
  - If optional dependencies are missing (gcs_core, TensorFlow, etc.), it gracefully
    skips those metrics and records warnings.
  - It can load images locally (PIL + numpy) when gcs_core loaders are unavailable.

Audit / Reproducibility (VERY IMPORTANT)
---------------------------------------
The CLI (app/main.py) injects a provenance block into `config`:

    config["run_meta"] = {
        "config_path": "...",
        "config_sha1": "...",
        "git_commit": "...",
        "caps": {...},
        "budget_per_class": ...
    }

This file persists those audit fields into BOTH:
  - summary_*.json
  - latest.json

so future aggregation (Phase-1/Phase-2 tables) can prove exactly what ran.

Config (configs/config.yaml)
----------------------------
evaluator:
  domain_encoder: "malware_encoder_v1"   # optional; used if supported by gcs_core
  per_class_cap: 200                     # max synthetic samples per class to load
  save_nn_stats: true                    # compute memorization proxy (NN distances), if possible
  compute_fid: true
  compute_cfid: false
  fid_split: "val"                       # or "test"
paths:
  artifacts: "artifacts"                 # output root (model folders live under here)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import subprocess

from scripts.eval_write_summary import write_phase2_summary

# -----------------------------------------------------------------------------
# Optional dependency: gcs_core
# -----------------------------------------------------------------------------
# We try to use gcs_core if it exists; otherwise, we fall back to local loaders
# and skip metrics that require gcs_core functionality.
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


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def _sha1_file(p: str | None) -> str | None:
    if not p:
        return None
    try:
        b = Path(p).read_bytes()
        return hashlib.sha1(b).hexdigest()
    except Exception:
        return None


def _infer_budget_per_class(config: Dict[str, Any], synth_manifest: Dict[str, Any] | None) -> int | None:
    # Prefer manifest truth (what was actually generated)
    if isinstance(synth_manifest, dict):
        # If adapters wrote budget_per_class explicitly, prefer it
        bpc = synth_manifest.get("budget_per_class")
        if bpc is not None:
            try:
                return int(bpc)
            except Exception:
                pass

        pcc = synth_manifest.get("per_class_counts")
        if isinstance(pcc, dict) and len(pcc) > 0:
            vals = [int(v) for v in pcc.values() if v is not None]
            if vals:
                # Common choice: minimum per-class count (conservative and stable)
                return min(vals)

    # Fallback: what config asked for
    synth = config.get("synth", {}) if isinstance(config.get("synth"), dict) else {}
    n = synth.get("n_per_class")
    return int(n) if n is not None else None


def _ensure_run_meta(
    config: Dict[str, Any],
    synth_manifest: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    rm = config.get("run_meta")
    rm = rm if isinstance(rm, dict) else {}

    # Preserve anything already set (e.g., num_real, job_id, etc.)
    rm2 = dict(rm)

    # allow passing config path via run_meta OR top-level
    cfg_path = rm.get("config_path") or config.get("config_path")
    cfg_sha1 = rm.get("config_sha1") or config.get("config_sha1") or _sha1_file(cfg_path)
    commit = rm.get("git_commit") or config.get("git_commit") or _git_commit()

    evaluator = config.get("evaluator", {}) if isinstance(config.get("evaluator"), dict) else {}
    per_class_cap = int(evaluator.get("per_class_cap", 200))

    caps = rm.get("caps")
    caps = caps if isinstance(caps, dict) else {}
    caps.setdefault("per_class_cap", per_class_cap)

    # -----------------------------
    # Budget: manifest is truth
    # -----------------------------
    budget_pc = None

    # 1) Prefer manifest-derived budget (ignore zeros)
    if isinstance(synth_manifest, dict):
        budget_pc = synth_manifest.get("budget_per_class")
        if budget_pc is None:
            pcc = synth_manifest.get("per_class_counts", {})
            vals = []
            if isinstance(pcc, dict):
                for v in pcc.values():
                    try:
                        iv = int(v)
                        if iv > 0:
                            vals.append(iv)
                    except Exception:
                        pass
            budget_pc = min(vals) if vals else None

    # 2) If still unknown, fall back to existing rm/config heuristic
    if budget_pc is None:
        budget_pc = rm2.get("budget_per_class")
    if budget_pc is None:
        budget_pc = _infer_budget_per_class(config, synth_manifest)

    # Store final answer (key fix)
    rm2["budget_per_class"] = budget_pc

    # Guarantee required provenance fields (without deleting other rm2 keys)
    rm2.update({
        "config_path": cfg_path,
        "config_sha1": cfg_sha1,
        "git_commit": commit,
        "caps": caps,
        "budget_per_class": budget_pc,
    })

    config["run_meta"] = rm2

    # mirror top-level for convenience
    config["config_path"] = rm2.get("config_path")
    config["config_sha1"] = rm2.get("config_sha1")
    config["git_commit"] = rm2.get("git_commit")
    config["caps"] = rm2.get("caps")
    config["budget_per_class"] = rm2.get("budget_per_class")

    return rm2


def _require_run_meta_ok(config: Dict[str, Any]) -> None:
    prov = config.get("provenance", {}) if isinstance(config.get("provenance"), dict) else {}
    if not prov.get("require", False):
        return
    rm = config.get("run_meta")
    if not isinstance(rm, dict):
        raise RuntimeError("provenance.require=true but run_meta missing.")
    required = ["config_path", "config_sha1", "git_commit", "caps", "budget_per_class"]
    missing = [k for k in required if rm.get(k) is None]
    if missing:
        raise RuntimeError(f"provenance.require=true but missing run_meta fields: {missing}")

# -----------------------------------------------------------------------------
# Local fallbacks if gcs_core.* is missing
# -----------------------------------------------------------------------------
def _load_manifest_local(manifest_path: str) -> Dict[str, Any]:
    """
    Load a manifest JSON with minimal schema normalization.

    Expected-ish manifest schema:
      {
        "paths": [{"path": "...", "label": <int>}, ...],
        "per_class_counts": {...}
      }

    Some adapters may emit:
      {"samples": [{"path": "...", "label": ...}, ...]}

    We normalize "samples" -> "paths" so the rest of the code can be uniform.
    """
    with open(manifest_path, "r") as f:
        man = json.load(f)

    # Normalize common field name
    if "paths" not in man and isinstance(man.get("samples"), list):
        man["paths"] = man["samples"]

    man.setdefault("paths", [])  # list of {"path": "...", "label": int}
    man.setdefault("per_class_counts", {})
    return man


def _read_image(
    path: Path,
    *,
    min_hw: int = 11,
    target_hw: tuple[int, int] | None = None,
) -> Optional["np.ndarray"]:
    """
    Minimal image reader -> float32 HWC in [0,1].

    Why min_hw=11?
      - TensorFlow MS-SSIM/SSIM uses a default 11x11 window; TF asserts if
        H or W is < 11.

    target_hw:
      - If provided, resize exactly to that shape (e.g., (40, 40)).
      - In GenCyberSynth, malware images are 40x40, so we standardize to (40,40).

    Returns:
      - numpy array shape (H, W, 3) float32 in [0,1], or None on failure.
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


def _load_images_local(
    manifest: Dict[str, Any],
    per_class_cap: int = 200,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Load up to `per_class_cap` images per class from a manifest using local IO.

    Returns:
      imgs   : numpy array (N, H, W, 3) float32 in [0,1]
      labels : numpy array (N,) int32
    """
    import numpy as np

    xs: List[np.ndarray] = []
    ys: List[int] = []
    counts: Dict[int, int] = {}

    # In this project, images are 40x40. We force exact size for robustness.
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
        return (
            np.zeros((0, 0, 0, 0), dtype="float32"),
            np.zeros((0,), dtype="int32"),
        )

    return np.stack(xs, axis=0).astype("float32"), np.asarray(ys, dtype="int32")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """
    Fetch a nested config value by dotted path, e.g. 'evaluator.per_class_cap'.

    Example:
      per_class_cap = _cfg_get(config, "evaluator.per_class_cap", 200)
    """
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_dir(path: str) -> str:
    """Create a directory if it does not exist, then return the same path."""
    os.makedirs(path, exist_ok=True)
    return path


def _now_ts() -> str:
    """Timestamp used in filenames: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_metric(name: str, fn, *args, **kwargs):
    """
    Safely call a metric function:
      - If fn is None -> warn and return None
      - If fn raises -> record warning and return None
      - If returned value is convertible to float -> return float(value)

    This prevents a single broken metric from breaking the entire evaluation.
    """
    if fn is None:
        _WARNINGS.append(f"Metric '{name}' unavailable (function not found).")
        return None

    try:
        val = fn(*args, **kwargs)
        try:
            return float(val)
        except Exception:
            return val
    except Exception as e:  # pragma: no cover
        _WARNINGS.append(f"Metric '{name}' failed: {type(e).__name__}: {e}")
        return None


def _maybe_set_domain_encoder(encoder_name: Optional[str]):
    """
    If gcs_core exposes a domain-encoder hook, set it. Otherwise no-op.

    Some metrics depend on which feature extractor/encoder is active.
    """
    if not encoder_name or val_common is None:
        return
    setter = getattr(val_common, "set_domain_encoder", None)
    _safe_metric("set_domain_encoder", setter, encoder_name)


def _manifest_for_meta(man_path: str) -> Dict[str, Any] | None:
    try:
        man = _load_manifest_local(man_path)  # normalizes samples->paths, sets defaults
    except Exception:
        return None

    # If per_class_counts missing/empty, derive it from paths
    try:
        pcc = man.get("per_class_counts")
        if not isinstance(pcc, dict) or len(pcc) == 0:
            counts: Dict[str, int] = {}
            for item in man.get("paths", []) or []:
                y = item.get("label", None)
                if y is None:
                    continue
                y = str(int(y))
                counts[y] = counts.get(y, 0) + 1
            man["per_class_counts"] = counts
    except Exception:
        pass

    # Ensure stable derived fields exist for metadata inference
    try:
        paths_list = man.get("paths", [])
        man["num_fake"] = int(len(paths_list)) if isinstance(paths_list, list) else None

        pcc = man.get("per_class_counts")
        if isinstance(pcc, dict) and len(pcc) > 0:
            vals = [int(v) for v in pcc.values() if v is not None]
            man["budget_per_class"] = min(vals) if vals else None
        else:
            man["budget_per_class"] = None
    except Exception:
        pass

    return man

# -----------------------------------------------------------------------------
# Robust local MS-SSIM with SSIM fallback
# -----------------------------------------------------------------------------
def _ms_ssim_intra_class_local(imgs, labels, max_pairs_per_class: int = 200) -> float | None:
    """
    Robust intra-class diversity proxy.

    We measure *similarity* (MS-SSIM) between pairs of generated samples within
    each class. Interpretation:
      - Higher MS-SSIM => more similar images => lower diversity
      - Lower MS-SSIM  => more diverse images

    Steps:
      - Ensure NHWC float32 in [0,1]
      - Convert 1-channel -> 3 channels (TF SSIM expects 3 channels nicely)
      - Ensure H,W >= 11 (TF's SSIM window)
      - Try MS-SSIM; if it fails, fall back to SSIM
      - Aggregate across up to `max_pairs_per_class` random pairs per class

    Returns:
      Mean similarity in [0,1] or None if no class has >= 2 samples.
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

    # (N,H,W) -> (N,H,W,1)
    if x.ndim == 3:
        x = x[..., None]

    # Tile 1-channel -> 3-channel
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    # Ensure H,W >= 11
    H, W = int(x.shape[1]), int(x.shape[2])
    if min(H, W) < 11:
        new_h = max(11, H)
        new_w = max(11, W)
        x = tf.image.resize(tf.convert_to_tensor(x), [new_h, new_w], method="nearest").numpy()
        H, W = new_h, new_w

    # Choose a safe odd filter size <= min(H,W)
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

        # Upper bound of unique pairs: n*(n-1)/2
        n_pairs = min(max_pairs_per_class, len(idx) * (len(idx) - 1) // 2)

        for _ in range(n_pairs):
            i, j = rng.choice(idx, size=2, replace=False)
            a = tf.convert_to_tensor(x[i : i + 1])  # (1,H,W,3)
            b = tf.convert_to_tensor(x[j : j + 1])  # (1,H,W,3)

            v: float | None = None

            # Try MS-SSIM first
            try:
                v_tf = tf.image.ssim_multiscale(a, b, max_val=1.0, filter_size=fs)
                v = float(tf.reduce_mean(v_tf).numpy())
            except Exception:
                # Fall back to SSIM if MS-SSIM fails
                try:
                    v_tf = tf.image.ssim(a, b, max_val=1.0, filter_size=fs)
                    v = float(tf.reduce_mean(v_tf).numpy())
                except Exception:
                    v = None

            if v is not None and np.isfinite(v):
                vals.append(float(v))

    return float(np.mean(vals)) if vals else None


# -----------------------------------------------------------------------------
# Audit / Reproducibility helpers
# -----------------------------------------------------------------------------
def _attach_audit_fields(summary: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    try:
        rm = cfg.get("run_meta") if isinstance(cfg, dict) else None
        if not isinstance(rm, dict) or not rm:
            return

        if not isinstance(summary.get("run_meta"), dict):
            summary["run_meta"] = {}
        summary["run_meta"].update(rm)

        summary["config_path"] = rm.get("config_path")
        summary["config_sha1"] = rm.get("config_sha1")
        summary["git_commit"] = rm.get("git_commit")
        summary["caps"] = rm.get("caps")
        
        # Budget should reflect what was actually generated (manifest-derived)
        bpc = rm.get("budget_per_class")
        if bpc is None:
            bpc = cfg.get("budget_per_class")
        
        summary["budget_per_class"] = bpc
        summary["run_meta"]["budget_per_class"] = bpc


    except Exception:
        return


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def evaluate_model_suite(
    config: Dict[str, Any],
    model_name: str,
    no_synth: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single model family.

    Args:
      config:
        Loaded YAML config dict (optionally containing config["run_meta"]).
      model_name:
        The model folder under artifacts (e.g., "gan", "diffusion", "vae").
      no_synth:
        If True, skip any metrics requiring synthetic data.

    Returns:
      A dict "rec" (record) containing key metrics + counts + legacy fields.
      NOTE: The authoritative JSON written to disk is produced by
            write_phase2_summary(...) and then augmented below.
    """
    # ---------------------------------------------------------------------
    # Resolve filesystem locations
    # ---------------------------------------------------------------------
    artifacts_root = _cfg_get(config, "paths.artifacts", "artifacts")
    model_root = os.path.join(artifacts_root, model_name)
    synth_root = os.path.join(model_root, "synthetic")
    summaries_dir = _ensure_dir(os.path.join(model_root, "summaries"))

    # We expect synthesis to have produced:
    #   {artifacts}/{model}/synthetic/manifest.json
    man_path = os.path.join(synth_root, "manifest.json")
    have_synth = (not no_synth) and os.path.exists(man_path)
    
    # ALWAYS use the normalized manifest for meta inference (even if metrics won't run)
    synth_manifest_meta = _manifest_for_meta(man_path) if os.path.exists(man_path) else None
    
    # Now run_meta will infer budget_per_class from per_class_counts reliably
    _ensure_run_meta(config, synth_manifest=synth_manifest_meta)
    _require_run_meta_ok(config)


    # Evaluator params
    per_class_cap = int(_cfg_get(config, "evaluator.per_class_cap", 200))
    domain_encoder = _cfg_get(config, "evaluator.domain_encoder", None)

    # `metrics` is a working dict used for intermediate values and warnings.
    metrics: Dict[str, Any] = {}
    if _WARNINGS:
        metrics["_warnings"] = list(_WARNINGS)

    # ---------------------------------------------------------------------
    # 1) Load manifest and synthetic images
    # ---------------------------------------------------------------------
    if have_synth:
        # ---- Manifest ----
        if synth_loader is not None and hasattr(synth_loader, "load_manifest"):
            man = synth_loader.load_manifest(man_path)  # type: ignore[attr-defined]
        else:
            metrics.setdefault("_warnings", []).append(
                "gcs_core.synth_loader.load_manifest not found; using local loader."
            )
            man = _load_manifest_local(man_path)

        # ---- Images ----
        if synth_loader is not None and hasattr(synth_loader, "load_images"):
            imgs, labels = synth_loader.load_images(man, per_class_cap=per_class_cap)  # type: ignore[attr-defined]
        else:
            metrics.setdefault("_warnings", []).append(
                "gcs_core.synth_loader.load_images not found; using local loader."
            )
            imgs, labels = _load_images_local(man, per_class_cap=per_class_cap)

        # Normalize shapes/dtypes (belt & suspenders)
        try:
            import tensorflow as tf

            if getattr(imgs, "size", 0) > 0:
                # Ensure channel dimension exists
                if imgs.ndim == 3:
                    imgs = imgs[..., None]
                imgs = imgs.astype("float32", copy=False)

                # Normalize [0,255] -> [0,1] if needed
                if imgs.max() > 1.5:
                    imgs /= 255.0

                # Ensure H,W >= 11 for TF SSIM windows
                H, W = int(imgs.shape[1]), int(imgs.shape[2])
                if min(H, W) < 11:
                    imgs = tf.image.resize(
                        tf.convert_to_tensor(imgs),
                        [max(11, H), max(11, W)],
                        method="nearest",
                    ).numpy()
        except Exception:
            pass

        # Optional domain encoder selection (no-op if unsupported)
        _maybe_set_domain_encoder(domain_encoder)

        if getattr(imgs, "size", 0) == 0:
            metrics.setdefault("_warnings", []).append(
                "No images loaded from manifest; metrics may be empty."
            )

        # -----------------------------------------------------------------
        # 2) Core metrics (defensive calls)
        # -----------------------------------------------------------------
        # cFID
        cfid_fn = getattr(val_common, "compute_cfid", None) if val_common else None
        metrics["cfid"] = _safe_metric("cfid", cfid_fn, imgs, labels)

        # KID
        kid_fn = getattr(val_common, "compute_kid", None) if val_common else None
        metrics["kid"] = _safe_metric("kid", kid_fn, imgs, labels)

        # Generative Precision/Recall (if exposed by gcs_core)
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

        # MS-SSIM (robust local implementation)
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

    # ---------------------------------------------------------------------
    # 3) Optional: FID / CFID macro / NN-distance (memorization proxy)
    #
    # NOTE: This section depends on helper utilities and real split arrays
    # (X_val/y_val or X_test/y_test) being in scope elsewhere. If not, it
    # gracefully skips.
    # ---------------------------------------------------------------------
    from pathlib import Path as _Path

    synth_manifest = _Path(synth_root) / "manifest.json"
    eval_cfg: Dict[str, Any] = _cfg_get(config, "evaluator", {}) or {}

    want_fid: bool = bool(eval_cfg.get("compute_fid", True))
    want_cfid: bool = bool(eval_cfg.get("compute_cfid", False))
    want_nn: bool = bool(eval_cfg.get("save_nn_stats", True))
    fid_split = (eval_cfg.get("fid_split") or "val").lower()  # "val" or "test"

    # Try to find REAL arrays if the outer pipeline defines them
    real_X = real_y = None
    try:
        if fid_split == "test":
            real_X, real_y = X_test, y_test  # noqa: F821
        else:
            real_X, real_y = X_val, y_val  # noqa: F821
    except NameError:
        real_X = real_y = None

    # Optional helper imports (may not exist in some environments)
    try:
        from common.metrics.fid import compute_fid_features, fid_from_features  # type: ignore
    except Exception:
        compute_fid_features = fid_from_features = None
    try:
        from common.metrics.fid import compute_cfid  # type: ignore
    except Exception:
        compute_cfid = None
    try:
        from common.metrics.neighbors import compute_nn_dists  # type: ignore
    except Exception:
        compute_nn_dists = None
    try:
        from common.io import load_synth_images  # type: ignore
    except Exception:
        load_synth_images = None

    _gen_extra: Dict[str, Any] = {}
    _mem_extra: Dict[str, Any] = {}

    if have_synth and synth_manifest.exists() and real_X is not None:
        # FID
        if want_fid and compute_fid_features and fid_from_features and load_synth_images:
            try:
                feats_real = compute_fid_features(real_X, config)
                feats_synth = compute_fid_features(load_synth_images(synth_manifest), config)
                _gen_extra["fid"] = float(fid_from_features(feats_real, feats_synth))
            except Exception as e:
                metrics.setdefault("_warnings", []).append(f"FID skipped: {type(e).__name__}: {e}")

        # CFID (macro + per class)
        if want_cfid and compute_cfid and load_synth_images and real_y is not None:
            try:
                cfid_macro, cfid_per_class = compute_cfid((real_X, real_y), synth_manifest, config)
                _gen_extra["cfid_macro"] = float(cfid_macro)
                _gen_extra["cfid_per_class"] = [float(x) for x in (cfid_per_class or [])]
            except Exception as e:
                metrics.setdefault("_warnings", []).append(f"CFID skipped: {type(e).__name__}: {e}")

        # Nearest-neighbor distances (memorization proxy)
        if want_nn and compute_nn_dists and load_synth_images:
            try:
                dists = compute_nn_dists(load_synth_images(synth_manifest), real_X, config)
                if dists:
                    _mem_extra["nn_dist_mean"] = float(sum(dists) / len(dists))
            except Exception as e:
                metrics.setdefault("_warnings", []).append(f"NN distances skipped: {type(e).__name__}: {e}")

    # Placeholder: downstream utility metrics
    # (You will wire in real classifier evaluation later.)
    metrics["downstream"] = {"macro_f1": None, "macro_auprc": None, "balanced_acc": None}

    # ---------------------------------------------------------------------
    # 4) Counts: num_real / num_fake (best-effort)
    # ---------------------------------------------------------------------
    counts: Dict[str, Optional[int]] = {"num_real": None, "num_fake": None}

    # Real count: prefer run_meta.num_real, else fallback to npy heuristic
    try:
        rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}
        if rm.get("num_real") is not None:
            counts["num_real"] = int(rm.get("num_real"))
        else:
            import numpy as np

            # DATA_DIR can be set in config or env; otherwise default to USTC folder name
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

            counts["num_real"] = (real_total if real_total > 0 else None)
    except Exception:
        counts["num_real"] = None

    # Synthetic count: count entries in manifest ("paths" or "samples")
    try:
        rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}
        if rm.get("num_fake") is not None:
            counts["num_fake"] = int(rm.get("num_fake"))
            
        # elif have_synth and os.path.exists(man_path):
            # with open(man_path, "r") as f:
                # man_json = json.load(f)
                
        elif have_synth and os.path.exists(man_path):
            man_json = _manifest_for_meta(man_path) or _load_manifest_local(man_path)
            paths = man_json.get("paths", []) if isinstance(man_json, dict) else []
            counts["num_fake"] = len(paths) if isinstance(paths, list) else None


            if isinstance(man_json, dict):
                if isinstance(man_json.get("paths"), list):
                    counts["num_fake"] = len(man_json["paths"])
                elif isinstance(man_json.get("samples"), list):
                    counts["num_fake"] = len(man_json["samples"])
    except Exception:
        counts["num_fake"] = None

    # ---------------------------------------------------------------------
    # 5) Assemble summary record and write to disk
    # ---------------------------------------------------------------------
    stamp = _now_ts()
    out_path = os.path.join(summaries_dir, f"summary_{stamp}.json")

    # seed_ = int(_cfg_get(config, "seed", 0))
    seed_ = int(config.get("SEED", config.get("seed", 0)))

    # Consolidate generative metrics into a single dict
    gen = {
        "fid": _gen_extra.get("fid"),
        "fid_macro": None,  # reserved if you later compute per-class then macro-average
        "cfid_macro": _gen_extra.get("cfid_macro", metrics.get("cfid")),
        "kid": metrics.get("kid"),
        "ms_ssim": metrics.get("ms_ssim"),
    }

    # Utility blocks (placeholders for now)
    util_real = {"macro_f1": None}
    util_rs = {"macro_f1": None}

    # Cast counts to plain ints if present
    counts_map = {
        "train_real": (int(counts["num_real"]) if counts.get("num_real") is not None else None),
        "synthetic": (int(counts["num_fake"]) if counts.get("num_fake") is not None else None),
    }

    # This record is returned to the caller. The canonical JSON written to disk
    # is produced by `write_phase2_summary` and then augmented below.
    rec: Dict[str, Any] = {
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model_name,
        "seed": seed_,
        "run_id": f"{model_name}_{seed_}",
        "generative": {
            "fid": gen["fid"],
            "fid_macro": gen["fid_macro"],
            "cfid_macro": gen["cfid_macro"],
            "kid": gen["kid"],
            "ms_ssim": gen["ms_ssim"],
        },
        "memorization": ({"nn_dist_mean": _mem_extra.get("nn_dist_mean")} if _mem_extra else {}),
        "utility_real_only": {"macro_f1": util_real["macro_f1"]},
        "utility_real_plus_synth": {"macro_f1": util_rs["macro_f1"]},
        # Legacy “flattened” shims expected by older aggregators/plots
        "metrics.cfid": gen["cfid_macro"],
        "metrics.cfid_macro": gen["cfid_macro"],
        "metrics.fid": gen["fid"],
        "metrics.fid_macro": gen["fid_macro"],
        "metrics.kid": gen["kid"],
        "metrics.ms_ssim": gen["ms_ssim"],
        "metrics.nn_dist_mean": _mem_extra.get("nn_dist_mean"),
        "metrics.downstream.macro_f1": util_rs["macro_f1"],
        "counts.num_real": counts_map["train_real"],
        "counts.num_fake": counts_map["synthetic"],
    }

    # Add audit fields to the returned record as well (helpful for logging/tests)
    _attach_audit_fields(rec, config)

    # --- Primary write: phase2 summary writer --------------------------------
    # This function produces a clean, consistent JSON summary format used by your
    # downstream aggregation tools. We then patch/augment it below.
    write_phase2_summary(
        out_json=out_path,
        model=model_name,
        seed=seed_,
        generative=gen,
        util_real=util_real,
        util_rs=util_rs,
        counts=counts_map,
        run_id=rec["run_id"],
        run_meta=config.get("run_meta", {}),
    )
    print(f"[eval] Saved evaluation summary → {out_path}")

    # --- Post-write augmentation ---------------------------------------------
    # We re-open the written JSON and merge in:
    #   - audit fields (config path/hash/git commit/caps/budget)
    #   - any extra metrics computed after the initial record build (FID, NN stats)
    #
    # This guarantees that what aggregators read from disk is fully populated.
    try:
        with open(out_path, "r") as fsrc:
            _cur = json.load(fsrc)

        # Persist audit metadata into the file that is actually written to disk
        _attach_audit_fields(_cur, config)

        # Merge extra computed metrics into the nested generative/memorization blocks
        _cur.setdefault("generative", {}).update({k: v for k, v in _gen_extra.items() if v is not None})

        if _mem_extra:
            _cur.setdefault("memorization", {}).update(_mem_extra)

        # Maintain flattened keys for older tools
        _cur["metrics.fid"] = _cur.get("metrics.fid", _gen_extra.get("fid"))
        if "nn_dist_mean" in _mem_extra:
            _cur["metrics.nn_dist_mean"] = _mem_extra["nn_dist_mean"]

        with open(out_path, "w") as fdst:
            json.dump(_cur, fdst, indent=2)
    except Exception:
        # Never fail evaluation because summary patching failed
        pass

    # --- latest.json ----------------------------------------------------------
    # A human-friendly "most recent summary" copy. Many quick scripts read this.
    try:
        with open(out_path, "r") as fsrc:
            latest = json.load(fsrc)
        with open(os.path.join(summaries_dir, "latest.json"), "w") as fdst:
            json.dump(latest, fdst, indent=2)
    except Exception:
        pass

    return rec


__all__ = ["evaluate_model_suite"]


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def main():
    import argparse
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--overrides", default=None)
    p.add_argument("--model", required=True)  # keep it explicit to avoid guessing wrong
    p.add_argument("--no-synth", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    cfg = cfg if isinstance(cfg, dict) else {}
    cfg["config_path"] = args.config  # helps provenance

    if args.overrides:
        ov = yaml.safe_load(open(args.overrides, "r"))
        ov = ov if isinstance(ov, dict) else {}
        _deep_update(cfg, ov)

    rec = evaluate_model_suite(cfg, model_name=args.model, no_synth=args.no_synth)
    print("[runner] done:", rec.get("run_id"))


if __name__ == "__main__":
    main()

