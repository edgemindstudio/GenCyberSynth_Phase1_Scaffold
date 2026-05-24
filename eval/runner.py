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

    # -----------------------------
    # Config ID (Paper3)
    # -----------------------------
    # Prefer any existing config_id, otherwise derive it from config_path.
    config_id = rm.get("config_id") or config.get("config_id")
    if not config_id and isinstance(cfg_path, str) and cfg_path:
        stem = Path(cfg_path).stem  # e.g., paper3_regime_aug_balanced_b500
        if stem.startswith("paper3_regime_"):
            config_id = "paper3_" + stem[len("paper3_regime_"):]  # -> paper3_aug_balanced_b500
        else:
            config_id = stem

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
        "config_id": config_id,
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
    config["config_id"] = rm2.get("config_id")

    return rm2


def _require_run_meta_ok(config: Dict[str, Any]) -> None:
    prov = config.get("provenance", {}) if isinstance(config.get("provenance"), dict) else {}
    if not prov.get("require", False):
        return
    rm = config.get("run_meta")
    if not isinstance(rm, dict):
        raise RuntimeError("provenance.require=true but run_meta missing.")
    # required = ["config_path", "config_sha1", "git_commit", "caps", "budget_per_class"]
    required = ["config_path", "config_sha1", "git_commit", "config_id", "caps", "budget_per_class"]
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
    We also resolve relative image paths relative to the directory containing manifest.json.
    """
    with open(manifest_path, "r") as f:
        man = json.load(f)

    # Normalize common field name / schema
    # Convert "samples" -> canonical "paths" list of {"path": ..., "label": int}
    if "paths" not in man and isinstance(man.get("samples"), list):
        man["paths"] = []
        for s in man["samples"]:
            if not isinstance(s, dict):
                continue

            p = (
                s.get("path")
                or s.get("filepath")
                or s.get("file")
                or s.get("filename")
                or s.get("img_path")
            )

            y = (
                s.get("label")
                if "label" in s
                else s.get("y")
                if "y" in s
                else s.get("label_id")
                if "label_id" in s
                else s.get("class_id")
                if "class_id" in s
                else None
            )

            if p is None or y is None:
                continue

            try:
                man["paths"].append({"path": str(p), "label": int(y)})
            except Exception:
                continue

    man.setdefault("paths", [])  # list of {"path": "...", "label": int}
    man.setdefault("per_class_counts", {})

    # Resolve relative image paths robustly.
    # Baseline manifests live at synthetic/<config_id>/seed<seed>/manifest.json.
    # Policy manifests may live deeper, e.g. synthetic/<config_id>/seed<seed>/policy/<policy_id>/manifest.json.
    # Manifest entries are relative to the synthetic root, so search upward until the joined path exists.
    base_dir = os.path.dirname(manifest_path)

    for item in man["paths"]:
        if not isinstance(item, dict):
            continue
        rel = item.get("path")
        if isinstance(rel, str) and rel and not os.path.isabs(rel):
            cur = base_dir
            resolved = None
            while True:
                cand = os.path.normpath(os.path.join(cur, rel))
                if os.path.exists(cand):
                    resolved = cand
                    break
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent
            item["path"] = resolved if resolved is not None else os.path.normpath(os.path.join(base_dir, rel))

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


# def _load_images_local(
#         manifest: Dict[str, Any],
#         per_class_cap: int = 200,
# ) -> Tuple["np.ndarray", "np.ndarray"]:

def _load_images_local(
        manifest: Dict[str, Any],
        per_class_cap: int = 200,
        target_hw: tuple[int, int] = (40, 40),
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

    # # In this project, images are 40x40. We force exact size for robustness.
    # target_hw = (40, 40)

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


def _class_counts_dict(y) -> Dict[str, int]:

    """Return class counts as a JSON-friendly {class_id: count} dict."""

    try:

        import numpy as np

        yy = y.argmax(axis=1) if getattr(y, "ndim", 0) == 2 else y

        vals, counts = np.unique(yy.astype("int64"), return_counts=True)

        return {str(int(v)): int(c) for v, c in zip(vals, counts)}

    except Exception:

        return {}





def _apply_real_train_subsample(x_train, y_train, config: Dict[str, Any]):

    """

    Apply Paper 3 real-training imbalance before downstream utility training.



    Supported config:

      real_train_subsample:

        enabled: true

        strategy: minority_fraction

        minority_classes: [4, 7]

        minority_fraction: 0.2

        seed: 42



    Only x_train/y_train are changed. Validation and test sets remain untouched.

    """

    try:

        import numpy as np



        sub = config.get("real_train_subsample")

        if not isinstance(sub, dict) or not bool(sub.get("enabled", False)):

            return x_train, y_train, None



        strategy = str(sub.get("strategy", "minority_fraction"))

        if strategy != "minority_fraction":

            raise ValueError(f"Unsupported real_train_subsample.strategy: {strategy}")



        y_int = y_train.argmax(axis=1) if getattr(y_train, "ndim", 0) == 2 else y_train

        y_int = y_int.astype("int64")



        minority_classes = [int(c) for c in sub.get("minority_classes", [])]

        if not minority_classes:

            raise ValueError("real_train_subsample.minority_classes is empty")



        minority_fraction = float(sub.get("minority_fraction", 1.0))

        if not (0.0 < minority_fraction <= 1.0):

            raise ValueError(f"minority_fraction must be in (0,1], got {minority_fraction}")



        seed = int(sub.get("seed", _cfg_get(config, "run_meta.seed", config.get("SEED", config.get("seed", 0)))))

        rng = np.random.default_rng(seed)



        before = _class_counts_dict(y_int)



        keep_parts = []

        for cls in sorted(set(int(c) for c in np.unique(y_int))):

            idx = np.where(y_int == cls)[0]

            if cls in minority_classes:

                n_keep = max(1, int(round(len(idx) * minority_fraction)))

                idx = rng.choice(idx, size=n_keep, replace=False)

            keep_parts.append(idx)



        keep_idx = np.concatenate(keep_parts)

        rng.shuffle(keep_idx)



        x_new = x_train[keep_idx]

        y_new = y_train[keep_idx]



        after = _class_counts_dict(y_new)



        meta = {

            "enabled": True,

            "strategy": strategy,

            "minority_classes": minority_classes,

            "minority_fraction": minority_fraction,

            "seed": seed,

            "before_counts": before,

            "after_counts": after,

            "num_train_before": int(len(y_train)),

            "num_train_after": int(len(y_new)),

        }



        print("[imbalance] real_train_subsample:", meta)

        return x_new, y_new, meta



    except Exception as e:

        raise RuntimeError(f"real_train_subsample failed: {type(e).__name__}: {e}") from e





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


# ---------------------------------------------------------------------------
# Manifest selection (shared + per-run)
# ---------------------------------------------------------------------------
def _infer_config_variant(config: dict) -> str | None:
    """
    Best-effort inference of config variant letter (A/B/...) from run_meta.

    Priority:
      1) config["run_meta"]["config_variant"]  (preferred)
      2) config["run_meta"]["config_id"]       (e.g., "gan_A" -> "A")
    """
    rm = config.get("run_meta")
    if not isinstance(rm, dict):
        return None

    v = rm.get("config_variant")
    if isinstance(v, str) and v:
        return v

    cid = rm.get("config_id")
    if isinstance(cid, str) and "_" in cid:
        maybe = cid.split("_")[-1]
        return maybe if maybe else None

    return None


def _per_run_manifest_path(synth_root: str, model_name: str, config: dict) -> str | None:
    """
    Seed/config-specific manifest path:

      <synth_root>/<model>_<CFG>_seed<SEED>/manifest.json

    where synth_root is:
      <artifacts>/<model>/synthetic
    """
    cfg_variant = _infer_config_variant(config)
    seed = config.get("SEED")

    if not isinstance(cfg_variant, str) or not cfg_variant:
        return None

    # YAML loads SEED as int normally, but allow "42" too.
    if not isinstance(seed, int):
        if isinstance(seed, str) and seed.isdigit():
            seed = int(seed)
        else:
            return None

    run_dir = os.path.join(synth_root, f"{model_name}_{cfg_variant}_seed{seed}")
    return os.path.join(run_dir, "manifest.json")


def _derive_config_id_from_config(config: dict) -> str | None:
    """
    Derive a stable config_id for Paper3/Paper4 runs.

    Preferred sources:
      1) config["run_meta"]["config_id"]
      2) basename of config["run_meta"]["config_path"], with:
           "paper3_regime_aug_balanced_b500.yaml" -> "paper3_aug_balanced_b500"
           "paper4_smoke.yaml" -> "paper4_smoke"
    """
    rm = config.get("run_meta") if isinstance(config, dict) else None
    rm = rm if isinstance(rm, dict) else {}

    # First preference: explicit config_id already set in YAML
    cfg_id = rm.get("config_id")
    if isinstance(cfg_id, str) and cfg_id.strip():
        return cfg_id.strip()

    # Fallback: derive from config_path if available
    cfg_path = rm.get("config_path")
    if isinstance(cfg_path, str) and cfg_path.strip():
        stem = os.path.splitext(os.path.basename(cfg_path))[0]
        # Normalize "paper3_regime_xxx" -> "paper3_xxx"
        # and same idea for other papers if needed.
        stem = stem.replace("_regime_", "_")
        return stem

    return None


def _manifest_path_config_seed(synth_root: str, config_id: str, seed: int) -> str:
    """
    New Paper3 layout:
      <synth_root>/<config_id>/seed<seed>/manifest.json
    """
    return os.path.join(synth_root, config_id, f"seed{seed}", "manifest.json")


def _seed_from_config(config: dict) -> int:
    """Match the seed logic used elsewhere in this file."""
    return int(
        _cfg_get(
            config,
            "synth.seed",
            _cfg_get(
                config,
                "train.seed",
                _cfg_get(config, "run_meta.seed", config.get("SEED", config.get("seed", 0))),
            ),
        )
    )


# def _select_manifest_path(synth_root: str, model_name: str, config: dict) -> str:
#     """
#     Prefer per-run manifest if it exists, else fall back to shared manifest.
#
#     Shared (backwards-compatible):
#       <synth_root>/manifest.json
#
#     Per-run (tuning-safe):
#       <synth_root>/<model>_<CFG>_seed<SEED>/manifest.json
#     """
#     shared = os.path.join(synth_root, "manifest.json")
#     per_run = _per_run_manifest_path(synth_root, model_name, config)
#
#     if per_run and os.path.exists(per_run):
#         return per_run
#
#     return shared


def _select_manifest_path(synth_root: str, model_name: str, config: dict) -> str:
    """
    Manifest resolution order (most specific -> least):
      1) Config + seed scoped:
           <synth_root>/<config_id>/seed<seed>/manifest.json
      2) Config scoped only:
           <synth_root>/<config_id>/manifest.json
      3) Legacy per-run layout:
           <synth_root>/<model>_<CFG>_seed<SEED>/manifest.json
      4) Legacy seed-only layout:
           <synth_root>/seed<seed>/manifest.json
      5) Shared latest pointer:
           <synth_root>/manifest.json
    """
    explicit = ((config.get("run_meta") or {}).get("manifest_path")) if isinstance(config, dict) else None
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(f"Explicit run_meta.manifest_path does not exist: {explicit}")

    shared = os.path.join(synth_root, "manifest.json")
    seed = _seed_from_config(config)
    config_id = _derive_config_id_from_config(config)

    # 1) New config/seed-scoped layout
    if config_id:
        p = os.path.join(synth_root, config_id, f"seed{seed}", "manifest.json")
        if os.path.exists(p):
            return p

        # 2) Config-scoped without explicit seed
        p2 = os.path.join(synth_root, config_id, "manifest.json")
        if os.path.exists(p2):
            return p2

    # 3) Legacy per-run layout
    per_run = _per_run_manifest_path(synth_root, model_name, config)
    if per_run and os.path.exists(per_run):
        return per_run

    # 4) Legacy seed-only layout
    seed_only = os.path.join(synth_root, f"seed{seed}", "manifest.json")
    if os.path.exists(seed_only):
        return seed_only

    # 5) Shared latest manifest
    return shared


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
            a = tf.convert_to_tensor(x[i: i + 1])  # (1,H,W,3)
            b = tf.convert_to_tensor(x[j: j + 1])  # (1,H,W,3)

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
    # man_path = os.path.join(synth_root, "manifest.json")
    man_path = _select_manifest_path(synth_root, model_name, config)
    have_synth = (not no_synth) and os.path.exists(man_path)

    imgs = None
    labels = None

    # ALWAYS use the normalized manifest for meta inference (even if metrics won't run)
    synth_manifest_meta = _manifest_for_meta(man_path) if os.path.exists(man_path) else None

    # Now run_meta will infer budget_per_class from per_class_counts reliably
    _ensure_run_meta(config, synth_manifest=synth_manifest_meta)

    # Record which manifest path this eval run actually used (audit-friendly)
    try:
        rm = config.get("run_meta")
        rm = rm if isinstance(rm, dict) else {}
        rm["manifest_path"] = man_path
        config["run_meta"] = rm
    except Exception:
        pass

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

            # imgs, labels = _load_images_local(man, per_class_cap=per_class_cap)

            img_shape_cfg = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
            target_hw = tuple(img_shape_cfg[:2])
            imgs, labels = _load_images_local(man, per_class_cap=per_class_cap, target_hw=target_hw)

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

        # # KID
        # kid_fn = getattr(val_common, "compute_kid", None) if val_common else None
        # metrics["kid"] = _safe_metric("kid", kid_fn, imgs, labels)

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

    # synth_manifest = _Path(synth_root) / "manifest.json"
    # Use the same manifest path selected above (per-run preferred, shared fallback)
    synth_manifest = Path(man_path)

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
    # stamp = _now_ts()
    # out_path = os.path.join(summaries_dir, f"summary_{stamp}.json")
    #
    # # seed_ = int(_cfg_get(config, "seed", 0))
    # seed_ = int(config.get("SEED", config.get("seed", 0)))

    stamp = _now_ts()
    out_path = os.path.join(summaries_dir, f"summary_{stamp}.json")

    # Seed (prefer explicit config fields used by train/synth; fall back to run_meta/legacy)
    seed_ = int(
        _cfg_get(config, "synth.seed",
                 _cfg_get(config, "train.seed",
                          _cfg_get(config, "run_meta.seed",
                                   config.get("SEED", config.get("seed", 0))
                                   )
                          )
                 )
    )

    # Persist config_id into run_meta + summary for Paper3 collectors
    cid = _derive_config_id_from_config(config)
    if cid:
        rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}
        rm["config_id"] = cid
        config["run_meta"] = rm

    # Persist seed into run_meta for collectors/tables
    rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}
    rm["seed"] = seed_
    config["run_meta"] = rm

    # ADDED BLOCK
    # Optional tuning-lite: include config_id in run_id so cfgA/cfgB are distinguishable
    cfg_id = rm.get("config_id") or rm.get("config_tag")
    bpc = rm.get("budget_per_class") or config.get("budget_per_class")
    if cfg_id and bpc:
        run_id = f"{model_name}_{cfg_id}_pc{int(bpc)}_s{seed_}"
    elif cfg_id:
        run_id = f"{model_name}_{cfg_id}_s{seed_}"
    else:
        run_id = f"{model_name}_s{seed_}"

    # Consolidate generative metrics into a single dict
    gen = {
        "fid": _gen_extra.get("fid"),
        "fid_macro": None,  # reserved if you later compute per-class then macro-average
        "cfid_macro": _gen_extra.get("cfid_macro", metrics.get("cfid")),
        "kid": metrics.get("kid"),
        "ms_ssim": metrics.get("ms_ssim"),
    }

    # # Utility blocks (placeholders for now)
    # util_real = {"macro_f1": None}
    # util_rs = {"macro_f1": None}

    # Downstream utility (REAL vs REAL+SYNTH)
    metrics["downstream"] = {
        "macro_f1": None,
        "macro_auprc": None,
        "bal_acc": None,
        "balanced_acc": None,
        "precision": None,
        "recall": None,
    }

    util_real = {
        "macro_f1": None,
        "macro_auprc": None,
        "bal_acc": None,
        "balanced_acc": None,
        "macro_precision": None,
        "macro_recall": None,
    }
    util_rs = {
        "macro_f1": None,
        "macro_auprc": None,
        "bal_acc": None,
        "balanced_acc": None,
        "macro_precision": None,
        "macro_recall": None,
    }

    delta_macro_f1 = None
    delta_macro_auprc = None
    delta_bal_acc = None

    # Load real train/val/test so downstream utility can be computed
    x_train_real = y_train_real = x_val_real = y_val_real = x_test_real = y_test_real = None
    imgs_for_util = None

    try:
        from common.data import load_dataset_npy

        data_dir = _cfg_get(
            config,
            "DATA_DIR",
            _cfg_get(config, "data.root", "/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware_nhwc"),
        )
        img_shape = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        num_classes = int(_cfg_get(config, "NUM_CLASSES", 9))
        val_fraction = float(_cfg_get(config, "VAL_FRACTION", 0.1))

        x_train_real, y_train_real, x_val_real, y_val_real, x_test_real, y_test_real = load_dataset_npy(
            data_dir,
            img_shape=img_shape,
            num_classes=num_classes,
            val_fraction=val_fraction,
        )


        # Paper 3 imbalance regimes: optionally reduce real training samples

        # before both real-only and real+synthetic downstream utility runs.

        x_train_real, y_train_real, _real_subsample_meta = _apply_real_train_subsample(

            x_train_real, y_train_real, config

        )

        if _real_subsample_meta:

            rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}

            rm["real_train_subsample"] = _real_subsample_meta

            config["run_meta"] = rm

    except Exception as e:
        metrics.setdefault("_warnings", []).append(
            f"Real-data load skipped for downstream utility: {type(e).__name__}: {e}"
        )

    if (
            val_common is not None
            and x_train_real is not None and y_train_real is not None
            and x_val_real is not None and y_val_real is not None
            and x_test_real is not None and y_test_real is not None
    ):

    # if (
    #         val_common is not None
    #         and x_train_real is not None and y_train_real is not None
    #         and x_val_real is not None and y_val_real is not None
    #         and x_test_real is not None and y_test_real is not None
    #         # and x_synth is not None and y_synth is not None
    #         # and len(x_synth) > 0
    #         and imgs is not None and labels is not None
    #         and len(imgs) > 0
    # ):

        try:
            utility_epochs = int(_cfg_get(config, "evaluator.utility_epochs", 10))

            # util_bundle = val_common.compute_all_metrics(
            #     img_shape=tuple(x_train_real.shape[1:]),
            #     x_train_real=x_train_real, y_train_real=y_train_real,
            #     x_val_real=x_val_real, y_val_real=y_val_real,
            #     x_test_real=x_test_real, y_test_real=y_test_real,
            #     x_synth=x_synth, y_synth=y_synth,
            #     per_class_cap=per_class_cap,
            #     seed=seed_,
            #     epochs=utility_epochs,
            #     compute_fid=False,
            #     compute_cfid=False,
            #     compute_similarity=False,
            #     compute_diversity=False,
            # )

            # imgs_for_util = None
            # labels_for_util = None
            # if have_synth and imgs is not None and labels is not None and len(imgs) > 0:
            #     imgs_for_util = imgs
            #     labels_for_util = labels

            # For downstream utility we MUST use the correct manifest (config+seed scoped),
            # and we should load synth from that manifest (not rely on any "latest" pointer).
            imgs_for_util = None
            labels_for_util = None

            if have_synth and os.path.exists(man_path):
                try:
                    # Ensure we load the manifest in a normalized way
                    man_util = _load_manifest_local(man_path)
                    # Load images/labels (use a very large cap so utility sees full budget)
                    # NOTE: _load_images_local already supports target_hw; reuse the same shape logic
                    img_shape_cfg = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
                    target_hw = tuple(img_shape_cfg[:2])

                    imgs_u, labels_u = _load_images_local(
                        man_util,
                        per_class_cap=10 ** 9,  # effectively uncapped
                        target_hw=target_hw
                    )

                    if imgs_u is not None and labels_u is not None and len(imgs_u) > 0:
                        imgs_for_util = imgs_u
                        labels_for_util = labels_u
                except Exception as _e:
                    print("[warn] failed to load synth for utility from manifest:", type(_e).__name__, _e)
                    imgs_for_util = None
                    labels_for_util = None

            if (
                    imgs_for_util is not None
                    and x_train_real is not None
                    and getattr(imgs_for_util, "ndim", 0) == 4
                    and imgs_for_util.shape[-1] == 3
                    and x_train_real.ndim == 4
                    and x_train_real.shape[-1] == 1
            ):
                imgs_for_util = imgs_for_util.mean(axis=-1, keepdims=True).astype("float32")

            # Convert one-hot real labels to integer class ids for downstream utility
            if y_train_real is not None and getattr(y_train_real, "ndim", 0) == 2:
                y_train_real = y_train_real.argmax(axis=1)
            if y_val_real is not None and getattr(y_val_real, "ndim", 0) == 2:
                y_val_real = y_val_real.argmax(axis=1)
            if y_test_real is not None and getattr(y_test_real, "ndim", 0) == 2:
                y_test_real = y_test_real.argmax(axis=1)

            # Be defensive about synth labels too
            if labels_for_util is not None and getattr(labels_for_util, "ndim", 0) == 2:
                labels_for_util = labels_for_util.argmax(axis=1)

            print("[debug] x_train_real:", None if x_train_real is None else (x_train_real.shape, x_train_real.dtype))
            print("[debug] y_train_real:", None if y_train_real is None else (y_train_real.shape, y_train_real.dtype))
            print("[debug] x_val_real:", None if x_val_real is None else (x_val_real.shape, x_val_real.dtype))
            print("[debug] y_val_real:", None if y_val_real is None else (y_val_real.shape, y_val_real.dtype))
            print("[debug] x_test_real:", None if x_test_real is None else (x_test_real.shape, x_test_real.dtype))
            print("[debug] y_test_real:", None if y_test_real is None else (y_test_real.shape, y_test_real.dtype))
            print("[debug] imgs_for_util:",
                  None if imgs_for_util is None else (imgs_for_util.shape, imgs_for_util.dtype))
            print("[debug] labels_for_util:",
                  None if labels_for_util is None else (labels_for_util.shape, labels_for_util.dtype))

            try:
                import numpy as np
                if y_train_real is not None:
                    print("[debug] y_train_real unique:", np.unique(y_train_real)[:20])
                if y_val_real is not None:
                    print("[debug] y_val_real unique:", np.unique(y_val_real)[:20])
                if y_test_real is not None:
                    print("[debug] y_test_real unique:", np.unique(y_test_real)[:20])
                if labels_for_util is not None:
                    print("[debug] labels_for_util unique:", np.unique(labels_for_util)[:20])
            except Exception as _e:
                print("[debug] unique-label inspection failed:", type(_e).__name__, _e)

            # util_bundle = val_common.compute_all_metrics(
            #     img_shape=tuple(x_train_real.shape[1:]),
            #     x_train_real=x_train_real, y_train_real=y_train_real,
            #     x_val_real=x_val_real, y_val_real=y_val_real,
            #     x_test_real=x_test_real, y_test_real=y_test_real,
            #     x_synth=imgs_for_util, y_synth=labels_for_util,
            #     fid_cap_per_class=per_class_cap,
            #     seed=seed_,
            #     epochs=utility_epochs,
            # )

            # --- Always compute REAL-ONLY utility (works even when no_synth=True) ---
            util_bundle_real = val_common.compute_all_metrics(
                img_shape=tuple(x_train_real.shape[1:]),
                x_train_real=x_train_real, y_train_real=y_train_real,
                x_val_real=x_val_real, y_val_real=y_val_real,
                x_test_real=x_test_real, y_test_real=y_test_real,
                x_synth=None, y_synth=None,
                fid_cap_per_class=per_class_cap,
                seed=seed_,
                epochs=utility_epochs,
            )

            # --- Compute REAL+SYNTH only if synth is available and non-empty ---
            util_bundle_rs = None
            if have_synth and imgs_for_util is not None and labels_for_util is not None and len(imgs_for_util) > 0:
                util_bundle_rs = val_common.compute_all_metrics(
                    img_shape=tuple(x_train_real.shape[1:]),
                    x_train_real=x_train_real, y_train_real=y_train_real,
                    x_val_real=x_val_real, y_val_real=y_val_real,
                    x_test_real=x_test_real, y_test_real=y_test_real,
                    x_synth=imgs_for_util, y_synth=labels_for_util,
                    fid_cap_per_class=per_class_cap,
                    seed=seed_,
                    epochs=utility_epochs,
                )

            # Merge into one util_bundle dict (so downstream code stays unchanged)
            util_bundle = util_bundle_real if isinstance(util_bundle_real, dict) else {}
            # Ensure real-only is always available under the expected key
            util_bundle["utility_real_only"] = util_bundle.get("utility_real_only") or util_bundle.get("real_only")
            if isinstance(util_bundle_rs, dict):
                # prefer RS keys from RS run, but keep real-only from real run
                util_bundle["utility_real_plus_synth"] = (
                        util_bundle_rs.get("utility_real_plus_synth") or util_bundle_rs.get("real_plus_synth")
                )
                util_bundle["real_plus_synth"] = util_bundle_rs.get("real_plus_synth")

            # -----------------------------
            # ADD: deltas = (real_plus_synth - real_only)
            # -----------------------------
            def _delta(a, b):
                if a is None or b is None:
                    return None
                try:
                    return float(b) - float(a)
                except Exception:
                    return None

            ro = util_bundle.get("real_only") or {}
            rps = util_bundle.get("real_plus_synth") or {}

            # Only compute deltas if real_plus_synth exists and has numbers
            if isinstance(rps, dict) and (rps.get("macro_f1") is not None or rps.get("accuracy") is not None):
                util_bundle["deltas_RS_minus_R"] = {
                    "delta_accuracy": _delta(ro.get("accuracy"), rps.get("accuracy")),
                    "delta_macro_f1": _delta(ro.get("macro_f1"), rps.get("macro_f1")),
                    "delta_bal_acc": _delta(ro.get("bal_acc"), rps.get("bal_acc")),
                    "delta_macro_auprc": _delta(ro.get("macro_auprc"), rps.get("macro_auprc")),
                    "delta_ece": _delta(ro.get("ece"), rps.get("ece")),
                    "delta_brier": _delta(ro.get("brier"), rps.get("brier")),
                    "delta_recall_at_1pct_fpr": _delta(ro.get("recall_at_1pct_fpr"), rps.get("recall_at_1pct_fpr")),
                }
            else:
                util_bundle["deltas_RS_minus_R"] = None

            print("[debug] util_bundle type:", type(util_bundle).__name__)
            if isinstance(util_bundle, dict):
                print("[debug] util_bundle keys:", sorted(util_bundle.keys()))
                print("[debug] utility_real_only:", util_bundle.get("utility_real_only"))
                print("[debug] utility_real_plus_synth:", util_bundle.get("utility_real_plus_synth"))
                print("[debug] real_only:", util_bundle.get("real_only"))
                print("[debug] real_plus_synth:", util_bundle.get("real_plus_synth"))
                print("[debug] deltas:", util_bundle.get("deltas") or util_bundle.get("deltas_RS_minus_R"))
            else:
                print("[debug] util_bundle repr:", repr(util_bundle))

            util_real = (
                    util_bundle.get("utility_real_only")
                    or util_bundle.get("real_only")
                    or util_real
            )
            util_rs = (
                    util_bundle.get("utility_real_plus_synth")
                    or util_bundle.get("real_plus_synth")
                    or util_rs
            )

            if util_real.get("balanced_acc") is None and util_real.get("bal_acc") is not None:
                util_real["balanced_acc"] = util_real["bal_acc"]
            if util_rs.get("balanced_acc") is None and util_rs.get("bal_acc") is not None:
                util_rs["balanced_acc"] = util_rs["bal_acc"]

            deltas = util_bundle.get("deltas") or util_bundle.get("deltas_RS_minus_R") or {}
            delta_macro_f1 = deltas.get("delta_macro_f1", deltas.get("macro_f1"))
            delta_macro_auprc = deltas.get("delta_macro_auprc", deltas.get("macro_auprc"))
            delta_bal_acc = deltas.get("delta_bal_acc", deltas.get("balanced_accuracy", deltas.get("bal_acc")))

            metrics["downstream"]["macro_f1"] = util_rs.get("macro_f1")
            metrics["downstream"]["macro_auprc"] = util_rs.get("macro_auprc")
            metrics["downstream"]["bal_acc"] = util_rs.get("bal_acc")
            metrics["downstream"]["balanced_acc"] = util_rs.get("balanced_acc")
            metrics["downstream"]["precision"] = util_rs.get("macro_precision")
            metrics["downstream"]["recall"] = util_rs.get("macro_recall")

            metrics["gen_precision"] = util_rs.get("macro_precision")
            metrics["gen_recall"] = util_rs.get("macro_recall")

        except Exception as e:
            metrics.setdefault("_warnings", []).append(
                f"Downstream utility skipped: {type(e).__name__}: {e}"
            )

    # KID: REAL val vs SYNTH
    metrics["kid"] = None

    try:
        if (
                val_common is not None
                and hasattr(val_common, "kid_keras")
                and x_val_real is not None
                and imgs_for_util is not None
                and len(x_val_real) > 1
                and len(imgs_for_util) > 1
        ):
            subset = min(200, len(x_val_real), len(imgs_for_util))
            if subset >= 2:
                metrics["kid"] = val_common.kid_keras(
                    x_val_real.astype("float32"),
                    imgs_for_util.astype("float32"),
                    subset=subset,
                    n_subsets=10,
                    seed=seed_,
                )
            else:
                metrics.setdefault("_warnings", []).append(
                    "KID skipped: fewer than 2 samples available after alignment."
                )
        else:
            metrics.setdefault("_warnings", []).append(
                "KID skipped: missing kid_keras, real val split, or synthetic images."
            )
    except Exception as e:
        metrics["kid"] = None
        metrics.setdefault("_warnings", []).append(
            f"KID compute failed: {type(e).__name__}: {e}"
        )

    gen["kid"] = metrics.get("kid")

    # Cast counts to plain ints if present
    counts_map = {
        "train_real": (int(counts["num_real"]) if counts.get("num_real") is not None else None),
        "synthetic": (int(counts["num_fake"]) if counts.get("num_fake") is not None else None),
    }

    # Pull config_id once (avoid duplicate keys)
    cid = (config.get("run_meta") or {}).get("config_id")

    # Compute deltas in a robust way:
    # 1) Prefer util_bundle’s deltas if present
    # 2) Otherwise compute deltas from util_real vs util_rs if both exist
    deltas_rs_minus_r = None
    if isinstance(util_bundle, dict):
        deltas_rs_minus_r = (
                util_bundle.get("deltas_RS_minus_R")
                or util_bundle.get("deltas")  # some paths store it here
        )

    def _delta(a, b):
        if a is None or b is None:
            return None
        try:
            return float(b) - float(a)
        except Exception:
            return None

    if deltas_rs_minus_r is None and isinstance(util_real, dict) and isinstance(util_rs, dict):
        deltas_rs_minus_r = {
            "delta_accuracy": _delta(util_real.get("accuracy"), util_rs.get("accuracy")),
            "delta_macro_f1": _delta(util_real.get("macro_f1"), util_rs.get("macro_f1")),
            "delta_bal_acc": _delta(util_real.get("bal_acc"), util_rs.get("bal_acc")),
            "delta_macro_auprc": _delta(util_real.get("macro_auprc"), util_rs.get("macro_auprc")),
            "delta_ece": _delta(util_real.get("ece"), util_rs.get("ece")),
            "delta_brier": _delta(util_real.get("brier"), util_rs.get("brier")),
            "delta_recall_at_1pct_fpr": _delta(util_real.get("recall_at_1pct_fpr"), util_rs.get("recall_at_1pct_fpr")),
        }

    rec: Dict[str, Any] = {
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model_name,
        "seed": seed_,

        # Fix A payload
        "config_id": cid,
        "deltas_RS_minus_R": deltas_rs_minus_r,

        "run_id": run_id,

        "generative": {
            "fid": gen.get("fid"),
            "fid_macro": gen.get("fid_macro"),
            "cfid_macro": gen.get("cfid_macro"),
            "kid": gen.get("kid"),
            "ms_ssim": gen.get("ms_ssim"),
        },
        "memorization": ({"nn_dist_mean": _mem_extra.get("nn_dist_mean")} if _mem_extra else {}),

        # keep safe even if util_real/util_rs are None
        # Keep full downstream utility blocks, including per-class reports.
        # This is needed for Paper 3 journal diagnostics on minority classes 4 and 7.
        "utility_real_only": (dict(util_real) if isinstance(util_real, dict) else {"macro_f1": None}),
        "utility_real_plus_synth": (dict(util_rs) if isinstance(util_rs, dict) else {"macro_f1": None}),

        # Legacy flattened shims expected by older aggregators/plots
        "metrics.cfid": gen.get("cfid_macro"),
        "metrics.cfid_macro": gen.get("cfid_macro"),
        "metrics.fid": gen.get("fid"),
        "metrics.fid_macro": gen.get("fid_macro"),
        "metrics.kid": gen.get("kid"),
        "metrics.ms_ssim": gen.get("ms_ssim"),
        "metrics.nn_dist_mean": (_mem_extra.get("nn_dist_mean") if _mem_extra else None),
        "metrics.downstream.macro_f1": (util_rs.get("macro_f1") if isinstance(util_rs, dict) else None),

        "counts.num_real": counts_map.get("train_real"),
        "counts.num_fake": counts_map.get("synthetic"),
    }

    # Helpful, greppable field (top-level)
    rec["manifest_path"] = man_path
    _attach_audit_fields(rec, config)

    rm = config.get("run_meta")
    rm = rm if isinstance(rm, dict) else {}
    rm["manifest_path"] = man_path
    config["run_meta"] = rm

    if metrics.get("_warnings"):
        rec["_warnings"] = list(metrics["_warnings"])

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
    except Exception as e:
        print(f"[eval] ERROR: could not read summary for patching: {type(e).__name__}: {e}")
        _cur = None  # keep going

    if _cur is not None:
        # 1) Always patch manifest_path (safe, no dependencies)
        # Preserve full downstream utility blocks, including per-class metrics.
        # write_phase2_summary may compact fields, so restore from in-memory util_real/util_rs.
        if isinstance(util_real, dict):
            _cur["utility_real_only"] = dict(util_real)
            _cur["real_only"] = dict(util_real)
        if isinstance(util_rs, dict):
            _cur["utility_real_plus_synth"] = dict(util_rs)
            _cur["real_plus_synth"] = dict(util_rs)

        _cur["manifest_path"] = man_path
        rm2 = _cur.get("run_meta")
        rm2 = rm2 if isinstance(rm2, dict) else {}
        rm2["manifest_path"] = man_path

        rm2["config_id"] = (config.get("run_meta") or {}).get("config_id")
        _cur["config_id"] = (config.get("run_meta") or {}).get("config_id")

        # --- persist deltas into saved JSON summary ---
        try:
            ro = _cur.get("real_only") or _cur.get("utility_real_only") or {}
            rps = _cur.get("real_plus_synth") or _cur.get("utility_real_plus_synth") or {}

            if isinstance(ro, dict) and isinstance(rps, dict) and (
                    rps.get("macro_f1") is not None or rps.get("accuracy") is not None):
                d = {
                    "delta_accuracy": _delta(ro.get("accuracy"), rps.get("accuracy")),
                    "delta_macro_f1": _delta(ro.get("macro_f1"), rps.get("macro_f1")),
                    "delta_bal_acc": _delta(ro.get("bal_acc"), rps.get("bal_acc")),
                    "delta_macro_auprc": _delta(ro.get("macro_auprc"), rps.get("macro_auprc")),
                    "delta_ece": _delta(ro.get("ece"), rps.get("ece")),
                    "delta_brier": _delta(ro.get("brier"), rps.get("brier")),
                    "delta_recall_at_1pct_fpr": _delta(ro.get("recall_at_1pct_fpr"), rps.get("recall_at_1pct_fpr")),
                }
            else:
                d = None

            _cur.setdefault("utility", {})
            _cur["utility"]["deltas_RS_minus_R"] = d
            _cur["deltas_RS_minus_R"] = d
        except Exception as e:
            print(f"[eval] WARNING: could not persist deltas: {type(e).__name__}: {e}")

        _cur["run_meta"] = rm2

        print(f"[eval] patched manifest_path into summary: {man_path}")

        # 2) Patch audit fields
        try:
            _attach_audit_fields(_cur, config)
        except Exception as e:
            print(f"[eval] WARNING: _attach_audit_fields failed: {type(e).__name__}: {e}")

        # 3) Merge extra computed metrics
        try:
            _cur.setdefault("generative", {}).update({k: v for k, v in _gen_extra.items() if v is not None})
            if _mem_extra:
                _cur.setdefault("memorization", {}).update(_mem_extra)

            _cur["metrics.fid"] = _cur.get("metrics.fid", _gen_extra.get("fid"))
            if "nn_dist_mean" in _mem_extra:
                _cur["metrics.nn_dist_mean"] = _mem_extra["nn_dist_mean"]

            if metrics.get("kid") is not None:
                _cur.setdefault("generative", {})["kid"] = metrics["kid"]
                _cur["metrics.kid"] = metrics["kid"]

        except Exception as e:
            print(f"[eval] WARNING: metric merge failed: {type(e).__name__}: {e}")

        # 4) Write patched file
        try:
            with open(out_path, "w") as fdst:
                # --- persist deltas into saved JSON summary (compute from _cur itself) ---
                def _delta(a, b):
                    if a is None or b is None:
                        return None
                    try:
                        return float(b) - float(a)
                    except Exception:
                        return None

                try:
                    ro = _cur.get("real_only") or _cur.get("utility_real_only") or {}
                    rps = _cur.get("real_plus_synth") or _cur.get("utility_real_plus_synth") or {}

                    if isinstance(ro, dict) and isinstance(rps, dict) and (
                            rps.get("macro_f1") is not None or rps.get("accuracy") is not None):
                        d = {
                            "delta_accuracy": _delta(ro.get("accuracy"), rps.get("accuracy")),
                            "delta_macro_f1": _delta(ro.get("macro_f1"), rps.get("macro_f1")),
                            "delta_bal_acc": _delta(ro.get("bal_acc"), rps.get("bal_acc")),
                            "delta_macro_auprc": _delta(ro.get("macro_auprc"), rps.get("macro_auprc")),
                            "delta_ece": _delta(ro.get("ece"), rps.get("ece")),
                            "delta_brier": _delta(ro.get("brier"), rps.get("brier")),
                            "delta_recall_at_1pct_fpr": _delta(ro.get("recall_at_1pct_fpr"),
                                                               rps.get("recall_at_1pct_fpr")),
                        }
                    else:
                        d = None

                    _cur.setdefault("utility", {})
                    _cur["utility"]["deltas_RS_minus_R"] = d
                    _cur["deltas_RS_minus_R"] = d
                except Exception as e:
                    print(f"[eval] WARNING: could not persist deltas: {type(e).__name__}: {e}")

                json.dump(_cur, fdst, indent=2)

        except Exception as e:
            print(f"[eval] ERROR: could not write patched summary: {type(e).__name__}: {e}")

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
