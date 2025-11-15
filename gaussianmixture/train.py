# gaussianmixture/train.py
"""
Trainer for the GaussianMixture baseline (per-class GMMs).

What this script does
---------------------
- Loads config (dict or --config path).
- Loads dataset: images in [0,1], channels-last (H,W,C); labels int or one-hot.
- Trains one GMM per class (components capped by class count); optional global fallback.
- Saves checkpoints to:
    <paths.artifacts>/gaussianmixture/checkpoints/
      GMM_class_{k}.joblib
      GMM_global_fallback.joblib  (optional)
- Writes a small preview grid sampled from saved checkpoints.

Conventions
-----------
- Uses scikit-learn GaussianMixture (EM). CPU-only, no GPUs required.
- Images flattened for GMM fitting; sampling reshapes back to (H,W,C).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import joblib
from sklearn.mixture import GaussianMixture

# Prefer shared loader; fallback to raw .npy below
try:
    from common.data import load_dataset_npy  # type: ignore
except Exception:
    load_dataset_npy = None  # fallback path used if import fails

# Local helpers from gaussianmixture package
from gaussianmixture.models import (
    reshape_to_images,
    flatten_images,
    create_gmm,  # convenience factory; falls back to sklearn if needed
)
from gaussianmixture.sample import save_grid_from_dir


# =============================================================================
# Small utilities
# =============================================================================
def _to_int_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Accept (N,) ints or (N,K) one-hot -> return (N,) ints."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return np.argmax(y, axis=1).astype(int)
    return y.astype(int)


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: Dict, dotted: str, default=None):
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _normalize_artifacts(cfg: Dict) -> Dict:
    """
    Normalize artifact paths, honoring `paths.artifacts` if present,
    with sane defaults under artifacts/gaussianmixture/* .
    """
    arts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    cfg.setdefault("ARTIFACTS", {})
    A = cfg["ARTIFACTS"]
    A.setdefault("gaussianmixture_checkpoints", str(arts_root / "gaussianmixture" / "checkpoints"))
    A.setdefault("gaussianmixture_synthetic",   str(arts_root / "gaussianmixture" / "synthetic"))
    A.setdefault("gaussianmixture_summaries",   str(arts_root / "gaussianmixture" / "summaries"))
    return cfg


def _default_artifact_paths(cfg: Dict) -> Tuple[Path, Path, Path]:
    A = cfg["ARTIFACTS"]
    ckpt = Path(A["gaussianmixture_checkpoints"])
    synth = Path(A["gaussianmixture_synthetic"])
    sums = Path(A["gaussianmixture_summaries"])
    return ckpt, synth, sums


# =============================================================================
# Core training
# =============================================================================
def train_per_class_gmms(
    *,
    x_train: np.ndarray,          # (N,H,W,C) in [0,1]
    y_train: np.ndarray,          # (N,) ints or (N,K) one-hot
    img_shape: Tuple[int, int, int],
    num_classes: int,
    ckpt_dir: Path,
    n_components: int = 10,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    random_state: int = 42,
    train_global_fallback: bool = True,
    verbose: bool = True,
) -> None:
    """
    Fit one GaussianMixture per class and save to ckpt_dir.

    - Caps components per class to <= #samples for that class.
    - If a class has < 2 samples, we skip and rely on the global fallback.
    """
    H, W, C = img_shape
    D = H * W * C

    _ensure_dirs(ckpt_dir)

    y_int = _to_int_labels(y_train, num_classes)
    X_flat = flatten_images(x_train) if x_train.ndim == 4 else x_train.reshape((-1, D))

    # Optional: global fallback GMM trained on all data
    if train_global_fallback:
        if verbose:
            print(f"[GMM] Training global fallback on {X_flat.shape[0]} samples, D={D}...")
        fallback = create_gmm(
            n_components=min(n_components, max(1, X_flat.shape[0])),
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
            max_iter=max_iter,
            verbose=1 if verbose else 0,
        )
        fallback.fit(X_flat)
        joblib.dump(fallback, ckpt_dir / "GMM_global_fallback.joblib")

    # Per-class models
    for k in range(num_classes):
        idx = (y_int == k)
        count = int(idx.sum())
        if verbose:
            print(f"[GMM] Class {k}: {count} samples")

        if count < 2:
            print(f"[warn] Class {k} too small for GMM (n={count}); will use fallback at sampling.")
            continue

        Xk = X_flat[idx]
        k_components = min(n_components, count)  # cap components to available samples

        gmm = create_gmm(
            n_components=k_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
            max_iter=max_iter,
            verbose=1 if verbose else 0,
        )
        gmm.fit(Xk)
        joblib.dump(gmm, ckpt_dir / f"GMM_class_{k}.joblib")

    if verbose:
        print(f"[GMM] Saved checkpoints to {ckpt_dir}")


# =============================================================================
# Orchestration
# =============================================================================
def run_train(cfg: Dict) -> None:
    """
    High-level runner:
      - sets defaults & normalizes artifact paths
      - loads dataset
      - trains per-class GMMs (+ optional global fallback)
      - writes a preview grid PNG
    """
    # -------- Defaults --------
    cfg = dict(cfg)  # shallow copy
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", 0.5)
    cfg.setdefault("GMM_COMPONENTS", 10)
    cfg.setdefault("GMM_COVARIANCE", "full")
    cfg.setdefault("GMM_REG_COVAR", 1e-6)
    cfg.setdefault("GMM_MAX_ITER", 200)
    cfg.setdefault("GMM_TRAIN_GLOBAL", True)
    _normalize_artifacts(cfg)

    seed = int(cfg["SEED"])
    img_shape = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    num_classes = int(cfg.get("NUM_CLASSES", cfg.get("num_classes", 9)))
    data_dir = Path(cfg.get("DATA_DIR", _cfg_get(cfg, "data.root", "data")))

    # GMM knobs
    n_components = int(cfg["GMM_COMPONENTS"])
    covariance_type = cfg["GMM_COVARIANCE"]
    reg_covar = float(cfg["GMM_REG_COVAR"])
    max_iter = int(cfg["GMM_MAX_ITER"])
    train_global_fallback = bool(cfg["GMM_TRAIN_GLOBAL"])

    ckpt_dir, synth_dir, sums_dir = _default_artifact_paths(cfg)
    _ensure_dirs(ckpt_dir, synth_dir, sums_dir)

    # -------- Reproducibility --------
    np.random.seed(seed)

    # -------- Load dataset --------
    if load_dataset_npy is not None:
        x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_npy(
            data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
        )
    else:
        # Minimal fallback loader (expects 4 files under DATA_DIR)
        x_train = np.load(data_dir / "train_data.npy").astype("float32")
        y_train = np.load(data_dir / "train_labels.npy")
        x_test  = np.load(data_dir / "test_data.npy").astype("float32")
        y_test  = np.load(data_dir / "test_labels.npy")

        if x_train.max() > 1.5:
            x_train = x_train / 255.0
            x_test  = x_test  / 255.0

        H, W, C = img_shape
        x_train = x_train.reshape((-1, H, W, C))
        x_test  = x_test.reshape((-1, H, W, C))

        n_val = int(len(x_test) * float(cfg.get("VAL_FRACTION", 0.5)))
        x_val, y_val = x_test[:n_val], y_test[:n_val]
        x_test, y_test = x_test[n_val:], y_test[n_val:]

    # -------- Train per-class GMMs --------
    print(f"[config] GMM components={n_components}, cov='{covariance_type}', reg_covar={reg_covar}, max_iter={max_iter}")
    print(f"[paths]  ckpts={ckpt_dir.resolve()} | synthetic={synth_dir.resolve()} | summaries={sums_dir.resolve()}")

    train_per_class_gmms(
        x_train=x_train,
        y_train=y_train,
        img_shape=img_shape,
        num_classes=num_classes,
        ckpt_dir=ckpt_dir,
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=seed,
        train_global_fallback=train_global_fallback,
        verbose=True,
    )

    # -------- Preview grid (1 sample per class) --------
    preview_path = sums_dir / "gmm_train_preview.png"
    try:
        save_grid_from_dir(
            ckpt_dir=ckpt_dir,
            img_shape=img_shape,
            num_classes=num_classes,
            path=preview_path,
            per_class=1,
        )
        print(f"[preview] Saved GMM preview grid to {preview_path}")
    except Exception as e:
        print(f"[warn] Could not save preview grid: {e}")


# --- adapter so app.main can call train([...]) or train({}) ---
def _coerce_cfg(cfg_or_argv):
    """
    Accept either a parsed dict or an argv list/tuple like:
      ['--config', 'configs/config.yaml']
    Returns a Python dict.
    """
    if isinstance(cfg_or_argv, dict):
        return dict(cfg_or_argv)
    if isinstance(cfg_or_argv, (list, tuple)):
        import yaml
        # default path if none provided
        cfg_path = None
        if "--config" in cfg_or_argv:
            i = cfg_or_argv.index("--config")
            if i + 1 < len(cfg_or_argv):
                cfg_path = Path(cfg_or_argv[i + 1])
        if cfg_path is None:
            cfg_path = Path("configs/config.yaml")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    raise TypeError(f"Unsupported config payload type: {type(cfg_or_argv)}")


# =============================================================================
# CLI entrypoints
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-class GaussianMixture models")
    p.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    return p.parse_args()


def main(argv=None):
    """
    CLI entrypoint compatible with `python -m gaussianmixture.train`
    and callable as `main(['--config','configs/config.yaml'])`.
    """
    if isinstance(argv, dict):
        run_train(argv)
        return 0
    if argv is None:
        args = parse_args()
        cfg = _load_yaml(Path(args.config))
    else:
        p = argparse.ArgumentParser(description="Train per-class GaussianMixture models")
        p.add_argument("--config", default="configs/config.yaml")
        args = p.parse_args(argv)
        cfg = _load_yaml(Path(args.config))
    run_train(cfg)
    return 0


def _train_from_cfg(cfg: Dict) -> Dict[str, float]:
    """
    Adapter for the unified CLI (app.main calls train(cfg_or_argv)).
    Returns a small dict, but more importantly returns rc=0 via `train`.
    """
    run_train(cfg)
    return {"status": "ok"}


def train(cfg_or_argv) -> int:
    """
    Unified-CLI adapter. Guarantees rc=0 on success to satisfy the orchestrator.
    """
    cfg = _coerce_cfg(cfg_or_argv)
    _train_from_cfg(cfg)  # will raise on hard errors
    return 0


__all__ = [
    "train_per_class_gmms",
    "run_train",
    "train",
    "main",
]
