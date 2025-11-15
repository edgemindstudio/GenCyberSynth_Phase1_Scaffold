# restrictedboltzmann/train.py
"""
Bernoulli–Bernoulli Restricted Boltzmann Machine (RBM) training for GenCyberSynth.

What this module provides
-------------------------
- `build_visible_dataset(...)`  → tf.data over flattened visible vectors.
- `cd_k_update(...)`            → one Contrastive Divergence (CD-k) step.
- `train_rbm(...)`              → full loop with early stopping + checkpoints.
- `main(argv=None)`             → argv-style entrypoint (works with app.main).
- `train(cfg_or_argv)`          → dict/argv adapter (works with app.main).

Checkpoints
-----------
Saved per class under:
  ${paths.artifacts}/restrictedboltzmann/checkpoints/class_{k}/
    - RBM_best.weights.h5
    - RBM_last.weights.h5
    - RBM_epoch_XXXX.weights.h5 (periodic)

Notes
-----
- Inputs are expected in [0,1], shaped (N,H,W,C). We binarize at 0.5 for RBM.
- If a class has very few samples, we emit a tiny stub so synth won't skip it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
import yaml
import sys

# Try shared loader first; fall back to raw .npy
try:
    from common.data import load_dataset_npy  # type: ignore
except Exception:
    load_dataset_npy = None  # noqa: E305

# Local model (should expose save_weights/load_weights or variables)
from .models import BernoulliRBM  # noqa: E402


# --------- GPU niceness (harmless on CPU nodes) ----------
for d in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(d, True)
    except Exception:
        pass


# ===========================
# Small helpers
# ===========================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).reshape(-1)[0])


def _cfg_get(cfg: Dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _coerce_cfg(cfg_or_argv) -> Dict:
    """
    Accept either a Python dict or an argv list/tuple like:
      ['--config', 'configs/config.yaml']
    Return a config dict.
    """
    if isinstance(cfg_or_argv, dict):
        return dict(cfg_or_argv)
    if isinstance(cfg_or_argv, (list, tuple)):
        import argparse
        p = argparse.ArgumentParser(description="RBM trainer")
        p.add_argument("--config", default="configs/config.yaml")
        args = p.parse_args(list(cfg_or_argv))
        with open(args.config, "r") as f:
            return yaml.safe_load(f) or {}
    raise TypeError(f"Unsupported config payload type: {type(cfg_or_argv)}")


def _normalize_artifacts(cfg: Dict) -> Dict:
    """Honor paths.artifacts and derive RBM subpaths."""
    arts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    cfg.setdefault("ARTIFACTS", {})
    A = cfg["ARTIFACTS"]
    A.setdefault("rbm_ckpts",     str(arts_root / "restrictedboltzmann" / "checkpoints"))
    A.setdefault("rbm_summaries", str(arts_root / "restrictedboltzmann" / "summaries"))
    return cfg


def _load_dataset(cfg: Dict, img_shape: Tuple[int, int, int], num_classes: int):
    """Shared loader: use common.data.load_dataset_npy if present; else raw .npy."""
    data_dir = Path(cfg.get("DATA_DIR", _cfg_get(cfg, "data.root", "data")))
    if load_dataset_npy is not None:
        return load_dataset_npy(
            data_dir, img_shape, num_classes, val_fraction=float(cfg.get("VAL_FRACTION", 0.5))
        )
    # Fallback
    xtr = np.load(data_dir / "train_data.npy").astype("float32")
    ytr = np.load(data_dir / "train_labels.npy")
    xte = np.load(data_dir / "test_data.npy").astype("float32")
    yte = np.load(data_dir / "test_labels.npy")
    if xtr.max() > 1.5:
        xtr /= 255.0
        xte /= 255.0
    H, W, C = img_shape
    xtr = xtr.reshape((-1, H, W, C))
    xte = xte.reshape((-1, H, W, C))
    n_val = int(len(xte) * float(cfg.get("VAL_FRACTION", 0.5)))
    xva, yva = xte[:n_val], yte[:n_val]
    xte, yte = xte[n_val:], yte[n_val:]
    return xtr, ytr, xva, yva, xte, yte


def _int_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    return (np.argmax(y, axis=1) if (y.ndim == 2 and y.shape[1] == num_classes) else y).astype(int)


# ===========================
# Data pipeline
# ===========================
def build_visible_dataset(
    x: np.ndarray,
    *,
    img_shape: Tuple[int, int, int],
    batch_size: int,
    shuffle: bool = True,
    binarize: bool = True,
    threshold: float = 0.5,
) -> tf.data.Dataset:
    """Return tf.data over flattened visible vectors (B,V) in {0,1} or [0,1]."""
    H, W, C = img_shape
    V = H * W * C

    def _prep(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype("float32")
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = arr.reshape((-1, V))
        if binarize:
            arr = (arr > threshold).astype("float32")
        return arr

    x_flat = _prep(x)
    ds = tf.data.Dataset.from_tensor_slices(x_flat)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x_flat), 8192), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ===========================
# CD-k update
# ===========================
@tf.function(reduce_retracing=True)
def cd_k_update(
    W: tf.Variable,
    v_bias: tf.Variable,
    h_bias: tf.Variable,
    v0: tf.Tensor,
    *,
    k: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    One CD-k step using probabilities (less noisy).
    Returns (v_recon_prob, mse).
    """
    # Positive phase
    h0_prob = tf.nn.sigmoid(tf.matmul(v0, W) + h_bias)  # (B,H)

    # Negative phase (Gibbs)
    h = tf.cast(tf.random.uniform(tf.shape(h0_prob)) < h0_prob, tf.float32)
    for _ in range(k):
        v_prob = tf.nn.sigmoid(tf.matmul(h, tf.transpose(W)) + v_bias)  # (B,V)
        v = tf.cast(tf.random.uniform(tf.shape(v_prob)) < v_prob, tf.float32)
        h_prob = tf.nn.sigmoid(tf.matmul(v, W) + h_bias)                # (B,H)
        h = tf.cast(tf.random.uniform(tf.shape(h_prob)) < h_prob, tf.float32)

    vk_prob = v_prob
    hk_prob = h_prob
    B = tf.cast(tf.shape(v0)[0], tf.float32)

    dW  = (tf.matmul(tf.transpose(v0), h0_prob) - tf.matmul(tf.transpose(vk_prob), hk_prob)) / B
    dvb = tf.reduce_mean(v0 - vk_prob, axis=0)
    dhb = tf.reduce_mean(h0_prob - hk_prob, axis=0)
    if weight_decay > 0.0:
        dW -= weight_decay * W

    W.assign_add(lr * dW)
    v_bias.assign_add(lr * dvb)
    h_bias.assign_add(lr * dhb)

    mse = tf.reduce_mean(tf.square(v0 - vk_prob))
    return vk_prob, mse


# ===========================
# Training loop
# ===========================
@dataclass
class RBMTrainConfig:
    img_shape: Tuple[int, int, int] = (40, 40, 1)
    epochs: int = 50
    batch_size: int = 128
    cd_k: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    save_every: int = 10
    ckpt_dir: str = "artifacts/restrictedboltzmann/checkpoints"


def train_rbm(
    rbm: BernoulliRBM,
    x_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    *,
    cfg: RBMTrainConfig,
    log_cb: Optional[Callable[[int, float, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    """Train an RBM with CD-k + early stopping; save BEST/LAST + periodic."""
    H, W, C = cfg.img_shape
    V = H * W * C

    ds_tr = build_visible_dataset(x_train, img_shape=cfg.img_shape, batch_size=cfg.batch_size, shuffle=True,  binarize=True)
    ds_va = build_visible_dataset(x_val,   img_shape=cfg.img_shape, batch_size=cfg.batch_size, shuffle=False, binarize=True) if x_val is not None else None

    # Ensure variables exist (for some subclass patterns)
    try:
        _ = rbm.W.shape
    except Exception:
        rbm(tf.zeros((1, V), dtype=tf.float32))

    ckpt_dir = Path(cfg.ckpt_dir)
    _ensure_dir(ckpt_dir)
    best_path = ckpt_dir / "RBM_best.weights.h5"
    last_path = ckpt_dir / "RBM_last.weights.h5"

    best_val = np.inf
    patience_ctr = 0
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        # --- Train ---
        tr_losses = []
        for v0 in ds_tr:
            v0 = tf.convert_to_tensor(v0, dtype=tf.float32)
            Vdim = tf.shape(v0)[1]
            v0 = tf.reshape(v0, (-1, V)) if Vdim != V else v0
            _, mse = cd_k_update(rbm.W, rbm.v_bias, rbm.h_bias, v0, k=cfg.cd_k, lr=cfg.lr, weight_decay=cfg.weight_decay)
            tr_losses.append(_as_float(mse))
        tr = float(np.mean(tr_losses)) if tr_losses else np.nan

        # --- Val ---
        va = None
        if ds_va is not None:
            va_losses = []
            for vv in ds_va:
                vv = tf.convert_to_tensor(vv, dtype=tf.float32)
                Vdim = tf.shape(vv)[1]
                vv = tf.reshape(vv, (-1, V)) if Vdim != V else vv
                h_prob = tf.nn.sigmoid(tf.matmul(vv, rbm.W) + rbm.h_bias)
                v_prob = tf.nn.sigmoid(tf.matmul(h_prob, tf.transpose(rbm.W)) + rbm.v_bias)
                va_losses.append(_as_float(tf.reduce_mean(tf.square(vv - v_prob))))
            va = float(np.mean(va_losses)) if va_losses else np.nan

        if log_cb:
            log_cb(epoch, tr, va)

        # Periodic epoch snapshot
        if (epoch == 1) or (epoch % max(1, int(cfg.save_every)) == 0):
            rbm.save_weights(str(ckpt_dir / f"RBM_epoch_{epoch:04d}.weights.h5"))

        # Early stopping (prefer val if available)
        monitor = va if va is not None else tr
        improved = (monitor < best_val) if np.isfinite(monitor) else False
        if improved:
            best_val = monitor
            best_epoch = epoch
            patience_ctr = 0
            rbm.save_weights(str(best_path))
        else:
            patience_ctr += 1
            if patience_ctr >= int(cfg.patience):
                rbm.save_weights(str(last_path))
                return {"best_val": float(best_val), "best_epoch": int(best_epoch), "last_train": float(tr), "stopped_early": True}

    rbm.save_weights(str(last_path))
    return {"best_val": float(best_val if np.isfinite(best_val) else tr), "best_epoch": int(best_epoch if best_epoch > 0 else cfg.epochs), "last_train": float(tr), "stopped_early": False}


# ===========================
# High-level runner
# ===========================
def _run_train(cfg: Dict) -> int:
    # Defaults & artifacts
    cfg = dict(cfg)
    cfg.setdefault("SEED", 42)
    np.random.seed(int(cfg["SEED"]))
    tf.random.set_seed(int(cfg["SEED"]))
    _normalize_artifacts(cfg)

    H, W, C = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    K = int(cfg.get("NUM_CLASSES", cfg.get("num_classes", 9)))
    V = H * W * C

    # Hparams (RBM-specific fallbacks)
    hidden   = int(cfg.get("RBM_HIDDEN", 256))
    epochs   = int(cfg.get("RBM_EPOCHS", cfg.get("EPOCHS", 50)))
    batch    = int(cfg.get("RBM_BATCH",  cfg.get("BATCH_SIZE", 128)))
    cd_k     = int(cfg.get("CD_K", 1))
    lr       = float(cfg.get("RBM_LR", cfg.get("LR", 1e-3)))
    wd       = float(cfg.get("WEIGHT_DECAY", 0.0))
    patience = int(cfg.get("PATIENCE", 10))
    save_every = int(cfg.get("SAVE_EVERY", 10))

    # Paths
    ckpt_root = Path(cfg["ARTIFACTS"]["rbm_ckpts"])
    sums_dir  = Path(cfg["ARTIFACTS"]["rbm_summaries"])
    _ensure_dir(ckpt_root); _ensure_dir(sums_dir)

    # Data
    x_tr, y_tr, x_va, y_va, _x_te, _y_te = _load_dataset(cfg, (H, W, C), K)
    y_tr_i = _int_labels(y_tr, K)
    y_va_i = _int_labels(y_va, K) if y_va is not None else None

    print(f"[rbm] img_shape={(H,W,C)} K={K} V={V} hidden={hidden} epochs={epochs} batch={batch} cd_k={cd_k}")

    # Train per class (or emit stub if too few)
    for k in range(K):
        idx = (y_tr_i == k)
        n_k = int(idx.sum())
        class_dir = ckpt_root / f"class_{k}"
        _ensure_dir(class_dir)

        if n_k < 2:
            # Emit a minimal stub so downstream synth won't skip
            stub = class_dir / "RBM_last.weights.h5"
            np.savez(class_dir / "RBM_stub_class_%d.npz" % k, W=np.zeros((V, 1), dtype=np.float32), v_bias=np.zeros((V,), dtype=np.float32), h_bias=np.zeros((1,), dtype=np.float32))
            # Touch h5-ish markers so adapter is happy
            (class_dir / "RBM_best.weights.h5").touch()
            stub.touch()
            print(f"[rbm] class {k}: too few samples (n={n_k}); wrote stub checkpoints.")
            continue

        rbm = BernoulliRBM(visible_dim=V, hidden_dim=hidden)
        cfg_train = RBMTrainConfig(
            img_shape=(H, W, C),
            epochs=epochs,
            batch_size=batch,
            cd_k=cd_k,
            lr=lr,
            weight_decay=wd,
            patience=patience,
            save_every=save_every,
            ckpt_dir=str(class_dir),
        )

        def _log(ep, tr, va):
            if (ep == 1) or (ep % max(1, save_every) == 0):
                msg = f"[rbm][k={k}] epoch={ep:04d} train_mse={tr:.5f}"
                if va is not None:
                    msg += f" val_mse={va:.5f}"
                print(msg)

        print(f"[rbm] training class {k} on n={n_k} samples → {class_dir}")
        _ = train_rbm(
            rbm,
            x_train=x_tr[idx],
            x_val=(x_va[y_va_i == k] if (y_va_i is not None) else None),
            cfg=cfg_train,
            log_cb=_log,
        )

    # Optional: preview grid if sampler helper exists
    try:
        from .sample import save_grid_from_checkpoints
        out = sums_dir / "rbm_train_preview.png"
        save_grid_from_checkpoints(
            ckpt_root=ckpt_root,
            img_shape=(H, W, C),
            num_classes=K,
            path=out,
            per_class=1,
        )
        print(f"[rbm] preview grid → {out}")
    except Exception as e:
        print(f"[rbm][warn] preview grid failed: {e}")

    return 0


# ===========================
# Public entrypoints
# ===========================
def main(argv=None) -> int:
    """
    argv-style entrypoint so `app.main` can call:
      main(['--config','configs/config.yaml'])
    Also accepts a dict config for flexibility.
    """
    if isinstance(argv, dict):
        return _run_train(argv)
    if argv is None:
        import argparse
        p = argparse.ArgumentParser(description="RBM trainer")
        p.add_argument("--config", default="configs/config.yaml")
        args = p.parse_args()
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return _run_train(cfg)
    # explicit argv list/tuple
    cfg = _coerce_cfg(argv)
    return _run_train(cfg)


def train(cfg_or_argv) -> int:
    """
    Dict/argv adapter so `app.main` can call train(config_dict) *or*
    train(['--config','configs/config.yaml']). Returns 0 on success.
    """
    cfg = _coerce_cfg(cfg_or_argv)
    return _run_train(cfg)


__all__ = [
    "RBMTrainConfig",
    "build_visible_dataset",
    "cd_k_update",
    "train_rbm",
    "main",
    "train",
]

if __name__ == "__main__":
    raise SystemExit(main())
