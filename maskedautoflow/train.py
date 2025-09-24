# maskedautoflow/train.py

# =============================================================================
# Training utilities for the Masked Autoregressive Flow (MAF).
#
# Highlights
# ----------
# - Keras 3–compatible weight saving (no `save_format=`).
# - Optional dequantization noise (helps continuous likelihoods on near-binary data).
# - Clean tf.data input pipeline (float32 in [0,1], flattened to (N, D)).
# - Early stopping, gradient clipping, TensorBoard summaries.
# - Unified-CLI adapter: `train(cfg: dict)` entrypoint.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

import numpy as np
import tensorflow as tf

# Preferred shared loader
try:
    from common.data import load_dataset_npy  # type: ignore
except Exception:
    load_dataset_npy = None  # fallback used below

# Try dict first; fall back to argv if it complains
try:
    pass
except TypeError:
    pass


from maskedautoflow.models import (
    MAF,
    MAFConfig,
    build_maf_model,
    flatten_images,
)

for d in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(d, True)
    except Exception: pass


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _cfg_get(cfg: Dict, dotted: str, default=None):
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _default_artifact_paths(cfg: Dict) -> tuple[Path, Path, Path]:
    """
    Standardize artifact directories with sensible defaults:
      artifacts/maskedautoflow/checkpoints
      artifacts/maskedautoflow/synthetic
      artifacts/maskedautoflow/summaries
    """
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    arts = cfg.get("ARTIFACTS", {})
    ckpt = Path(arts.get("maskedautoflow_checkpoints", artifacts_root / "maskedautoflow" / "checkpoints"))
    synth = Path(arts.get("maskedautoflow_synthetic", artifacts_root / "maskedautoflow" / "synthetic"))
    sums = Path(arts.get("maskedautoflow_summaries", artifacts_root / "maskedautoflow" / "summaries"))
    return ckpt, synth, sums


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Data / shapes
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)

    # Loader / batching
    BATCH_SIZE: int = 256

    # Optimization
    EPOCHS: int = 50
    LR: float = 2e-4
    PATIENCE: int = 10
    CLIP_GRAD: float = 1.0
    SEED: int = 42

    # Model
    NUM_FLOWS: int = 5
    HIDDEN_DIMS: Tuple[int, ...] = (128, 128)

    # Data smoothing for continuous likelihoods (recommended for 0/1-ish inputs)
    DEQUANT_NOISE: bool = True
    DEQUANT_EPS: float = 1.0 / 256.0  # uniform noise U(0, eps)


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
def _to_float01(x: np.ndarray) -> np.ndarray:
    """
    Cast to float32 and map byte-like inputs to [0,1].
    """
    x = x.astype("float32", copy=False)
    if x.max() > 1.5:  # looks like 0..255
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def build_datasets(
    x_train: np.ndarray,
    x_val: np.ndarray,
    cfg: TrainConfig,
    *,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Turn HWC (or flat) numpy arrays into flattened float32 tf.data pipelines.
    Labels are ignored (unconditional density estimation).

    Returns
    -------
    train_ds, val_ds : tf.data.Dataset
        Datasets yielding tensors of shape (B, D) in [0,1].
    """
    H, W, C = cfg.IMG_SHAPE

    # Convert to [0,1] in HWC if needed (leave flat arrays as-is)
    Xtr = _to_float01(x_train)
    Xva = _to_float01(x_val)

    # Optional dequantization noise (before flattening)
    if cfg.DEQUANT_NOISE and cfg.DEQUANT_EPS > 0.0:
        rng = np.random.default_rng(cfg.SEED)
        Xtr = np.clip(Xtr + rng.uniform(0.0, cfg.DEQUANT_EPS, size=Xtr.shape).astype("float32"), 1e-6, 1.0 - 1e-6)
        Xva = np.clip(Xva + rng.uniform(0.0, cfg.DEQUANT_EPS, size=Xva.shape).astype("float32"), 1e-6, 1.0 - 1e-6)

    # Flatten to (N, D)
    # (flatten_images signature in this repo supports assume_01 + clip)
    Xtr = flatten_images(Xtr, (H, W, C), assume_01=True, clip=True)
    Xva = flatten_images(Xva, (H, W, C), assume_01=True, clip=True)

    def _make(x: np.ndarray, do_shuffle: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(x.astype("float32", copy=False))
        if do_shuffle:
            ds = ds.shuffle(
                buffer_size=min(10_000, x.shape[0]),
                seed=cfg.SEED,
                reshuffle_each_iteration=True,
            )
        return ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return _make(Xtr, shuffle), _make(Xva, False)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def build_model(cfg: TrainConfig) -> MAF:
    """
    Instantiate a MAF and create variables under the current strategy/device.
    """
    model = build_maf_model(
        MAFConfig(IMG_SHAPE=cfg.IMG_SHAPE, NUM_FLOWS=cfg.NUM_FLOWS, HIDDEN_DIMS=cfg.HIDDEN_DIMS)
    )
    H, W, C = cfg.IMG_SHAPE
    D = H * W * C
    _ = model(tf.zeros((1, D), dtype=tf.float32))  # build variables
    return model


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train_maf_model(
    model: MAF,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    cfg: TrainConfig,
    ckpt_dir: Path,
    writer: Optional[tf.summary.SummaryWriter] = None,
    log_cb: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """
    Train a MAF using negative log-likelihood with early stopping.

    Returns
    -------
    summary: dict with keys:
      - best_path, last_path (str)
      - best_val (float)
      - epochs_run (int)
    """
    # Seeding (TF & Python RNGs used under the hood)
    tf.keras.utils.set_random_seed(cfg.SEED)

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "MAF_best.weights.h5"
    last_path = ckpt_dir / "MAF_last.weights.h5"

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.LR)

    @tf.function(reduce_retracing=True)
    def train_step(x: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            nll = -tf.reduce_mean(model.log_prob(x))  # scalar
        grads = tape.gradient(nll, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, cfg.CLIP_GRAD)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return nll

    @tf.function(reduce_retracing=True)
    def val_step(x: tf.Tensor) -> tf.Tensor:
        return -tf.reduce_mean(model.log_prob(x))

    def _log(stage: str, msg: str) -> None:
        if log_cb:
            try:
                log_cb(stage, msg)
                return
            except Exception:
                pass
        print(f"[{stage}] {msg}")

    best_val = float("inf")
    patience = 0
    epochs_run = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        epochs_run = epoch
        tr_metric = tf.keras.metrics.Mean()
        va_metric = tf.keras.metrics.Mean()

        # ---- Train ----
        for xb in train_ds:
            tr_metric.update_state(train_step(xb))

        # ---- Validate ----
        for xb in val_ds:
            va_metric.update_state(val_step(xb))

        tr = float(tr_metric.result().numpy())
        va = float(va_metric.result().numpy())

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/Train_NLL", tr, step=epoch)
                tf.summary.scalar("Loss/Val_NLL", va, step=epoch)

        _log("train", f"epoch {epoch:03d}: train_nll={tr:.4f} | val_nll={va:.4f}")

        # Save "last" each epoch (Keras 3: extension selects format; no `save_format=`)
        model.save_weights(str(last_path), overwrite=True)

        # Early stopping on best validation NLL
        if va < best_val - 1e-6:
            best_val = va
            patience = 0
            model.save_weights(str(best_path), overwrite=True)
            _log("ckpt", f"Saved {best_path.name}")
        else:
            patience += 1
            _log("train", f"patience {patience}/{cfg.PATIENCE}")
            if patience >= cfg.PATIENCE:
                _log("train", "early stopping.")
                break

    (ckpt_dir / "MAF_LAST_OK").write_text("ok", encoding="utf-8")

    return {
        "best_path": str(best_path),
        "last_path": str(last_path),
        "best_val": best_val,
        "epochs_run": epochs_run,
    }


# ---------------------------------------------------------------------
# Unified CLI adapter
# ---------------------------------------------------------------------
def _load_split(cfg: Dict, img_shape: Tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (x_train, x_val) shaped (N,H,W,C) in [0,1].
    """
    data_dir = Path(_cfg_get(cfg, "DATA_DIR", _cfg_get(cfg, "data.root", "data")))
    val_frac = float(cfg.get("VAL_FRACTION", 0.5))

    if load_dataset_npy is not None:
        x_train, _ytr, _x_val, _yva, _x_test, _yte = load_dataset_npy(
            data_dir, img_shape, int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 1))), val_fraction=val_frac
        )
        return x_train, _x_val

    # Fallback: expect 4 files under data_dir
    x_tr = np.load(data_dir / "train_data.npy").astype("float32")
    x_te = np.load(data_dir / "test_data.npy").astype("float32")

    # Normalize & reshape
    if x_tr.max() > 1.5:
        x_tr = x_tr / 255.0
        x_te = x_te / 255.0
    H, W, C = img_shape
    x_tr = x_tr.reshape((-1, H, W, C))
    x_te = x_te.reshape((-1, H, W, C))

    # Split test -> (val, test) just to obtain a val set
    n_val = int(len(x_te) * val_frac)
    x_val = x_te[:n_val]
    return x_tr, x_val


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
        from pathlib import Path
        # default path if none provided
        cfg_path = None
        if "--config" in cfg_or_argv:
            i = cfg_or_argv.index("--config")
            if i + 1 < len(cfg_or_argv):
                cfg_path = Path(cfg_or_argv[i + 1])
        if cfg_path is None:
            # fall back to repo default if the module has one; else use configs/config.yaml
            cfg_path = Path("configs/config.yaml")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    raise TypeError(f"Unsupported config payload type: {type(cfg_or_argv)}")


def _train_from_cfg(cfg: Dict) -> Dict[str, Any]:
    """
    Adapter used by the GenCyberSynth unified CLI.

    Behavior:
      - Resolves config knobs (supports both NEW and LEGACY keys).
      - Loads dataset (x_train, x_val) for unconditional density modeling.
      - Builds tf.data datasets and the MAF model.
      - Trains with early stopping and saves Keras 3-style weights:
          artifacts/maskedautoflow/checkpoints/MAF_best.weights.h5
          artifacts/maskedautoflow/checkpoints/MAF_last.weights.h5
      - Writes TensorBoard logs under artifacts/maskedautoflow/summaries/tb/
    """
    # ---- Resolve shapes and hyperparams from cfg ----
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    bs = int(_cfg_get(cfg, "BATCH_SIZE", 256))
    epochs = int(_cfg_get(cfg, "EPOCHS", 50))
    lr = float(_cfg_get(cfg, "LR", 2e-4))
    patience = int(_cfg_get(cfg, "PATIENCE", 10))
    clip = float(_cfg_get(cfg, "CLIP_GRAD", 1.0))
    seed = int(_cfg_get(cfg, "SEED", 42))
    num_flows = int(_cfg_get(cfg, "NUM_FLOWS", 5))
    hidden = tuple(int(h) for h in _cfg_get(cfg, "HIDDEN_DIMS", (128, 128)))
    deq = bool(_cfg_get(cfg, "DEQUANT_NOISE", True))
    deq_eps = float(_cfg_get(cfg, "DEQUANT_EPS", 1.0 / 256.0))

    # ---- Artifacts ----
    ckpt_dir, _synth_dir, sums_dir = _default_artifact_paths(cfg)
    (ckpt_dir).mkdir(parents=True, exist_ok=True)
    (sums_dir).mkdir(parents=True, exist_ok=True)
    tb_dir = sums_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # ---- Compose TrainConfig ----
    tcfg = TrainConfig(
        IMG_SHAPE=(H, W, C),
        BATCH_SIZE=bs,
        EPOCHS=epochs,
        LR=lr,
        PATIENCE=patience,
        CLIP_GRAD=clip,
        SEED=seed,
        NUM_FLOWS=num_flows,
        HIDDEN_DIMS=hidden,
        DEQUANT_NOISE=deq,
        DEQUANT_EPS=deq_eps,
    )

    # ---- Data ----
    x_train, x_val = _load_split(cfg, (H, W, C))
    train_ds, val_ds = build_datasets(x_train, x_val, tcfg, shuffle=True)

    # ---- Model ----
    model = build_model(tcfg)

    # ---- Train ----
    writer = tf.summary.create_file_writer(str(tb_dir))

    # Nice one-line config summary (moved here from global scope)
    print(
        f"[maf] HWC={(H, W, C)} bs={bs} epochs={epochs} lr={lr} "
        f"flows={num_flows} hidden={hidden} dequant={deq} eps={deq_eps} seed={seed}"
    )

    summary = train_maf_model(
        model, train_ds, val_ds, cfg=tcfg, ckpt_dir=ckpt_dir, writer=writer
    )

    print(
        f"[maf] training complete | best_val={summary['best_val']:.4f} | "
        f"epochs={summary['epochs_run']} | ckpts={ckpt_dir}"
    )
    return summary


def train(cfg_or_argv) -> Dict[str, Any]:
    cfg = _coerce_cfg(cfg_or_argv)
    return _train_from_cfg(cfg)


__all__ = [
    "TrainConfig",
    "build_datasets",
    "build_model",
    "train_maf_model",
    "train",
]

# --- simple CLI so `python -m maskedautoflow.train` works ---
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Train Masked Autoregressive Flow (MAF)")
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args(argv)
    # route through the same path used by the unified adapter
    _train_from_cfg(_coerce_cfg(["--config", args.config]))

if __name__ == "__main__":
    main()