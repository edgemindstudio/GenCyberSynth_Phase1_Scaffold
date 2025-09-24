# vae/train.py

"""
Train the Conditional VAE (cVAE) and save checkpoints + a small preview grid.

Usage
-----
# From repo root (or any directory), run:
python -m vae.train --config config.yaml

What this script does
---------------------
1) Reads a YAML config (see keys below).
2) Loads dataset from four .npy files under DATA_DIR:
   - train_data.npy, train_labels.npy, test_data.npy, test_labels.npy
3) Converts inputs for training (decoder is tanh → expects [-1, 1]).
4) Trains the VAE via `VAEPipeline(cfg)` with TensorBoard-friendly logging.
5) Saves a 1×N preview grid from the decoder to ARTIFACTS.summaries.

Expected config keys (with sensible defaults if missing)
--------------------------------------------------------
SEED: 42
DATA_DIR: "USTC-TFC2016_malware"
IMG_SHAPE: [40, 40, 1]
NUM_CLASSES: 9
LATENT_DIM: 100
EPOCHS: 2000
BATCH_SIZE: 256
LR: 2e-4
BETA_1: 0.5
BETA_KL: 1.0
VAL_FRACTION: 0.5
ARTIFACTS:
  checkpoints: artifacts/vae/checkpoints
  synthetic:   artifacts/vae/synthetic
  summaries:   artifacts/vae/summaries
  tensorboard: artifacts/tensorboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from common.data import load_dataset_npy, to_minus1_1
import yaml

# Ensure sibling packages are importable (vae/, eval/, etc.)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vae.pipeline import VAEPipeline
from vae.sample import save_grid_from_decoder as save_grid


# ---------------------------
# Small utilities
# ---------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs(cfg: Dict) -> None:
    arts = cfg.get("ARTIFACTS", {})
    for key in ("checkpoints", "synthetic", "summaries", "tensorboard"):
        p = arts.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


# ---------------------------
# TensorBoard-friendly logging callback
# ---------------------------
def make_log_cb(tboard_dir: Path | None):
    writer = None
    if tboard_dir:
        tboard_dir.mkdir(parents=True, exist_ok=True)
        writer = tf.summary.create_file_writer(str(tboard_dir))

    def cb(epoch: int, train_loss: float, recon_loss: float, kl_loss: float, val_loss: float):
        print(
            f"[epoch {epoch:05d}] "
            f"train={train_loss:.4f} | recon={recon_loss:.4f} | KL={kl_loss:.4f} | val={val_loss:.4f}"
        )
        if writer:
            with writer.as_default():
                tf.summary.scalar("loss/train_total", train_loss, step=epoch)
                tf.summary.scalar("loss/train_recon", recon_loss, step=epoch)
                tf.summary.scalar("loss/train_kl",    kl_loss,    step=epoch)
                tf.summary.scalar("loss/val_total",   val_loss,   step=epoch)
                writer.flush()

    return cb


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Conditional VAE (cVAE)")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    # Be nice to GPUs
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    args = parse_args()
    cfg = load_yaml(Path(args.config))

    # Sensible defaults
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", 0.5)
    cfg.setdefault("LATENT_DIM", 100)
    cfg.setdefault("ARTIFACTS", {})
    cfg["ARTIFACTS"].setdefault("checkpoints", "artifacts/vae/checkpoints")
    cfg["ARTIFACTS"].setdefault("synthetic",   "artifacts/vae/synthetic")
    cfg["ARTIFACTS"].setdefault("summaries",   "artifacts/vae/summaries")
    cfg["ARTIFACTS"].setdefault("tensorboard", "artifacts/tensorboard")

    print(f"[config] Using {Path(args.config).resolve()}")

    set_seed(int(cfg.get("SEED", 42)))
    ensure_dirs(cfg)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])

    # Load dataset in [0,1]; split test → (val, test)
    x_train01, y_train, x_val01, y_val, _, _ = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    # Map to [-1,1] for tanh decoder
    x_train_m11 = to_minus1_1(x_train01)
    x_val_m11   = to_minus1_1(x_val01)

    # Attach a logging callback (TensorBoard) for VAEPipeline
    cfg["LOG_CB"] = make_log_cb(Path(cfg["ARTIFACTS"]["tensorboard"]))

    # Train
    pipe = VAEPipeline(cfg)
    enc, dec = pipe.train(
        x_train=x_train_m11, y_train=y_train,
        x_val=x_val_m11,     y_val=y_val,
    )

    # Save a small decoder preview grid
    preview_path = save_grid(
        dec, num_classes, cfg["LATENT_DIM"],
        n=min(9, num_classes),
        path=Path(cfg["ARTIFACTS"]["summaries"]) / "train_preview.png",
    )
    print(f"Saved preview grid to {preview_path}")


# ---- add under your existing helpers ----
def _cfg_get(cfg: Dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def _normalize_artifacts(cfg: Dict) -> Dict:
    """Fill ARTIFACTS with sensible defaults, honoring paths.artifacts if present."""
    arts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    cfg.setdefault("ARTIFACTS", {})
    A = cfg["ARTIFACTS"]
    A.setdefault("checkpoints", str(arts_root / "vae" / "checkpoints"))
    A.setdefault("synthetic",   str(arts_root / "vae" / "synthetic"))
    A.setdefault("summaries",   str(arts_root / "vae" / "summaries"))
    A.setdefault("tensorboard", str(arts_root / "tensorboard"))
    return cfg


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


# ---- unified CLI entrypoint ----
def _train_from_cfg(cfg: Dict) -> Dict[str, float]:
    # be nice to GPUs
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # defaults + artifacts normalization
    cfg = dict(cfg)  # shallow copy
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", 0.5)
    cfg.setdefault("LATENT_DIM", 100)
    _normalize_artifacts(cfg)

    set_seed(int(cfg["SEED"]))
    ensure_dirs(cfg)

    data_dir    = Path(cfg.get("DATA_DIR", _cfg_get(cfg, "data.root", "data")))
    img_shape   = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    num_classes = int(cfg.get("NUM_CLASSES", cfg.get("num_classes", 9)))

    # load [0,1], split test->(val,test)
    x_train01, y_train, x_val01, y_val, _, _ = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    # map to [-1,1] for tanh decoder
    x_train_m11 = to_minus1_1(x_train01)
    x_val_m11   = to_minus1_1(x_val01)

    # tensorboard callback
    cfg["LOG_CB"] = make_log_cb(Path(cfg["ARTIFACTS"]["tensorboard"]))

    # train
    pipe = VAEPipeline(cfg)
    enc, dec = pipe.train(x_train=x_train_m11, y_train=y_train,
                          x_val=x_val_m11,     y_val=y_val)

    # tiny decoder preview grid
    preview_path = save_grid(
        dec, num_classes, cfg["LATENT_DIM"],
        n=min(9, num_classes),
        path=Path(cfg["ARTIFACTS"]["summaries"]) / "train_preview.png",
    )
    print(f"[vae] saved preview grid → {preview_path}")
    return {"preview": str(preview_path)}


if __name__ == "__main__":
    main()
