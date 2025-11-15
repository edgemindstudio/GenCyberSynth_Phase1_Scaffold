# vae/train.py
"""
Train the Conditional VAE (cVAE) and save checkpoints + a small preview grid.

Usage
-----
# From repo root (or any directory), run:
python -m vae.train --config configs/config.yaml

What this script does
---------------------
1) Reads a YAML config (see keys below).
2) Loads dataset from four .npy files under DATA_DIR:
   - train_data.npy, train_labels.npy, test_data.npy, test_labels.npy
3) Converts inputs for training (decoder is tanh → expects [-1, 1]).
4) Trains the VAE via `VAEPipeline(cfg)` with TensorBoard-friendly logging.
5) Saves a 1×N preview grid from the decoder to ARTIFACTS.summaries and
   checkpoints to ${paths.artifacts}/vae/checkpoints.

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
paths:
  artifacts: "artifacts"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
import yaml

from common.data import load_dataset_npy, to_minus1_1

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


def _cfg_get(cfg: Dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _normalize_artifacts(cfg: Dict) -> Dict:
    """
    Fill ARTIFACTS with sensible defaults, honoring paths.artifacts if present.
    """
    arts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    cfg.setdefault("ARTIFACTS", {})
    A = cfg["ARTIFACTS"]
    A.setdefault("checkpoints", str(arts_root / "vae" / "checkpoints"))
    A.setdefault("synthetic",   str(arts_root / "vae" / "synthetic"))
    A.setdefault("summaries",   str(arts_root / "vae" / "summaries"))
    A.setdefault("tensorboard", str(arts_root / "tensorboard"))
    return cfg


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
# Core training runner
# ---------------------------
def _run_train(cfg: Dict) -> Dict[str, str]:
    # be nice to GPUs; also harmless on CPU nodes
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

    # Required basics
    data_dir    = Path(cfg.get("DATA_DIR", _cfg_get(cfg, "data.root", "data")))
    img_shape   = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    num_classes = int(cfg.get("NUM_CLASSES", cfg.get("num_classes", 9)))

    print(f"[config] DATA_DIR={data_dir} | IMG_SHAPE={img_shape} | NUM_CLASSES={num_classes}")
    print(f"[paths]  checkpoints={cfg['ARTIFACTS']['checkpoints']} | summaries={cfg['ARTIFACTS']['summaries']} | tb={cfg['ARTIFACTS']['tensorboard']}")

    # Load dataset in [0,1]; split test → (val, test)
    x_train01, y_train, x_val01, y_val, _, _ = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    # Map to [-1,1] for tanh decoder
    x_train_m11 = to_minus1_1(x_train01)
    x_val_m11   = to_minus1_1(x_val01)

    # TensorBoard callback
    cfg["LOG_CB"] = make_log_cb(Path(cfg["ARTIFACTS"]["tensorboard"]))

    # Train via pipeline
    pipe = VAEPipeline(cfg)
    enc, dec = pipe.train(
        x_train=x_train_m11, y_train=y_train,
        x_val=x_val_m11,     y_val=y_val,
    )

    # Ensure checkpoints exist (even if pipeline already saves)
    ckpt_dir = Path(cfg["ARTIFACTS"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        enc.save_weights(str(ckpt_dir / "VAE_encoder_last.weights.h5"))
        dec.save_weights(str(ckpt_dir / "VAE_decoder_last.weights.h5"))
    except Exception:
        # Some model wrappers may not expose .save_weights cleanly; ignore
        pass

    # Save a small decoder preview grid
    preview_path = save_grid(
        dec, num_classes, cfg["LATENT_DIM"],
        n=min(9, num_classes),
        path=Path(cfg["ARTIFACTS"]["summaries"]) / "train_preview.png",
    )
    print(f"[vae] saved preview grid → {preview_path}")

    return {
        "preview": str(preview_path),
        "checkpoints": str(ckpt_dir),
    }


# ---------------------------
# CLI and unified entrypoints
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Conditional VAE (cVAE)")
    p.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    return p.parse_args()


def main(argv=None) -> int:
    """
    Accepts either:
      • argv list (e.g., ['--config','configs/config.yaml']) — used by app.main
      • dict (already-parsed config) — also accepted for flexibility
    Returns 0 on success (so app.main treats it as OK).
    """
    # GPU memory growth hint (safe on CPU)
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # If app.main passed a dict (fallback path), honor it
    if isinstance(argv, dict):
        _run_train(argv)
        return 0

    # Else argv-style parsing
    if argv is None:
        args = parse_args()
        cfg_path = Path(args.config)
    else:
        p = argparse.ArgumentParser(description="Train Conditional VAE (cVAE)")
        p.add_argument("--config", default="configs/config.yaml")
        args = p.parse_args(argv)
        cfg_path = Path(args.config)

    cfg = load_yaml(cfg_path)
    print(f"[config] Using {cfg_path.resolve()}")
    _run_train(cfg)
    return 0


def train(cfg: Dict) -> int:
    """
    Direct dict-style entrypoint so app.main can call train(config_dict).
    Returns 0 on success.
    """
    _run_train(cfg)
    return 0


__all__ = [
    "main",
    "train",
]
if __name__ == "__main__":
    raise SystemExit(main())
