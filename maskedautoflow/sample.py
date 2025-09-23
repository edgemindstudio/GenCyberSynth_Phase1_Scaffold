# maskedautoflow/sample.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image

from maskedautoflow.models import (
    MAF,
    MAFConfig,
    build_maf_model,
    reshape_to_images,
)

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = img01
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        mode = "RGB"
    else:
        x = x.squeeze()
        mode = "L"
    Image.fromarray(_to_uint8(x), mode=mode).save(out_path)


def _cfg_get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------
def load_maf_from_checkpoints(
    ckpt_dir: Path | str,
    img_shape: Tuple[int, int, int],
    num_flows: int = 5,
    hidden_dims: Tuple[int, ...] = (128, 128),
) -> MAF:
    """
    Instantiate a MAF and load weights from best→last checkpoint.

    Raises
    ------
    FileNotFoundError if no checkpoint exists.
    """
    ckpt_dir = Path(ckpt_dir)
    H, W, C = img_shape
    D = H * W * C

    model = build_maf_model(
        MAFConfig(IMG_SHAPE=img_shape, NUM_FLOWS=num_flows, HIDDEN_DIMS=hidden_dims)
    )
    # Build variables
    _ = model(tf.zeros((1, D), dtype=tf.float32))

    best = ckpt_dir / "MAF_best.weights.h5"
    last = ckpt_dir / "MAF_last.weights.h5"
    to_load = best if best.exists() else last
    if not to_load.exists():
        raise FileNotFoundError(f"No MAF checkpoint found under {ckpt_dir}")
    model.load_weights(str(to_load))
    print(f"[ckpt] Loaded {to_load.name}")
    return model


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------
def sample_unconditional(
    model: MAF,
    n_total: int,
    img_shape: Tuple[int, int, int],
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Draw unconditional samples and reshape to images in [0,1].

    Returns
    -------
    x : float32, shape (n_total, H, W, C) in [0,1]
    """
    H, W, C = img_shape
    D = H * W * C
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    z = tf.random.normal(shape=(n_total, D), dtype=tf.float32, seed=seed)
    x_flat = model.inverse(z).numpy().astype(np.float32)
    x_flat = np.clip(x_flat, 0.0, 1.0)
    return reshape_to_images(x_flat, img_shape, clip=True)


def write_balanced_per_class(
    x: np.ndarray,
    num_classes: int,
    per_class: int,
    synth_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evenly split unconditional samples across classes and write evaluator files.

    Files written
    -------------
    synth_dir/
      gen_class_{k}.npy
      labels_class_{k}.npy
      x_synth.npy
      y_synth.npy

    Returns
    -------
    x_synth, y_synth (one-hot)
    """
    _ensure_dir(Path(synth_dir))
    xs, ys = [], []
    for k in range(num_classes):
        start, end = k * per_class, (k + 1) * per_class
        xk = x[start:end]
        np.save(Path(synth_dir) / f"gen_class_{k}.npy", xk)
        np.save(Path(synth_dir) / f"labels_class_{k}.npy", np.full((len(xk),), k, dtype=np.int32))
        xs.append(xk)
        y1h = np.zeros((len(xk), num_classes), dtype=np.float32)
        y1h[:, k] = 1.0
        ys.append(y1h)

    x_synth = np.concatenate(xs, axis=0).astype(np.float32)
    y_synth = np.concatenate(ys, axis=0).astype(np.float32)

    np.save(Path(synth_dir) / "x_synth.npy", x_synth)
    np.save(Path(synth_dir) / "y_synth.npy", y_synth)
    print(f"[synthesize] {x_synth.shape[0]} samples ({per_class} per class) -> {synth_dir}")
    return x_synth, y_synth


def save_preview_row_png(x: np.ndarray, out_path: Path) -> None:
    """
    Save a single-row preview grid (one image per class) to PNG.

    Args
    ----
    x: (K,H,W,C) float in [0,1]
    """
    import matplotlib.pyplot as plt

    k = x.shape[0]
    x_vis = x[..., 0] if x.shape[-1] == 1 else x
    fig, axes = plt.subplots(1, k, figsize=(1.4 * k, 1.6))
    if k == 1:
        axes = [axes]
    for i in range(k):
        axes[i].imshow(x_vis[i], cmap="gray" if x.shape[-1] == 1 else None, vmin=0.0, vmax=1.0)
        axes[i].set_axis_off()
        axes[i].set_title(f"C{i}", fontsize=9)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Unified CLI synth
# ---------------------------------------------------------------------
def synth(cfg: dict, output_root: str, seed: int = 42) -> Dict:
    """
    Generate S PNGs/class into {output_root}/{class}/{seed}/... and return a manifest.

    Expects checkpoints at:
      artifacts/maskedautoflow/checkpoints/MAF_best.weights.h5 (or MAF_last.weights.h5)
    """
    # Resolve shape/classes/count (support both NEW and LEGACY keys)
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))
    num_flows = int(_cfg_get(cfg, "NUM_FLOWS", 5))
    hidden_dims = tuple(int(h) for h in _cfg_get(cfg, "HIDDEN_DIMS", (128, 128)))

    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    ckpt_dir = Path(_cfg_get(cfg, "ARTIFACTS.maskedautoflow_checkpoints",
                             artifacts_root / "maskedautoflow" / "checkpoints"))
    sums_dir = Path(_cfg_get(cfg, "ARTIFACTS.maskedautoflow_summaries",
                             artifacts_root / "maskedautoflow" / "summaries"))

    # Load model & sample
    model = load_maf_from_checkpoints(ckpt_dir, (H, W, C), num_flows=num_flows, hidden_dims=hidden_dims)
    x = sample_unconditional(model, n_total=K * S, img_shape=(H, W, C), seed=seed)

    # Save a tiny preview (1 per class)
    try:
        save_preview_row_png(x[:K], sums_dir / "maf_synth_preview.png")
    except Exception as e:
        print(f"[warn] preview failed: {e}")

    # Write per-class PNGs and build manifest
    out_root = Path(output_root)
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict] = []

    for k in range(K):
        start, end = k * S, (k + 1) * S
        cls_imgs = x[start:end]  # (S,H,W,C)
        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j in range(cls_imgs.shape[0]):
            p = cls_dir / f"maf_{j:05d}.png"
            _save_png(cls_imgs[j], p)
            paths.append({"path": str(p), "label": int(k)})
        per_class_counts[str(k)] = int(cls_imgs.shape[0])

    manifest = {
        "dataset": _cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return manifest


__all__ = [
    "load_maf_from_checkpoints",
    "sample_unconditional",
    "write_balanced_per_class",
    "save_preview_row_png",
    "synth",
]
