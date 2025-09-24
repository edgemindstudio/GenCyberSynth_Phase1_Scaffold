# common/data.py
# =============================================================================
# Unified dataset utilities for the GenCyberSynth scaffold.
#
# What this module guarantees:
# - Loads four .npy files from a data directory:
#       train_data.npy, train_labels.npy, test_data.npy, test_labels.npy
# - Images returned in float32 range [0, 1], channels-last (N, H, W, C)
# - Labels returned as float32 one-hot of shape (N, K)
# - Provided test set is split into (val, test) by `val_fraction`
# - Robust shape handling:
#       * (N, H, W, C) -> kept
#       * (N, H, W) with C==1 -> channel added
#       * (N, H*W*C) -> reshaped
#       * (N, C, H, W) (NCHW) -> transposed to NHWC
#
# Public API (stable):
#   - load_dataset_npy(...)
#   - load_for_training_01(...)
#   - load_for_training_minus1_1(...)
#   - to_01(...), to_minus1_1(...), to_01_from_minus1_1(...)
#   - make_tf_dataset(...)
#   - one_hot(...), describe_labels(...)
#
# Keep this file the single source of truth for data loading/normalization so
# all model packages behave consistently.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Basic range conversions
# -----------------------------------------------------------------------------
def to_01(x: np.ndarray) -> np.ndarray:
    """
    Ensure images are float32 in [0, 1]. If data appears to be 0..255, scale.
    NaNs (if any) are preserved except for clipping.
    """
    x = x.astype("float32", copy=False)
    # Heuristic: values > 1.5 suggest 0..255 byte data
    if np.nanmax(x) > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def to_minus1_1(x01: np.ndarray) -> np.ndarray:
    """Map images from [0,1] → [-1,1] (useful for tanh decoders / some GANs)."""
    return np.clip((x01 - 0.5) * 2.0, -1.0, 1.0)


def to_01_from_minus1_1(xm11: np.ndarray) -> np.ndarray:
    """Map images from [-1,1] → [0,1]."""
    return np.clip((xm11 + 1.0) / 2.0, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Shape handling (NHWC as canonical)
# -----------------------------------------------------------------------------
def _reshape_to_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Coerce `x` to (N, H, W, C) given img_shape=(H,W,C).

    Acceptable inputs:
      - (N, H, W, C) -> unchanged (validated)
      - (N, H, W)    -> add channel axis if C == 1
      - (N, H*W*C)   -> reshape to (N, H, W, C)
      - (N, C, H, W) -> transpose to (N, H, W, C)  <-- NEW
    """
    H, W, C = img_shape

    # (N,H,W,C)
    if x.ndim == 4 and tuple(x.shape[1:]) == (H, W, C):
        return x.astype("float32", copy=False)

    # (N,H,W) and C==1
    if x.ndim == 3 and x.shape[1:] == (H, W) and C == 1:
        return x.astype("float32", copy=False)[..., None]

    # (N,H*W*C)
    if x.ndim == 2 and x.shape[1] == H * W * C:
        return x.astype("float32", copy=False).reshape((-1, H, W, C))

    # (N,C,H,W)  <-- handle channels-first
    if x.ndim == 4 and x.shape[1:] == (C, H, W):
        x = x.astype("float32", copy=False)
        return np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

    # More permissive check for NCHW where only C matches and spatial matches
    if x.ndim == 4 and x.shape[1] == C and x.shape[2] == H and x.shape[3] == W:
        x = x.astype("float32", copy=False)
        return np.transpose(x, (0, 2, 3, 1))

    raise ValueError(
        f"Cannot coerce array of shape {x.shape} to (N,{H},{W},{C}). "
        "Expected (N,H,W,C), (N,H,W) with C=1, (N,H*W*C), or (N,C,H,W)."
    )



# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Idempotent one-hot:
      - If y already has shape (N, K) with K==num_classes, returns as float32.
      - Otherwise, treats y as integer class ids and converts.
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32", copy=False)
    return tf.keras.utils.to_categorical(y.astype("int64").ravel(), num_classes).astype("float32")


def describe_labels(y_oh: np.ndarray) -> Dict[int, int]:
    """Quick histogram {class_id: count} from one-hot labels."""
    if y_oh.ndim != 2:
        raise ValueError(f"Expected one-hot labels of shape (N, K), got {y_oh.shape}")
    ids = np.argmax(y_oh, axis=1)
    vals, cnts = np.unique(ids, return_counts=True)
    return {int(k): int(v) for k, v in zip(vals, cnts)}


# -----------------------------------------------------------------------------
# Core loader
# -----------------------------------------------------------------------------
def _load_required_files(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load four required .npy files from `data_dir`. Raises a clear error if any are missing.
    """
    data_dir = Path(data_dir)
    req = ["train_data.npy", "train_labels.npy", "test_data.npy", "test_labels.npy"]
    missing = [p for p in req if not (data_dir / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required dataset files under {data_dir}: {missing}. "
            "Expected: train_data.npy, train_labels.npy, test_data.npy, test_labels.npy"
        )

    x_train = np.load(data_dir / "train_data.npy", allow_pickle=False)
    y_train = np.load(data_dir / "train_labels.npy", allow_pickle=False)
    x_test  = np.load(data_dir / "test_data.npy",  allow_pickle=False)
    y_test  = np.load(data_dir / "test_labels.npy", allow_pickle=False)
    return x_train, y_train, x_test, y_test


def load_dataset_npy(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from four .npy files in `data_dir`, normalize to [0,1], coerce to NHWC,
    convert labels to one-hot, and split test → (val, test).

    Returns:
      x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh   (all float32)

    Notes:
      - `val_fraction` must be in (0,1). Typical: 0.5 to split test evenly.
      - This function does NOT shuffle; keep splits deterministic.
    """
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

    H, W, C = img_shape
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = _load_required_files(Path(data_dir))

    # Normalize → [0,1]
    x_train01 = to_01(x_train_raw)
    x_test01  = to_01(x_test_raw)

    # Shape to NHWC (handles NHWC, HWC (C==1), flat, or NCHW)
    x_train01 = _reshape_to_hwc(x_train01, (H, W, C))
    x_test01  = _reshape_to_hwc(x_test01,  (H, W, C))

    # Labels → one-hot
    y_train_oh = one_hot(y_train_raw, num_classes)
    y_test_oh  = one_hot(y_test_raw,  num_classes)

    # Split test → (val, test)
    n_val = int(round(len(x_test01) * float(val_fraction)))
    x_val01, y_val_oh = x_test01[:n_val], y_test_oh[:n_val]
    x_test01, y_test_oh = x_test01[n_val:], y_test_oh[n_val:]

    return x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh


# -----------------------------------------------------------------------------
# tf.data helper (optional but convenient for trainers)
# -----------------------------------------------------------------------------
def make_tf_dataset(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    shuffle_buffer: int = 10240,
    cache: bool = False,
    drop_remainder: bool = False,
    prefetch: bool = True,
    augment: Optional[callable] = None,
) -> tf.data.Dataset:
    """
    Build a performant tf.data pipeline from (x[, y]) arrays.

    - If `y` is provided, yields (x, y); else yields x.
    - `augment` can be fn(x)->x' or fn(x,y)->(x',y').
    - This function does NOT change numeric range; pre-process x beforehand.

    Example:
        x01, y1h, xv01, yv1h, _, _ = load_dataset_npy(...)
        ds = make_tf_dataset(to_minus1_1(x01), y1h, batch_size=256)
    """
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, y))

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    if augment is not None:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# -----------------------------------------------------------------------------
# Convenience wrappers for common training ranges
# -----------------------------------------------------------------------------
def load_for_training_01(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
):
    """
    Shorthand for evaluators / models that expect inputs in [0,1].
    Returns:
      x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh
    """
    return load_dataset_npy(data_dir, img_shape, num_classes, val_fraction)


def load_for_training_minus1_1(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
):
    """
    Shorthand for models that expect inputs in [-1,1] (tanh decoders, many GANs).
    Returns:
      x_train_m11, y_train_oh, x_val_m11, y_val_oh, x_test_m11, y_test_oh
    """
    x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction
    )
    return (
        to_minus1_1(x_train01),
        y_train_oh,
        to_minus1_1(x_val01),
        y_val_oh,
        to_minus1_1(x_test01),
        y_test_oh,
    )
