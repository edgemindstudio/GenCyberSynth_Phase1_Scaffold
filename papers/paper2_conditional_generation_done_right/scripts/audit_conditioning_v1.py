#!/usr/bin/env python3
"""
Paper 2 - Conditioning Audit v1
- Trains a simple CNN classifier on REAL train split.
- Predicts labels on SYNTH samples from manifest.json.
- Produces confusion matrix + per-class conditioning accuracy.

Outputs:
papers/paper2_conditional_generation_done_right/results/tables/
papers/paper2_conditional_generation_done_right/results/figures/
"""

from __future__ import annotations
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple
import csv

import tensorflow as tf
import matplotlib.pyplot as plt

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../GenCyberSynth_Phase1_Scaffold
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gcs-core"))

from common.data import load_dataset_npy  # already used in gan/train.py

PAPER_DIR = Path("papers/paper2_conditional_generation_done_right")
OUT_TBL = PAPER_DIR / "results" / "tables"
OUT_FIG = PAPER_DIR / "results" / "figures"
OUT_TBL.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

DEFAULT_CFG = PAPER_DIR / "configs" / "paper2_smoke.yaml"
DEFAULT_ARTS = Path("/home/bruno.fonkeng/gencys/artifacts_paper2")
DEFAULT_MODEL = "gan"

def load_cfg(p: Path) -> Dict[str, Any]:
    with p.open("r") as f:
        return yaml.safe_load(f) or {}

def load_manifest(arts_root: Path, model: str) -> Dict[str, Any]:
    mp = arts_root / model / "synthetic" / "manifest.json"
    if not mp.exists():
        raise FileNotFoundError(f"Missing manifest: {mp}")
    with mp.open("r") as f:
        return json.load(f)

def build_cnn(img_shape: Tuple[int,int,int], num_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=img_shape)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

def read_image(path: str, img_shape: Tuple[int,int,int]) -> np.ndarray:
    # Manifest entries can be relative or absolute; handle both.
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p

    raw = tf.io.read_file(str(p))
    img = tf.image.decode_png(raw, channels=img_shape[2])
    img = tf.image.resize(img, [img_shape[0], img_shape[1]], method="nearest")
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()

def main(cfg_path: Path = DEFAULT_CFG, arts_root: Path = DEFAULT_ARTS, model: str = DEFAULT_MODEL) -> int:
    cfg = load_cfg(cfg_path)

    # The GAN trainer uses top-level DATA_DIR; we reuse it here.
    data_dir = cfg.get("DATA_DIR")
    if not data_dir:
        raise ValueError("Config missing DATA_DIR (top-level). Add DATA_DIR: /path/to/USTC...")

    img_shape = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    num_classes = int(cfg.get("NUM_CLASSES", 9))

    # Load REAL data (returns [0,1] and one-hot labels)
    x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh = load_dataset_npy(
        Path(data_dir), img_shape, num_classes, val_fraction=float(cfg.get("VAL_FRACTION", 0.5))
    )
    y_train = np.argmax(y_train_oh, axis=1).astype(np.int32)
    y_val = np.argmax(y_val_oh, axis=1).astype(np.int32)

    # Train quick classifier (deterministic-ish)
    tf.random.set_seed(42)
    np.random.seed(42)
    clf = build_cnn(img_shape, num_classes)
    clf.fit(x_train01, y_train, validation_data=(x_val01, y_val),
            epochs=3, batch_size=256, verbose=2)

    # Load SYNTH from manifest
    manifest = load_manifest(arts_root, model)
    items = manifest.get("paths", [])
    if not items:
        raise ValueError("Manifest has no 'paths' entries.")

    Xs = np.zeros((len(items), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    ys_req = np.zeros((len(items),), dtype=np.int32)

    for i, it in enumerate(items):
        ys_req[i] = int(it["label"])
        Xs[i] = read_image(it["path"], img_shape)

    probs = clf.predict(Xs, batch_size=256, verbose=0)
    ys_pred = np.argmax(probs, axis=1).astype(np.int32)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for a, b in zip(ys_req, ys_pred):
        cm[a, b] += 1

    # Per-class conditioning accuracy
    per_acc = []
    for c in range(num_classes):
        total = cm[c].sum()
        acc = (cm[c, c] / total) if total > 0 else np.nan
        per_acc.append(acc)

    overall = (ys_req == ys_pred).mean()

    # Save tables
    out_cm = OUT_TBL / f"paper2_{model}_conditioning_confusion.csv"
    with out_cm.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["requested\\predicted"] + [str(i) for i in range(num_classes)])
        for i in range(num_classes):
            w.writerow([str(i)] + cm[i].tolist())

    out_acc = OUT_TBL / f"paper2_{model}_conditioning_accuracy.csv"
    with out_acc.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "cond_accuracy", "count"])
        for i in range(num_classes):
            w.writerow([i, per_acc[i], int(cm[i].sum())])

    out_json = OUT_TBL / f"paper2_{model}_conditioning_summary.json"
    out_json.write_text(json.dumps({
        "overall_conditioning_accuracy": float(overall),
        "per_class_accuracy": [None if np.isnan(x) else float(x) for x in per_acc],
        "counts_per_class": [int(cm[i].sum()) for i in range(num_classes)]
    }, indent=2))

    # Save confusion figure
    fig_path = OUT_FIG / f"paper2_{model}_conditioning_confusion.png"
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Requested label vs Predicted label (Confusion)")
    plt.xlabel("Predicted")
    plt.ylabel("Requested")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[ok] overall conditioning acc = {overall:.4f}")
    print(f"[ok] wrote: {out_cm}")
    print(f"[ok] wrote: {out_acc}")
    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {fig_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())