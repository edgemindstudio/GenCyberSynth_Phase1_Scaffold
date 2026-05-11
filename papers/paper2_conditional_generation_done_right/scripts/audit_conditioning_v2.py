#!/usr/bin/env python3
"""
Paper 2 conditioning audit v2.

Purpose:
    Evaluate whether synthetic samples generated under requested labels are
    actually class-consistent according to a strong real-only CNN classifier.

Outputs:
    - conditioning confusion matrix CSV
    - per-class conditioning accuracy CSV
    - summary JSON
    - confusion heatmap
    - per-class accuracy bar chart
    - predicted-label histogram
    - saved real-only audit classifier weights

Run from repo root:
    python papers/paper2_conditional_generation_done_right/scripts/audit_conditioning_v2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


# ---------------------------------------------------------------------
# Repo-root and path setup
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]

if Path.cwd().resolve() != REPO_ROOT:
    os.chdir(REPO_ROOT)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------
# Imports after repo-root setup
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------
# Paper 2 constants
# ---------------------------------------------------------------------

PAPER_DIR = REPO_ROOT / "papers" / "paper2_conditional_generation_done_right"
RESULTS_DIR = PAPER_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

ARTIFACTS_ROOT = Path("/home/bruno.fonkeng/gencys/artifacts_paper2")
AUDIT_ARTIFACT_DIR = ARTIFACTS_ROOT / "audit" / "real_only_cnn"

DATA_DIR = Path("/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware_nhwc")
SYNTH_MANIFEST = ARTIFACTS_ROOT / "gan" / "synthetic" / "manifest.json"

SEED = 42
BOTTOM_K_MINORITY = 3
MAX_SYNTH_PER_CLASS = 25

np.random.seed(SEED)
tf.random.set_seed(SEED)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _load_npy_first_existing(candidates: List[Path]) -> np.ndarray:
    for path in candidates:
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(
        "None of the candidate files exist:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_real_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load real train/val/test splits.

    USTC anchor dataset layout:
        train_data.npy
        train_labels.npy
        test_data.npy
        test_labels.npy

    This dataset has no explicit validation split, so for the audit classifier
    only, we create a deterministic validation split from the real training set.
    This does not alter the dataset files or GAN training split.
    """

    ustc_train_x = data_dir / "train_data.npy"
    ustc_train_y = data_dir / "train_labels.npy"
    ustc_test_x = data_dir / "test_data.npy"
    ustc_test_y = data_dir / "test_labels.npy"

    if all(p.exists() for p in [ustc_train_x, ustc_train_y, ustc_test_x, ustc_test_y]):
        print("[audit-loader] Detected USTC anchor dataset layout.")
        print(f"[audit-loader] Loading: {ustc_train_x}")
        print(f"[audit-loader] Loading: {ustc_train_y}")
        print(f"[audit-loader] Loading: {ustc_test_x}")
        print(f"[audit-loader] Loading: {ustc_test_y}")

        X_train_full = np.load(ustc_train_x)
        y_train_full = np.load(ustc_train_y).astype("int64").reshape(-1)

        X_test = np.load(ustc_test_x)
        y_test = np.load(ustc_test_y).astype("int64").reshape(-1)

        rng = np.random.default_rng(SEED)

        train_indices = []
        val_indices = []

        # Stratified deterministic 80/20 split from original train only.
        for cls in sorted(np.unique(y_train_full)):
            cls_idx = np.where(y_train_full == cls)[0]
            rng.shuffle(cls_idx)

            n_val = max(1, int(round(0.20 * len(cls_idx))))
            val_indices.extend(cls_idx[:n_val].tolist())
            train_indices.extend(cls_idx[n_val:].tolist())

        train_indices = np.asarray(train_indices, dtype=np.int64)
        val_indices = np.asarray(val_indices, dtype=np.int64)

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        X_train = X_train_full[train_indices]
        y_train = y_train_full[train_indices]

        X_val = X_train_full[val_indices]
        y_val = y_train_full[val_indices]

        print(f"[audit-loader] X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"[audit-loader] X_val:   {X_val.shape}, y_val:   {y_val.shape}")
        print(f"[audit-loader] X_test:  {X_test.shape}, y_test:  {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    raise FileNotFoundError(
        "Could not find expected USTC anchor dataset files:\n"
        f"{ustc_train_x}\n"
        f"{ustc_train_y}\n"
        f"{ustc_test_x}\n"
        f"{ustc_test_y}"
    )


def normalize_images_for_classifier(X: np.ndarray) -> np.ndarray:
    """
    Make images classifier-friendly.

    Handles common ranges:
        [-1, 1] -> [0, 1]
        [0, 255] -> [0, 1]
        [0, 1] -> unchanged
    """

    X = X.astype("float32")

    if X.ndim == 3:
        X = X[..., None]

    xmin = float(np.min(X))
    xmax = float(np.max(X))

    if xmin >= -1.05 and xmax <= 1.05 and xmin < 0:
        X = (X + 1.0) / 2.0
    elif xmax > 1.5:
        X = X / 255.0

    return np.clip(X, 0.0, 1.0)


def build_real_only_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="paper2_real_only_audit_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_or_load_audit_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> Tuple[keras.Model, Dict[str, float]]:
    weights_path = AUDIT_ARTIFACT_DIR / f"seed{SEED}_real_only_cnn.weights.h5"

    model = build_real_only_cnn(tuple(X_train.shape[1:]), num_classes)

    if weights_path.exists():
        print(f"[audit] Loading existing real-only CNN weights: {weights_path}")
        model.load_weights(str(weights_path))
    else:
        print("[audit] Training real-only CNN classifier...")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
            )
        ]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=128,
            callbacks=callbacks,
            verbose=2,
        )

        model.save_weights(str(weights_path))
        print(f"[audit] Saved real-only CNN weights: {weights_path}")

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    metrics = {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "weights_path": str(weights_path),
    }

    return model, metrics


def _read_manifest_entries(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Synthetic manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if isinstance(manifest, list):
        return manifest

    # Paper 2 GAN manifest uses "paths".
    for key in ["paths", "samples", "items", "records", "files", "images"]:
        if key in manifest and isinstance(manifest[key], list):
            print(f"[audit-manifest] Using manifest key: {key}")
            print(f"[audit-manifest] Number of synthetic entries: {len(manifest[key])}")
            return manifest[key]

    raise ValueError(
        f"Could not identify sample list in manifest: {manifest_path}. "
        "Expected a list or a dict with one of: "
        "paths/samples/items/records/files/images."
    )

def _extract_path_and_label(
    entry: Dict[str, Any],
    manifest_dir: Path | None = None,
) -> Tuple[Path, int]:
    path_keys = ["path", "filepath", "file_path", "image_path", "npy_path", "sample_path"]
    label_keys = ["label", "class", "class_id", "requested_label", "y", "condition"]

    sample_path = None
    for key in path_keys:
        if key in entry:
            sample_path = Path(entry[key])
            break

    if sample_path is None:
        raise KeyError(f"Could not find sample path key in manifest entry: {entry}")

    if not sample_path.is_absolute():
        base_dir = manifest_dir if manifest_dir is not None else SYNTH_MANIFEST.parent

        # Some manifests store paths relative to the manifest directory:
        #   0/gan_00000.png
        # Others store paths relative to the synthetic root:
        #   paper2_acgan_smoke/seed42/0/gan_00000.png
        #
        # Try manifest-relative first, then synthetic-root-relative.
        candidate_manifest_relative = base_dir / sample_path

        synthetic_root = ARTIFACTS_ROOT / "gan" / "synthetic"
        candidate_synthetic_relative = synthetic_root / sample_path

        if candidate_manifest_relative.exists():
            sample_path = candidate_manifest_relative
        elif candidate_synthetic_relative.exists():
            sample_path = candidate_synthetic_relative
        else:
            # Keep manifest-relative path for the eventual FileNotFoundError.
            sample_path = candidate_manifest_relative

    label = None
    for key in label_keys:
        if key in entry:
            label = int(entry[key])
            break

    if label is None:
        raise KeyError(f"Could not find label key in manifest entry: {entry}")

    return sample_path, label


def load_synthetic_from_manifest(
    manifest_path: Path,
    max_per_class: int | None = MAX_SYNTH_PER_CLASS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load synthetic images and requested labels from manifest.

    This supports .npy sample paths. If your manifest points to PNGs instead,
    add a small image loader here.
    """

    entries = _read_manifest_entries(manifest_path)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    per_class_counts: Dict[int, int] = {}

    for entry in entries:
        sample_path, label = _extract_path_and_label(entry, manifest_path.parent)

        if max_per_class is not None:
            if per_class_counts.get(label, 0) >= max_per_class:
                continue

        if not sample_path.exists():
            raise FileNotFoundError(f"Synthetic sample file missing: {sample_path}")

        if sample_path.suffix.lower() == ".npy":
            img = np.load(sample_path)
        else:
            # PNG/JPG fallback
            raw = tf.keras.utils.load_img(sample_path, color_mode="grayscale")
            img = tf.keras.utils.img_to_array(raw)

        X_list.append(img)
        y_list.append(label)
        per_class_counts[label] = per_class_counts.get(label, 0) + 1

    if not X_list:
        raise RuntimeError(f"No synthetic samples loaded from manifest: {manifest_path}")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    return X, y


def compute_confusion(
    requested: np.ndarray,
    predicted: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for r, p in zip(requested, predicted):
        mat[int(r), int(p)] += 1
    return mat


def save_confusion_csv(confusion: np.ndarray, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["requested\\predicted"] + list(range(confusion.shape[1]))
        writer.writerow(header)
        for i, row in enumerate(confusion):
            writer.writerow([i] + list(map(int, row)))


def save_per_class_accuracy_csv(confusion: np.ndarray, path: Path) -> Dict[int, Dict[str, float]]:
    rows = {}
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "cond_accuracy", "count", "correct", "failure_rate"])

        for class_id in range(confusion.shape[0]):
            count = int(confusion[class_id].sum())
            correct = int(confusion[class_id, class_id])
            acc = float(correct / count) if count else 0.0
            fail = float(1.0 - acc) if count else 0.0

            writer.writerow([class_id, acc, count, correct, fail])

            rows[class_id] = {
                "cond_accuracy": acc,
                "count": count,
                "correct": correct,
                "failure_rate": fail,
            }

    return rows


def save_predicted_histogram_csv(predicted: np.ndarray, num_classes: int, path: Path) -> Dict[int, int]:
    counts = {i: int(np.sum(predicted == i)) for i in range(num_classes)}

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted_label", "count", "fraction"])
        total = max(1, len(predicted))
        for i in range(num_classes):
            writer.writerow([i, counts[i], counts[i] / total])

    return counts


def top_confusions_from_matrix(confusion: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]:
    rows = []

    for requested in range(confusion.shape[0]):
        for predicted in range(confusion.shape[1]):
            if requested == predicted:
                continue
            count = int(confusion[requested, predicted])
            if count > 0:
                total_for_requested = int(confusion[requested].sum())
                rate = count / total_for_requested if total_for_requested else 0.0
                rows.append({
                    "requested": int(requested),
                    "predicted": int(predicted),
                    "count": count,
                    "within_requested_rate": float(rate),
                })

    rows.sort(key=lambda d: d["count"], reverse=True)
    return rows[:top_n]


def plot_confusion_heatmap(confusion: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation="nearest")
    plt.title("Paper 2 GAN Conditioning Audit Confusion Matrix")
    plt.xlabel("Predicted label by real-only CNN")
    plt.ylabel("Requested synthetic label")
    plt.colorbar()

    classes = list(range(confusion.shape[0]))
    plt.xticks(classes, classes)
    plt.yticks(classes, classes)

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = int(confusion[i, j])
            if value > 0:
                plt.text(j, i, str(value), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_per_class_accuracy(per_class_rows: Dict[int, Dict[str, float]], path: Path) -> None:
    classes = sorted(per_class_rows.keys())
    values = [per_class_rows[c]["cond_accuracy"] for c in classes]

    plt.figure(figsize=(8, 4))
    plt.bar(classes, values)
    plt.ylim(0, 1.0)
    plt.xlabel("Requested class")
    plt.ylabel("Conditioning accuracy")
    plt.title("Paper 2 GAN Per-Class Conditioning Accuracy")
    plt.xticks(classes, classes)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_predicted_histogram(pred_counts: Dict[int, int], path: Path) -> None:
    classes = sorted(pred_counts.keys())
    values = [pred_counts[c] for c in classes]

    plt.figure(figsize=(8, 4))
    plt.bar(classes, values)
    plt.xlabel("Predicted label by real-only CNN")
    plt.ylabel("Synthetic sample count")
    plt.title("Paper 2 GAN Predicted-Label Histogram")
    plt.xticks(classes, classes)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paper 2 conditioning audit")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SYNTH_MANIFEST,
        help="Path to synthetic manifest.json to audit",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="v2",
        help="Output tag used in result filenames",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=MAX_SYNTH_PER_CLASS,
        help="Maximum synthetic samples to audit per requested class.",
    )

    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    audit_tag = str(args.tag).strip().replace("/", "_").replace(" ", "_")

    print(f"[paper2-audit-v2] Repo root: {REPO_ROOT}")
    print(f"[paper2-audit-v2] Data dir: {DATA_DIR}")
    print(f"[paper2-audit-v2] Synth manifest: {manifest_path}")
    print(f"[paper2-audit-v2] Audit tag: {audit_tag}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_real_dataset(DATA_DIR)

    X_train = normalize_images_for_classifier(X_train)
    X_val = normalize_images_for_classifier(X_val)
    X_test = normalize_images_for_classifier(X_test)

    y_train = y_train.astype("int64").reshape(-1)
    y_val = y_val.astype("int64").reshape(-1)
    y_test = y_test.astype("int64").reshape(-1)

    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

    print(f"[paper2-audit-v2] X_train shape: {X_train.shape}")
    print(f"[paper2-audit-v2] num_classes: {num_classes}")

    train_counts = {
        int(c): int(np.sum(y_train == c))
        for c in range(num_classes)
    }

    unique_train_counts = sorted(set(train_counts.values()))
    minority_tie_detected = len(unique_train_counts) == 1

    minority_classes = sorted(
        train_counts,
        key=lambda c: (train_counts[c], c)
    )[:BOTTOM_K_MINORITY]

    print(f"[paper2-audit-v2] Train class counts: {train_counts}")

    if minority_tie_detected:
        print(
            "[paper2-audit-v2] NOTE: Real train class counts are balanced. "
            "Bottom-k frequency-ranked analysis is tied and reported only as a deterministic diagnostic."
        )

    print(
        f"[paper2-audit-v2] Bottom-{BOTTOM_K_MINORITY} frequency-ranked classes: "
        f"{minority_classes}"
    )

    model, classifier_metrics = train_or_load_audit_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        num_classes,
    )

    max_synth_per_class = int(args.max_per_class)
    X_synth, y_requested = load_synthetic_from_manifest(manifest_path, max_synth_per_class)
    X_synth = normalize_images_for_classifier(X_synth)
    y_requested = y_requested.astype("int64").reshape(-1)

    probs = model.predict(X_synth, batch_size=128, verbose=0)
    y_pred = np.argmax(probs, axis=1).astype("int64")

    confusion = compute_confusion(y_requested, y_pred, num_classes)

    overall_correct = int(np.sum(y_requested == y_pred))
    total = int(len(y_requested))
    overall_acc = float(overall_correct / total) if total else 0.0
    overall_failure_rate = float(1.0 - overall_acc)

    leakage_count = int(np.sum(y_requested != y_pred))
    leakage_rate = float(leakage_count / total) if total else 0.0

    minority_mask = np.isin(y_requested, np.asarray(minority_classes))
    minority_total = int(np.sum(minority_mask))
    minority_correct = int(np.sum(y_requested[minority_mask] == y_pred[minority_mask]))
    minority_acc = float(minority_correct / minority_total) if minority_total else 0.0
    minority_failure_rate = float(1.0 - minority_acc) if minority_total else 0.0

    # Save tables
    confusion_csv = TABLES_DIR / f"paper2_gan_conditioning_confusion_{audit_tag}.csv"
    per_class_csv = TABLES_DIR / f"paper2_gan_conditioning_accuracy_{audit_tag}.csv"
    pred_hist_csv = TABLES_DIR / f"paper2_gan_predicted_label_histogram_{audit_tag}.csv"
    summary_json = TABLES_DIR / f"paper2_gan_conditioning_summary_{audit_tag}.json"

    save_confusion_csv(confusion, confusion_csv)
    per_class_rows = save_per_class_accuracy_csv(confusion, per_class_csv)
    pred_counts = save_predicted_histogram_csv(y_pred, num_classes, pred_hist_csv)
    top_confusions = top_confusions_from_matrix(confusion, top_n=10)

    summary = {
        "paper_id": "paper2",
        "model_family": "gan",
        "audit_version": "conditioning_audit_v2",
        "audit_tag": audit_tag,
        "seed": SEED,
        "data_dir": str(DATA_DIR),
        "artifacts_root": str(ARTIFACTS_ROOT),
        "synth_manifest": str(manifest_path),
        "num_classes": num_classes,
        "max_synth_per_class": max_synth_per_class,
        "total_synthetic_audited": total,
        "overall_conditioning_accuracy": overall_acc,
        "overall_conditioning_failure_rate": overall_failure_rate,
        "leakage_count": leakage_count,
        "leakage_rate": leakage_rate,
        "train_class_counts": train_counts,
        "minority_definition": {
            "method": "bottom_k_by_real_train_count",
            "k": BOTTOM_K_MINORITY,
            "minority_classes": minority_classes,
            "minority_tie_detected": minority_tie_detected,
            "note": (
                "Real train class counts are balanced; bottom-k frequency analysis "
                "is tied and should be interpreted as a deterministic diagnostic."
                if minority_tie_detected
                else "Bottom-k classes are defined by real train frequency."
            ),
        },
        "minority_conditioning_accuracy": minority_acc,
        "minority_conditioning_failure_rate": minority_failure_rate,
        "minority_total": minority_total,
        "minority_correct": minority_correct,
        "predicted_label_histogram": pred_counts,
        "top_confusions": top_confusions,
        "real_only_classifier": classifier_metrics,
        "outputs": {
            "confusion_csv": str(confusion_csv),
            "per_class_accuracy_csv": str(per_class_csv),
            "predicted_histogram_csv": str(pred_hist_csv),
        },
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Save figures
    confusion_png = FIGURES_DIR / f"paper2_gan_conditioning_confusion_{audit_tag}.png"
    per_class_png = FIGURES_DIR / f"paper2_gan_conditioning_accuracy_{audit_tag}.png"
    pred_hist_png = FIGURES_DIR / f"paper2_gan_predicted_label_histogram_{audit_tag}.png"

    plot_confusion_heatmap(confusion, confusion_png)
    plot_per_class_accuracy(per_class_rows, per_class_png)
    plot_predicted_histogram(pred_counts, pred_hist_png)

    print("[paper2-audit-v2] DONE")
    print(f"[paper2-audit-v2] Overall conditioning accuracy: {overall_acc:.4f}")
    print(f"[paper2-audit-v2] Leakage rate: {leakage_rate:.4f}")
    print(f"[paper2-audit-v2] Minority failure rate: {minority_failure_rate:.4f}")
    print(f"[paper2-audit-v2] Summary: {summary_json}")


if __name__ == "__main__":
    main()