#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


REPO_ROOT = Path("/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold")
ARTS = Path("/home/bruno.fonkeng/gencys/artifacts_paper3")

DEFAULT_DATA_CANDIDATES = [
    Path("/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware_nhwc"),
    Path("/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware"),
]

OUT_RAW = REPO_ROOT / "papers/paper3_when_does_synth_help/results/raw/paper3_c4c7_synthetic_alignment_raw.csv"
OUT_HIST = REPO_ROOT / "papers/paper3_when_does_synth_help/results/raw/paper3_c4c7_synthetic_alignment_pred_hist.csv"
OUT_AGG = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_aggregate_frozen_20260524.csv"
OUT_MD = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_table.md"
OUT_TEX = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_table.tex"
OUT_SUMMARY = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_summary_20260524.json"

AUDIT_DIR = ARTS / "audit" / "paper3_c4c7_real_only_cnn"
WEIGHTS_PATH = AUDIT_DIR / "seed42_c4c7_minority_heavy_real_only_cnn.weights.h5"
CLASSIFIER_METRICS_PATH = AUDIT_DIR / "seed42_c4c7_minority_heavy_real_only_cnn_metrics.json"

MINORITY_CLASSES = [4, 7]
MINORITY_FRACTION = 0.2
AUDIT_SEED = 42


def find_data_dir() -> Path:
    for d in DEFAULT_DATA_CANDIDATES:
        if (d / "train_data.npy").exists() and (d / "train_labels.npy").exists() and (d / "test_data.npy").exists() and (d / "test_labels.npy").exists():
            return d
    raise FileNotFoundError("Could not find USTC-TFC2016 train/test npy files in expected data directories.")


def to_nhwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[..., None]
    if x.ndim == 4 and x.shape[1] == 1 and x.shape[-1] != 1:
        x = np.transpose(x, (0, 2, 3, 1))
    return x


def normalize_images(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    x = to_nhwc(x)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmin >= -1.05 and xmax <= 1.05 and xmin < 0:
        x = (x + 1.0) / 2.0
    elif xmax > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0).astype("float32")


def load_real_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy").astype("int64").reshape(-1)
    x_test = np.load(data_dir / "test_data.npy")
    y_test = np.load(data_dir / "test_labels.npy").astype("int64").reshape(-1)

    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    return x_train, y_train, x_test, y_test


def apply_minority_subsample(x: np.ndarray, y: np.ndarray, seed: int = AUDIT_SEED) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keep_indices = []
    before_counts = {str(c): int(np.sum(y == c)) for c in sorted(np.unique(y))}

    for c in sorted(np.unique(y)):
        idx = np.where(y == c)[0]
        if int(c) in MINORITY_CLASSES:
            n_keep = max(1, int(round(len(idx) * MINORITY_FRACTION)))
            idx = rng.choice(idx, size=n_keep, replace=False)
        keep_indices.extend(idx.tolist())

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    rng.shuffle(keep_indices)

    x_sub = x[keep_indices]
    y_sub = y[keep_indices]
    after_counts = {str(c): int(np.sum(y_sub == c)) for c in sorted(np.unique(y))}

    meta = {
        "minority_classes": MINORITY_CLASSES,
        "minority_fraction": MINORITY_FRACTION,
        "seed": seed,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "num_train_before": int(len(y)),
        "num_train_after": int(len(y_sub)),
    }
    return x_sub, y_sub, meta


def stratified_train_val_split(x: np.ndarray, y: np.ndarray, val_fraction: float = 0.1, seed: int = AUDIT_SEED):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for c in sorted(np.unique(y)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def build_audit_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="paper3_c4c7_real_only_audit_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_or_load_classifier(x_train, y_train, x_val, y_val, x_test, y_test, force_train: bool = False):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    model = build_audit_cnn(tuple(x_train.shape[1:]), num_classes)

    if WEIGHTS_PATH.exists() and not force_train:
        print(f"[audit] Loading audit classifier weights: {WEIGHTS_PATH}")
        model.load_weights(str(WEIGHTS_PATH))
    else:
        print("[audit] Training Paper 3 c4/c7 real-only audit classifier...")
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
        ]
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=30,
            batch_size=128,
            callbacks=callbacks,
            verbose=2,
        )
        model.save_weights(str(WEIGHTS_PATH))
        print(f"[audit] Saved weights: {WEIGHTS_PATH}")

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    metrics = {
        "weights_path": str(WEIGHTS_PATH),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }
    CLASSIFIER_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return model, metrics, num_classes


def read_manifest_entries(manifest_path: Path) -> List[Dict[str, Any]]:
    m = json.load(open(manifest_path))
    for key in ["paths", "samples", "items", "records", "files", "images"]:
        if isinstance(m.get(key), list):
            return m[key]
    raise ValueError(f"No sample list found in manifest: {manifest_path}")


def resolve_path(path_value: str, manifest_path: Path) -> Path:

    p = Path(path_value)

    if p.is_absolute():

        return p



    candidates = []



    # 1) Relative to the manifest directory.

    candidates.append(manifest_path.parent / p)



    # 2) Relative to the family synthetic root.

    # Example stored path:

    #   paper3_gan_aug_minority_heavy_c4c7_b100/seed42/4/gan_00000.png

    # Correct full path:

    #   /home/.../artifacts_paper3/gan/synthetic/paper3_gan_aug_minority_heavy_c4c7_b100/seed42/4/gan_00000.png

    try:

        candidates.append(manifest_path.parent.parent.parent / p)

    except Exception:

        pass



    # 3) Relative to each known Paper 3 family synthetic root.

    for fam in ["gan", "vae", "diffusion"]:

        candidates.append(ARTS / fam / "synthetic" / p)



    # 4) Relative to Paper 3 artifact root.

    candidates.append(ARTS / p)



    for candidate in candidates:

        if candidate.exists():

            return candidate



    # Return raw path so FileNotFoundError is informative.

    return p
def load_synthetic_from_manifest(manifest_path: Path, max_per_class: int | None = None):
    entries = read_manifest_entries(manifest_path)
    x_list = []
    y_list = []
    counts: Dict[int, int] = {}

    for e in entries:
        label = int(e.get("label", e.get("class_id", e.get("requested_label", -1))))
        if label not in MINORITY_CLASSES:
            continue
        if max_per_class is not None and counts.get(label, 0) >= max_per_class:
            continue

        path_value = e.get("path", e.get("file_path", e.get("image_path", None)))
        if path_value is None:
            raise KeyError(f"No path key in manifest entry: {e}")

        p = resolve_path(path_value, manifest_path)
        if not p.exists():
            raise FileNotFoundError(p)

        if p.suffix.lower() == ".npy":
            img = np.load(p)
        else:
            raw = tf.keras.utils.load_img(p, color_mode="grayscale")
            img = tf.keras.utils.img_to_array(raw)

        x_list.append(img)
        y_list.append(label)
        counts[label] = counts.get(label, 0) + 1

    if not x_list:
        raise RuntimeError(f"No c4/c7 synthetic samples loaded from {manifest_path}")

    x = normalize_images(np.stack(x_list, axis=0))
    y = np.asarray(y_list, dtype=np.int64)
    return x, y, counts


def infer_family_budget_seed(manifest_path: Path):
    parts = manifest_path.parts
    family = None
    for fam in ["gan", "vae", "diffusion"]:
        if fam in parts:
            family = fam.upper()
            break
    config_id = manifest_path.parent.parent.name
    seed_part = manifest_path.parent.name
    seed = int(seed_part.replace("seed", ""))
    m = re.search(r"_b(\d+)", config_id)
    budget = int(m.group(1)) if m else None
    return family, config_id, seed, budget


def dominant_label(preds: np.ndarray):
    vals, counts = np.unique(preds, return_counts=True)
    idx = int(np.argmax(counts))
    return int(vals[idx]), int(counts[idx]), float(counts[idx] / max(1, len(preds)))


def write_md_table(df: pd.DataFrame, path: Path):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")


def pm(m, sd):
    if pd.isna(m):
        return ""
    if abs(m) < 0.00005:
        m = 0.0
    if pd.isna(sd):
        sd = 0.0
    return f"{m:.4f} ± {sd:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=2000)
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()

    data_dir = find_data_dir()
    print(f"[paper3-audit] data_dir={data_dir}")

    x_train_full, y_train_full, x_test, y_test = load_real_dataset(data_dir)
    x_sub, y_sub, subsample_meta = apply_minority_subsample(x_train_full, y_train_full, seed=AUDIT_SEED)
    x_train, y_train, x_val, y_val = stratified_train_val_split(x_sub, y_sub, val_fraction=0.1, seed=AUDIT_SEED)

    print(f"[paper3-audit] x_train={x_train.shape} x_val={x_val.shape} x_test={x_test.shape}")
    print(f"[paper3-audit] subsample_meta={subsample_meta}")

    model, clf_metrics, num_classes = train_or_load_classifier(
        x_train, y_train, x_val, y_val, x_test, y_test, force_train=args.force_train
    )

    manifests = sorted(ARTS.glob("*/synthetic/*minority_heavy_c4c7*/seed*/manifest.json"))
    print(f"[paper3-audit] manifests={len(manifests)}")

    raw_rows = []
    hist_rows = []

    for mp in manifests:
        family, config_id, seed, budget = infer_family_budget_seed(mp)
        if family not in {"GAN", "VAE", "DIFFUSION"} or budget not in {100, 500, 2000}:
            continue

        print(f"[audit] {family} budget={budget} seed={seed} manifest={mp}")
        x_synth, y_req, loaded_counts = load_synthetic_from_manifest(mp, max_per_class=args.max_per_class)
        probs = model.predict(x_synth, batch_size=128, verbose=0)
        y_pred = np.argmax(probs, axis=1).astype("int64")

        for req in MINORITY_CLASSES:
            mask = y_req == req
            pred_req = y_pred[mask]
            n = int(mask.sum())
            correct = int(np.sum(pred_req == req))
            acc = float(correct / n) if n else 0.0
            leakage = float(1.0 - acc) if n else 0.0
            dom_label, dom_count, dom_frac = dominant_label(pred_req)

            raw_rows.append({
                "family": family,
                "config_id": config_id,
                "seed": seed,
                "budget_per_class": budget,
                "requested_class": req,
                "num_synthetic": n,
                "alignment_accuracy": acc,
                "leakage_rate": leakage,
                "correct": correct,
                "dominant_predicted_class": dom_label,
                "dominant_predicted_count": dom_count,
                "dominant_predicted_fraction": dom_frac,
                "manifest_path": str(mp),
            })

            for pred_label in range(num_classes):
                count = int(np.sum(pred_req == pred_label))
                hist_rows.append({
                    "family": family,
                    "config_id": config_id,
                    "seed": seed,
                    "budget_per_class": budget,
                    "requested_class": req,
                    "predicted_class": pred_label,
                    "count": count,
                    "fraction": float(count / n) if n else 0.0,
                })

    raw = pd.DataFrame(raw_rows).sort_values(["family", "budget_per_class", "seed", "requested_class"])
    hist = pd.DataFrame(hist_rows).sort_values(["family", "budget_per_class", "seed", "requested_class", "predicted_class"])

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)

    raw.to_csv(OUT_RAW, index=False)
    hist.to_csv(OUT_HIST, index=False)

    def mode_int(series):

        vc = series.value_counts()

        if vc.empty:

            return None

        return int(vc.index[0])



    agg = (
        raw.groupby(["family", "budget_per_class", "requested_class"])
        .agg(
            runs=("seed", "count"),
            num_synthetic_mean=("num_synthetic", "mean"),
            alignment_accuracy_mean=("alignment_accuracy", "mean"),
            alignment_accuracy_std=("alignment_accuracy", "std"),
            leakage_rate_mean=("leakage_rate", "mean"),
            leakage_rate_std=("leakage_rate", "std"),
            dominant_predicted_class_mode=("dominant_predicted_class", mode_int),
            dominant_predicted_fraction_mean=("dominant_predicted_fraction", "mean"),
            dominant_predicted_fraction_std=("dominant_predicted_fraction", "std"),
        )
        .reset_index()
        .sort_values(["budget_per_class", "family", "requested_class"])
    )

    agg.to_csv(OUT_AGG, index=False)

    table = pd.DataFrame({
        "Family": agg["family"],
        "Budget/Class": agg["budget_per_class"].astype(int),
        "Requested Class": agg["requested_class"].astype(int),
        "Runs": agg["runs"].astype(int),
        "N Audited": agg["num_synthetic_mean"].astype(int),
        "Audit Acc.": [pm(m, s) for m, s in zip(agg["alignment_accuracy_mean"], agg["alignment_accuracy_std"])],
        "Leakage": [pm(m, s) for m, s in zip(agg["leakage_rate_mean"], agg["leakage_rate_std"])],
        "Dominant Pred. Class": agg["dominant_predicted_class_mode"].astype(int),
        "Dominant Pred. Frac.": [pm(m, s) for m, s in zip(agg["dominant_predicted_fraction_mean"], agg["dominant_predicted_fraction_std"])],
    })

    write_md_table(table, OUT_MD)
    OUT_TEX.write_text(table.to_latex(index=False, escape=False))

    summary = {
        "audit_name": "paper3_c4c7_synthetic_alignment",
        "audit_seed": AUDIT_SEED,
        "minority_classes": MINORITY_CLASSES,
        "minority_fraction": MINORITY_FRACTION,
        "max_per_class": args.max_per_class,
        "data_dir": str(data_dir),
        "subsample_meta": subsample_meta,
        "real_only_classifier": clf_metrics,
        "num_manifests_audited": int(raw[["family", "config_id", "seed"]].drop_duplicates().shape[0]),
        "outputs": {
            "raw_csv": str(OUT_RAW),
            "hist_csv": str(OUT_HIST),
            "aggregate_csv": str(OUT_AGG),
            "table_md": str(OUT_MD),
            "table_tex": str(OUT_TEX),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    print("[ok] wrote", OUT_RAW)
    print("[ok] wrote", OUT_HIST)
    print("[ok] wrote", OUT_AGG)
    print("[ok] wrote", OUT_MD)
    print("[ok] wrote", OUT_TEX)
    print("[ok] wrote", OUT_SUMMARY)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
