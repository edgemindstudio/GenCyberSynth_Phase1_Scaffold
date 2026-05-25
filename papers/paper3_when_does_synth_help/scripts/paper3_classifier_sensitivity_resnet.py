#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, average_precision_score


REPO_ROOT = Path("/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold")
ARTS = Path("/home/bruno.fonkeng/gencys/artifacts_paper3")
DATA_CANDIDATES = [
    Path("/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware_nhwc"),
    Path("/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware"),
]

OUT_RAW = REPO_ROOT / "papers/paper3_when_does_synth_help/results/raw/paper3_resnet_sensitivity_raw.csv"
OUT_AGG = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_aggregate_frozen_20260524.csv"
OUT_MD = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_table.md"
OUT_TEX = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_table.tex"
OUT_SUMMARY = REPO_ROOT / "papers/paper3_when_does_synth_help/results/frozen/paper3_resnet_sensitivity_summary_20260524.json"

MINORITY_CLASSES = [4, 7]
MINORITY_FRACTION = 0.2
NUM_CLASSES = 9
IMG_SHAPE = (40, 40, 1)


def find_data_dir() -> Path:
    for d in DATA_CANDIDATES:
        if all((d / f).exists() for f in ["train_data.npy", "train_labels.npy", "test_data.npy", "test_labels.npy"]):
            return d
    raise FileNotFoundError("Could not locate USTC-TFC2016 npy dataset.")


def to_nhwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[..., None]
    if x.ndim == 4 and x.shape[1] == 1 and x.shape[-1] != 1:
        x = np.transpose(x, (0, 2, 3, 1))
    return x


def normalize_images(x: np.ndarray) -> np.ndarray:
    x = to_nhwc(x).astype("float32")
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmin >= -1.05 and xmax <= 1.05 and xmin < 0:
        x = (x + 1.0) / 2.0
    elif xmax > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0).astype("float32")


def load_real_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = find_data_dir()
    x_train_full = normalize_images(np.load(data_dir / "train_data.npy"))
    y_train_full = np.load(data_dir / "train_labels.npy").astype("int64").reshape(-1)
    x_test = normalize_images(np.load(data_dir / "test_data.npy"))
    y_test = np.load(data_dir / "test_labels.npy").astype("int64").reshape(-1)

    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for c in sorted(np.unique(y_train_full)):
        idx = np.where(y_train_full == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * 0.1)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return x_train_full[train_idx], y_train_full[train_idx], x_train_full[val_idx], y_train_full[val_idx], x_test, y_test


def apply_minority_subsample(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keep = []
    before = {str(c): int(np.sum(y_train == c)) for c in sorted(np.unique(y_train))}

    for c in sorted(np.unique(y_train)):
        idx = np.where(y_train == c)[0]
        if int(c) in MINORITY_CLASSES:
            n_keep = max(1, int(round(len(idx) * MINORITY_FRACTION)))
            idx = rng.choice(idx, size=n_keep, replace=False)
        keep.extend(idx.tolist())

    keep = np.asarray(keep, dtype=np.int64)
    rng.shuffle(keep)

    x_sub = x_train[keep]
    y_sub = y_train[keep]
    after = {str(c): int(np.sum(y_sub == c)) for c in sorted(np.unique(y_train))}

    meta = {
        "minority_classes": MINORITY_CLASSES,
        "minority_fraction": MINORITY_FRACTION,
        "before_counts": before,
        "after_counts": after,
        "num_before": int(len(y_train)),
        "num_after": int(len(y_sub)),
    }
    return x_sub, y_sub, meta


def resolve_path(path_value: str, manifest_path: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p

    candidates = [
        manifest_path.parent / p,
        manifest_path.parent.parent.parent / p,
        ARTS / p,
    ]

    for fam in ["gan", "vae", "diffusion"]:
        candidates.append(ARTS / fam / "synthetic" / p)

    for c in candidates:
        if c.exists():
            return c

    return p


def read_manifest_entries(manifest_path: Path) -> List[Dict[str, Any]]:
    m = json.load(open(manifest_path))
    for key in ["paths", "samples", "items", "records", "files", "images"]:
        if isinstance(m.get(key), list):
            return m[key]
    raise ValueError(f"No sample list found in manifest: {manifest_path}")


def load_synth_from_manifest(manifest_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    entries = read_manifest_entries(manifest_path)
    xs, ys = [], []

    for e in entries:
        label = int(e.get("label", e.get("class_id", e.get("requested_label", -1))))
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

        xs.append(img)
        ys.append(label)

    if not xs:
        raise RuntimeError(f"No synthetic samples loaded from {manifest_path}")

    return normalize_images(np.stack(xs, axis=0)), np.asarray(ys, dtype=np.int64)


def residual_block(x, filters: int, stride: int = 1):
    shortcut = x

    y = keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    y = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = keras.layers.BatchNormalization()(y)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    out = keras.layers.Add()([shortcut, y])
    out = keras.layers.Activation("relu")(out)
    return out


def build_resnet_classifier(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = residual_block(x, 32, stride=1)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 128, stride=2)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="Paper3SensitivityResNet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def macro_auprc(y_true: np.ndarray, probs: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    vals = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        try:
            vals.append(float(average_precision_score(y_bin, probs[:, c])))
        except Exception:
            vals.append(0.0)
    return float(np.mean(vals))


def evaluate_model(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    probs = model.predict(x_test, batch_size=256, verbose=0)
    pred = np.argmax(probs, axis=1)

    prec, rec, f1c, sup = precision_recall_fscore_support(
        y_test, pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
    )

    return {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_auprc": macro_auprc(y_test, probs, NUM_CLASSES),
        "class4_f1": float(f1c[4]),
        "class7_f1": float(f1c[7]),
        "per_class_f1": f1c.astype(float).tolist(),
        "per_class_support": sup.astype(int).tolist(),
    }


def train_eval(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int, epochs: int) -> Dict[str, Any]:
    tf.keras.utils.set_random_seed(seed)
    model = build_resnet_classifier(tuple(x_train.shape[1:]), NUM_CLASSES)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)
    ]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )
    return evaluate_model(model, x_test, y_test)


def infer_manifest(family: str, regime: str, budget: int, seed: int) -> Path:
    fam = family.lower()
    if regime == "balanced":
        candidates = sorted((ARTS / fam / "synthetic").glob(f"*balanced*b{budget}/seed{seed}/manifest.json"))
        if not candidates:
            candidates = sorted((ARTS / fam / "synthetic").glob(f"*b{budget}/seed{seed}/manifest.json"))
    elif regime == "minority_c4c7":
        candidates = sorted((ARTS / fam / "synthetic").glob(f"*minority_heavy_c4c7_b{budget}/seed{seed}/manifest.json"))
    else:
        raise ValueError(regime)

    if not candidates:
        raise FileNotFoundError(f"No manifest found for family={family} regime={regime} budget={budget} seed={seed}")
    return candidates[0]


def delta(a: Dict[str, Any], b: Dict[str, Any], key: str):
    return float(a[key] - b[key])


def run_one(family: str, regime: str, budget: int, seed: int, epochs: int) -> Dict[str, Any]:
    print(f"[run] family={family} regime={regime} budget={budget} seed={seed}")
    x_train, y_train, x_val, y_val, x_test, y_test = load_real_dataset(seed)

    subsample_meta = None
    if regime == "minority_c4c7":
        x_train, y_train, subsample_meta = apply_minority_subsample(x_train, y_train, seed)

    manifest = infer_manifest(family, regime, budget, seed)
    x_synth, y_synth = load_synth_from_manifest(manifest)

    print("[data] real train:", x_train.shape, y_train.shape)
    print("[data] synth:", x_synth.shape, y_synth.shape)
    print("[data] test:", x_test.shape, y_test.shape)

    real_only = train_eval(x_train, y_train, x_val, y_val, x_test, y_test, seed=seed, epochs=epochs)

    x_rs = np.concatenate([x_train, x_synth], axis=0)
    y_rs = np.concatenate([y_train, y_synth], axis=0)
    real_plus_synth = train_eval(x_rs, y_rs, x_val, y_val, x_test, y_test, seed=seed + 1000, epochs=epochs)

    row = {
        "classifier": "resnet_style_cnn",
        "family": family.upper(),
        "regime": regime,
        "budget_per_class": int(budget),
        "seed": int(seed),
        "manifest_path": str(manifest),
        "num_synthetic": int(len(y_synth)),
        "real_only_macro_f1": real_only["macro_f1"],
        "real_plus_synth_macro_f1": real_plus_synth["macro_f1"],
        "delta_macro_f1": delta(real_plus_synth, real_only, "macro_f1"),
        "real_only_balanced_accuracy": real_only["balanced_accuracy"],
        "real_plus_synth_balanced_accuracy": real_plus_synth["balanced_accuracy"],
        "delta_balanced_accuracy": delta(real_plus_synth, real_only, "balanced_accuracy"),
        "real_only_macro_auprc": real_only["macro_auprc"],
        "real_plus_synth_macro_auprc": real_plus_synth["macro_auprc"],
        "delta_macro_auprc": delta(real_plus_synth, real_only, "macro_auprc"),
        "real_only_class4_f1": real_only["class4_f1"],
        "real_plus_synth_class4_f1": real_plus_synth["class4_f1"],
        "delta_class4_f1": delta(real_plus_synth, real_only, "class4_f1"),
        "real_only_class7_f1": real_only["class7_f1"],
        "real_plus_synth_class7_f1": real_plus_synth["class7_f1"],
        "delta_class7_f1": delta(real_plus_synth, real_only, "class7_f1"),
        "real_train_subsample": json.dumps(subsample_meta) if subsample_meta else "",
    }
    print("[result]", json.dumps({k: row[k] for k in ["delta_macro_f1", "delta_balanced_accuracy", "delta_macro_auprc", "delta_class4_f1", "delta_class7_f1"]}, indent=2))
    return row


def pm(m, s):
    if pd.isna(m):
        return ""
    if abs(m) < 0.00005:
        m = 0.0
    if pd.isna(s):
        s = 0.0
    return f"{m:.4f} ± {s:.4f}"


def write_md_table(table: pd.DataFrame, path: Path):
    headers = list(table.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["gan", "vae", "diffusion"])
    ap.add_argument("--regimes", nargs="+", default=["balanced", "minority_c4c7"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[2000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out-raw", type=Path, default=None, help="Optional per-task raw CSV output path.")
    ap.add_argument("--no-aggregate", action="store_true", help="Only write raw output; skip aggregate/table outputs.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        families = ["gan"]
        regimes = ["minority_c4c7"]
        budgets = [2000]
        seeds = [42]
        epochs = min(args.epochs, 2)
    else:
        families = args.families
        regimes = args.regimes
        budgets = args.budgets
        seeds = args.seeds
        epochs = args.epochs

    rows = []
    for regime in regimes:
        for family in families:
            for budget in budgets:
                for seed in seeds:
                    rows.append(run_one(family, regime, budget, seed, epochs=epochs))

    raw = pd.DataFrame(rows)

    out_raw = args.out_raw if args.out_raw is not None else OUT_RAW
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_raw, index=False)

    if args.no_aggregate:
        print("[ok] wrote", out_raw)
        return

    agg = (
        raw.groupby(["classifier", "regime", "family", "budget_per_class"])
        .agg(
            runs=("seed", "count"),
            delta_macro_f1_mean=("delta_macro_f1", "mean"),
            delta_macro_f1_std=("delta_macro_f1", "std"),
            delta_balanced_accuracy_mean=("delta_balanced_accuracy", "mean"),
            delta_balanced_accuracy_std=("delta_balanced_accuracy", "std"),
            delta_macro_auprc_mean=("delta_macro_auprc", "mean"),
            delta_macro_auprc_std=("delta_macro_auprc", "std"),
            delta_class4_f1_mean=("delta_class4_f1", "mean"),
            delta_class4_f1_std=("delta_class4_f1", "std"),
            delta_class7_f1_mean=("delta_class7_f1", "mean"),
            delta_class7_f1_std=("delta_class7_f1", "std"),
        )
        .reset_index()
        .sort_values(["regime", "budget_per_class", "family"])
    )
    agg.to_csv(OUT_AGG, index=False)

    table = pd.DataFrame({
        "Classifier": agg["classifier"],
        "Regime": agg["regime"],
        "Family": agg["family"],
        "Budget/Class": agg["budget_per_class"].astype(int),
        "Runs": agg["runs"].astype(int),
        "Δ Macro-F1": [pm(m, s) for m, s in zip(agg["delta_macro_f1_mean"], agg["delta_macro_f1_std"])],
        "Δ Bal. Acc.": [pm(m, s) for m, s in zip(agg["delta_balanced_accuracy_mean"], agg["delta_balanced_accuracy_std"])],
        "Δ Macro-AUPRC": [pm(m, s) for m, s in zip(agg["delta_macro_auprc_mean"], agg["delta_macro_auprc_std"])],
        "Δ Class-4 F1": [pm(m, s) for m, s in zip(agg["delta_class4_f1_mean"], agg["delta_class4_f1_std"])],
        "Δ Class-7 F1": [pm(m, s) for m, s in zip(agg["delta_class7_f1_mean"], agg["delta_class7_f1_std"])],
    })

    write_md_table(table, OUT_MD)
    OUT_TEX.write_text(table.to_latex(index=False, escape=False))

    summary = {
        "analysis": "paper3_classifier_sensitivity_resnet",
        "classifier": "resnet_style_cnn",
        "families": families,
        "regimes": regimes,
        "budgets": budgets,
        "seeds": seeds,
        "epochs": epochs,
        "outputs": {
            "raw_csv": str(out_raw),
            "aggregate_csv": str(OUT_AGG),
            "table_md": str(OUT_MD),
            "table_tex": str(OUT_TEX),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    print("[ok] wrote", out_raw)
    print("[ok] wrote", OUT_AGG)
    print("[ok] wrote", OUT_MD)
    print("[ok] wrote", OUT_TEX)
    print("[ok] wrote", OUT_SUMMARY)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
