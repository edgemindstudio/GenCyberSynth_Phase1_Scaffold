#!/usr/bin/env python3

"""

Paper 2 multi-classifier external conditioning audit.



Purpose:

    Test whether Paper 2 conditioning-collapse results are robust across

    multiple independent real-only audit classifiers.



Audit classifiers:

    1. resnet_cnn       : small ResNet-style CNN trained only on real data

    2. linear_svm_flat  : linear SVM-style classifier on flattened images



Default audited manifests:

    - fake-class ACGAN b100 seed42

    - fake-class ACGAN b100 seed43

    - fake-class ACGAN b100 seed44



Run from repo root:

    python papers/paper2_conditional_generation_done_right/scripts/audit_conditioning_multiclf_v1.py --max-per-class 100

"""



from __future__ import annotations



import argparse

import csv

import importlib.util

import json

import os

import sys

from pathlib import Path

from typing import Any, Dict, List, Tuple



import numpy as np



# ---------------------------------------------------------------------

# Repo setup

# ---------------------------------------------------------------------



SCRIPT_PATH = Path(__file__).resolve()

REPO_ROOT = SCRIPT_PATH.parents[3]



if Path.cwd().resolve() != REPO_ROOT:

    os.chdir(REPO_ROOT)



if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



# sklearn is used only for the flattened-image linear SVM audit model.

try:

    import joblib

    from sklearn.metrics import accuracy_score

    from sklearn.pipeline import make_pipeline

    from sklearn.preprocessing import StandardScaler

    from sklearn.linear_model import SGDClassifier

except Exception as e:

    raise RuntimeError(

        "This script requires scikit-learn and joblib for the linear_svm_flat audit classifier. "

        f"Import failed with: {repr(e)}"

    )



# ---------------------------------------------------------------------

# Import shared Paper 2 audit utilities from audit_conditioning_v2.py

# ---------------------------------------------------------------------



AUDIT_V2_PATH = REPO_ROOT / "papers" / "paper2_conditional_generation_done_right" / "scripts" / "audit_conditioning_v2.py"

spec = importlib.util.spec_from_file_location("paper2_audit_v2", AUDIT_V2_PATH)

audit_v2 = importlib.util.module_from_spec(spec)

assert spec.loader is not None

spec.loader.exec_module(audit_v2)



load_real_dataset = audit_v2.load_real_dataset

normalize_images_for_classifier = audit_v2.normalize_images_for_classifier

load_synthetic_from_manifest = audit_v2.load_synthetic_from_manifest

compute_confusion = audit_v2.compute_confusion

save_confusion_csv = audit_v2.save_confusion_csv

save_per_class_accuracy_csv = audit_v2.save_per_class_accuracy_csv

save_predicted_histogram_csv = audit_v2.save_predicted_histogram_csv

plot_confusion_heatmap = audit_v2.plot_confusion_heatmap

plot_per_class_accuracy = audit_v2.plot_per_class_accuracy

plot_predicted_histogram = audit_v2.plot_predicted_histogram



# ---------------------------------------------------------------------

# Constants

# ---------------------------------------------------------------------



PAPER_DIR = REPO_ROOT / "papers" / "paper2_conditional_generation_done_right"

RESULTS_DIR = PAPER_DIR / "results"

TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR = RESULTS_DIR / "figures"



ARTIFACTS_ROOT = Path("/home/bruno.fonkeng/gencys/artifacts_paper2")

DATA_DIR = Path("/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware_nhwc")

AUDIT_ROOT = ARTIFACTS_ROOT / "audit"



SEED = 42



TABLES_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_ROOT.mkdir(parents=True, exist_ok=True)



DEFAULT_MANIFESTS = {

    "paper2_acgan_fakeclass_b100_seed42": ARTIFACTS_ROOT / "gan" / "synthetic" / "paper2_acgan_fakeclass_b100_seed42" / "seed42" / "manifest.json",

    "paper2_acgan_fakeclass_b100_seed43": ARTIFACTS_ROOT / "gan" / "synthetic" / "paper2_acgan_fakeclass_b100_seed43" / "seed43" / "manifest.json",

    "paper2_acgan_fakeclass_b100_seed44": ARTIFACTS_ROOT / "gan" / "synthetic" / "paper2_acgan_fakeclass_b100_seed44" / "seed44" / "manifest.json",

}



np.random.seed(SEED)

tf.random.set_seed(SEED)





# ---------------------------------------------------------------------

# ResNet-style CNN audit classifier

# ---------------------------------------------------------------------



def residual_block(x, filters: int, stride: int = 1, name: str = "res"):

    shortcut = x



    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1")(x)

    x = layers.BatchNormalization(name=f"{name}_bn1")(x)

    x = layers.ReLU(name=f"{name}_relu1")(x)



    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False, name=f"{name}_conv2")(x)

    x = layers.BatchNormalization(name=f"{name}_bn2")(x)



    if stride != 1 or int(shortcut.shape[-1]) != filters:

        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, name=f"{name}_proj_conv")(shortcut)

        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)



    x = layers.Add(name=f"{name}_add")([x, shortcut])

    x = layers.ReLU(name=f"{name}_relu2")(x)

    return x





def build_resnet_audit_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:

    inputs = keras.Input(shape=input_shape, name="x")



    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="stem_conv")(inputs)

    x = layers.BatchNormalization(name="stem_bn")(x)

    x = layers.ReLU(name="stem_relu")(x)



    x = residual_block(x, 32, stride=1, name="block1")

    x = residual_block(x, 64, stride=2, name="block2")

    x = residual_block(x, 128, stride=2, name="block3")



    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(128, activation="relu", name="dense128")(x)

    x = layers.Dropout(0.25, name="dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="class_out")(x)



    model = keras.Model(inputs, outputs, name="paper2_resnet_style_audit_cnn")

    model.compile(

        optimizer=keras.optimizers.Adam(learning_rate=1e-3),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],

    )

    return model





def train_or_load_resnet(

    X_train: np.ndarray,

    y_train: np.ndarray,

    X_val: np.ndarray,

    y_val: np.ndarray,

    X_test: np.ndarray,

    y_test: np.ndarray,

    num_classes: int,

    epochs: int,

    batch_size: int,

) -> Tuple[keras.Model, Dict[str, Any]]:

    out_dir = AUDIT_ROOT / "resnet_cnn"

    out_dir.mkdir(parents=True, exist_ok=True)



    weights_path = out_dir / f"seed{SEED}_resnet_cnn.weights.h5"

    metrics_path = out_dir / f"seed{SEED}_resnet_cnn_metrics.json"



    model = build_resnet_audit_cnn(tuple(X_train.shape[1:]), num_classes)



    if weights_path.exists():

        print(f"[resnet_cnn] Loading existing weights: {weights_path}")

        model.load_weights(str(weights_path))

    else:

        print("[resnet_cnn] Training ResNet-style real-only audit classifier...")

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

            epochs=epochs,

            batch_size=batch_size,

            callbacks=callbacks,

            verbose=2,

        )

        model.save_weights(str(weights_path))

        print(f"[resnet_cnn] Saved weights: {weights_path}")



    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)



    metrics = {

        "audit_classifier": "resnet_cnn",

        "val_loss": float(val_loss),

        "val_accuracy": float(val_acc),

        "test_loss": float(test_loss),

        "test_accuracy": float(test_acc),

        "weights_path": str(weights_path),

    }



    with open(metrics_path, "w") as f:

        json.dump(metrics, f, indent=2)



    return model, metrics





# ---------------------------------------------------------------------

# Linear SVM-style flattened-image audit classifier

# ---------------------------------------------------------------------



def flatten_for_classical(X: np.ndarray) -> np.ndarray:

    X = X.astype("float32")

    return X.reshape((X.shape[0], -1))





def train_or_load_linear_svm(

    X_train: np.ndarray,

    y_train: np.ndarray,

    X_val: np.ndarray,

    y_val: np.ndarray,

    X_test: np.ndarray,

    y_test: np.ndarray,

) -> Tuple[Any, Dict[str, Any]]:

    out_dir = AUDIT_ROOT / "linear_svm_flat"

    out_dir.mkdir(parents=True, exist_ok=True)



    model_path = out_dir / f"seed{SEED}_linear_svm_flat.joblib"

    metrics_path = out_dir / f"seed{SEED}_linear_svm_flat_metrics.json"



    Xtr = flatten_for_classical(X_train)

    Xva = flatten_for_classical(X_val)

    Xte = flatten_for_classical(X_test)



    if model_path.exists():

        print(f"[linear_svm_flat] Loading existing model: {model_path}")

        clf = joblib.load(model_path)

    else:

        print("[linear_svm_flat] Training flattened-image linear SVM-style audit classifier...")

        clf = make_pipeline(

            StandardScaler(),

            SGDClassifier(

                loss="hinge",

                alpha=1e-4,

                max_iter=2000,

                tol=1e-3,

                random_state=SEED,

                n_jobs=-1,

            ),

        )

        clf.fit(Xtr, y_train)

        joblib.dump(clf, model_path)

        print(f"[linear_svm_flat] Saved model: {model_path}")



    y_val_pred = clf.predict(Xva)

    y_test_pred = clf.predict(Xte)



    metrics = {

        "audit_classifier": "linear_svm_flat",

        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),

        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),

        "model_path": str(model_path),

    }



    with open(metrics_path, "w") as f:

        json.dump(metrics, f, indent=2)



    return clf, metrics





# ---------------------------------------------------------------------

# Auditing helpers

# ---------------------------------------------------------------------



def parse_seed_from_tag(tag: str) -> int | None:

    for seed in [42, 43, 44]:

        if f"seed{seed}" in tag:

            return seed

    return None





def predicted_label_pattern(pred_counts: Dict[int, int]) -> str:

    nonzero = [(k, v) for k, v in pred_counts.items() if int(v) > 0]

    if len(nonzero) == 1:

        return f"full collapse to class {nonzero[0][0]}"

    classes = "/".join(str(k) for k, _ in nonzero)

    return f"collapse into classes {classes}"





def predict_with_classifier(name: str, clf: Any, X: np.ndarray) -> np.ndarray:

    if name == "resnet_cnn":

        probs = clf.predict(X, batch_size=128, verbose=0)

        return np.argmax(probs, axis=1).astype("int64")



    if name == "linear_svm_flat":

        Xf = flatten_for_classical(X)

        return clf.predict(Xf).astype("int64")



    raise ValueError(f"Unknown classifier: {name}")





def audit_manifest_with_classifier(

    audit_classifier_name: str,

    clf: Any,

    clf_metrics: Dict[str, Any],

    manifest_tag: str,

    manifest_path: Path,

    max_per_class: int,

    num_classes: int,

) -> Dict[str, Any]:

    print(f"[audit] classifier={audit_classifier_name} tag={manifest_tag}")

    print(f"[audit] manifest={manifest_path}")



    X_synth, y_requested = load_synthetic_from_manifest(manifest_path, max_per_class=max_per_class)

    X_synth = normalize_images_for_classifier(X_synth)

    y_requested = y_requested.astype("int64").reshape(-1)



    y_pred = predict_with_classifier(audit_classifier_name, clf, X_synth)



    confusion = compute_confusion(y_requested, y_pred, num_classes)

    total = int(len(y_requested))

    correct = int(np.sum(y_requested == y_pred))

    cond_acc = float(correct / total) if total else 0.0

    leakage_count = int(np.sum(y_requested != y_pred))

    leakage_rate = float(leakage_count / total) if total else 0.0

    failure_rate = float(1.0 - cond_acc)



    out_tag = f"{manifest_tag}_{audit_classifier_name}"



    confusion_csv = TABLES_DIR / f"paper2_multiclf_conditioning_confusion_{out_tag}.csv"

    per_class_csv = TABLES_DIR / f"paper2_multiclf_conditioning_accuracy_{out_tag}.csv"

    pred_hist_csv = TABLES_DIR / f"paper2_multiclf_predicted_label_histogram_{out_tag}.csv"



    save_confusion_csv(confusion, confusion_csv)

    per_class_rows = save_per_class_accuracy_csv(confusion, per_class_csv)

    pred_counts = save_predicted_histogram_csv(y_pred, num_classes, pred_hist_csv)



    confusion_png = FIGURES_DIR / f"paper2_multiclf_conditioning_confusion_{out_tag}.png"

    per_class_png = FIGURES_DIR / f"paper2_multiclf_conditioning_accuracy_{out_tag}.png"

    pred_hist_png = FIGURES_DIR / f"paper2_multiclf_predicted_label_histogram_{out_tag}.png"



    plot_confusion_heatmap(confusion, confusion_png)

    plot_per_class_accuracy(per_class_rows, per_class_png)

    plot_predicted_histogram(pred_counts, pred_hist_png)



    row = {

        "audit_classifier": audit_classifier_name,

        "dataset": "ustc_tfc2016",

        "manifest_tag": manifest_tag,

        "seed": parse_seed_from_tag(manifest_tag),

        "max_synth_per_class": int(max_per_class),

        "total_synthetic_audited": total,

        "real_val_accuracy": clf_metrics.get("val_accuracy"),

        "real_test_accuracy": clf_metrics.get("test_accuracy"),

        "conditioning_accuracy": cond_acc,

        "conditioning_failure_rate": failure_rate,

        "leakage_count": leakage_count,

        "leakage_rate": leakage_rate,

        "predicted_label_pattern": predicted_label_pattern(pred_counts),

        "predicted_label_histogram_csv": str(pred_hist_csv),

        "confusion_csv": str(confusion_csv),

        "per_class_accuracy_csv": str(per_class_csv),

        "predicted_label_histogram_png": str(pred_hist_png),

        "confusion_png": str(confusion_png),

        "per_class_accuracy_png": str(per_class_png),

    }



    print(

        f"[audit-result] {audit_classifier_name} {manifest_tag}: "

        f"acc={cond_acc:.4f} leakage={leakage_rate:.4f} pattern={row['predicted_label_pattern']}"

    )



    return row





def parse_manifest_specs(specs: List[str] | None) -> Dict[str, Path]:

    if not specs:

        return dict(DEFAULT_MANIFESTS)



    out: Dict[str, Path] = {}

    for item in specs:

        if "=" not in item:

            raise ValueError(

                f"Invalid --manifest-spec: {item}. Expected format tag=/path/to/manifest.json"

            )

        tag, path = item.split("=", 1)

        out[tag.strip()] = Path(path.strip())

    return out





# ---------------------------------------------------------------------

# Main

# ---------------------------------------------------------------------



def main(argv: List[str] | None = None) -> int:

    parser = argparse.ArgumentParser(description="Paper 2 multi-classifier conditioning audit")

    parser.add_argument(

        "--max-per-class",

        type=int,

        default=100,

        help="Maximum synthetic samples to audit per requested class.",

    )

    parser.add_argument(

        "--classifiers",

        nargs="+",

        default=["resnet_cnn", "linear_svm_flat"],

        choices=["resnet_cnn", "linear_svm_flat"],

        help="Audit classifiers to run.",

    )

    parser.add_argument(

        "--manifest-spec",

        action="append",

        default=None,

        help="Manifest specification in format tag=/path/to/manifest.json. Can be repeated.",

    )

    parser.add_argument(

        "--resnet-epochs",

        type=int,

        default=30,

        help="Maximum epochs for ResNet-style audit classifier.",

    )

    parser.add_argument(

        "--batch-size",

        type=int,

        default=128,

        help="Batch size for ResNet-style audit classifier.",

    )

    parser.add_argument(

        "--out-csv",

        type=Path,

        default=TABLES_DIR / "paper2_audit_classifier_comparison.csv",

        help="Output CSV summary table.",

    )



    args = parser.parse_args(argv)



    manifests = parse_manifest_specs(args.manifest_spec)



    print("[multi-clf] Repo root:", REPO_ROOT)

    print("[multi-clf] Data dir:", DATA_DIR)

    print("[multi-clf] Max per class:", args.max_per_class)

    print("[multi-clf] Classifiers:", args.classifiers)

    print("[multi-clf] Manifests:")

    for tag, path in manifests.items():

        print(f"  - {tag}: {path}")



    for tag, path in manifests.items():

        if not path.exists():

            raise FileNotFoundError(f"Manifest for {tag} does not exist: {path}")



    X_train, y_train, X_val, y_val, X_test, y_test = load_real_dataset(DATA_DIR)



    X_train = normalize_images_for_classifier(X_train)

    X_val = normalize_images_for_classifier(X_val)

    X_test = normalize_images_for_classifier(X_test)



    y_train = y_train.astype("int64").reshape(-1)

    y_val = y_val.astype("int64").reshape(-1)

    y_test = y_test.astype("int64").reshape(-1)



    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)



    rows: List[Dict[str, Any]] = []



    classifier_objects: Dict[str, Tuple[Any, Dict[str, Any]]] = {}



    if "resnet_cnn" in args.classifiers:

        classifier_objects["resnet_cnn"] = train_or_load_resnet(

            X_train,

            y_train,

            X_val,

            y_val,

            X_test,

            y_test,

            num_classes,

            epochs=args.resnet_epochs,

            batch_size=args.batch_size,

        )



    if "linear_svm_flat" in args.classifiers:

        classifier_objects["linear_svm_flat"] = train_or_load_linear_svm(

            X_train,

            y_train,

            X_val,

            y_val,

            X_test,

            y_test,

        )



    for clf_name in args.classifiers:

        clf, metrics = classifier_objects[clf_name]

        print(f"[multi-clf] {clf_name} metrics: {metrics}")



        for manifest_tag, manifest_path in manifests.items():

            row = audit_manifest_with_classifier(

                audit_classifier_name=clf_name,

                clf=clf,

                clf_metrics=metrics,

                manifest_tag=manifest_tag,

                manifest_path=manifest_path,

                max_per_class=args.max_per_class,

                num_classes=num_classes,

            )

            rows.append(row)



    if not rows:

        raise RuntimeError("No audit rows generated.")



    args.out_csv.parent.mkdir(parents=True, exist_ok=True)



    fieldnames = [

        "audit_classifier",

        "dataset",

        "manifest_tag",

        "seed",

        "max_synth_per_class",

        "total_synthetic_audited",

        "real_val_accuracy",

        "real_test_accuracy",

        "conditioning_accuracy",

        "conditioning_failure_rate",

        "leakage_count",

        "leakage_rate",

        "predicted_label_pattern",

        "predicted_label_histogram_csv",

        "confusion_csv",

        "per_class_accuracy_csv",

        "predicted_label_histogram_png",

        "confusion_png",

        "per_class_accuracy_png",

    ]



    with open(args.out_csv, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for row in rows:

            writer.writerow(row)



    print(f"[multi-clf] Wrote summary CSV: {args.out_csv}")

    print("[multi-clf] DONE")

    return 0





if __name__ == "__main__":

    raise SystemExit(main())
