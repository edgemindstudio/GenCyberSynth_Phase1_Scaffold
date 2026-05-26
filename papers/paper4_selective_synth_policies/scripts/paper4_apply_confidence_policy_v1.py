#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_real_only_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def parse_img_shape(text: str) -> Tuple[int, int, int]:
    parts = tuple(int(x.strip()) for x in text.split(","))
    if len(parts) != 3:
        raise ValueError(f"--img_shape must be H,W,C, got: {text}")
    return parts


def resolve_path(manifest_path: Path, sample_path: str) -> Path:
    p = Path(sample_path)
    if p.is_absolute():
        return p

    cur = manifest_path.parent
    while True:
        cand = (cur / sample_path).resolve()
        if cand.exists():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent

    return (manifest_path.parent / sample_path).resolve()


def load_image(path: Path, img_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w, c = img_shape
    img = Image.open(path).convert("L").resize((w, h), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = arr * 2.0 - 1.0
    arr = arr[..., None]
    if c == 1:
        return arr
    return np.repeat(arr, c, axis=-1)


def load_manifest_samples(manifest_path: Path, img_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    entries = manifest.get("paths", [])
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"No manifest paths found in {manifest_path}")

    x_list: List[np.ndarray] = []
    y_req: List[int] = []
    resolved_paths: List[str] = []
    kept_entries: List[Dict[str, Any]] = []

    for e in entries:
        if not isinstance(e, dict):
            continue
        sample_path = e.get("path")
        label = e.get("label")
        if sample_path is None or label is None:
            continue

        full = resolve_path(manifest_path, str(sample_path))
        if not full.exists():
            raise FileNotFoundError(full)

        x_list.append(load_image(full, img_shape))
        y_req.append(int(label))
        resolved_paths.append(str(full))
        kept_entries.append(dict(e))

    if not x_list:
        raise RuntimeError(f"No loadable samples found in manifest: {manifest_path}")

    x = np.stack(x_list, axis=0).astype("float32")
    y = np.asarray(y_req, dtype="int64")
    return x, y, resolved_paths, kept_entries, manifest


def select_threshold(y_req: np.ndarray, y_pred: np.ndarray, max_conf: np.ndarray, threshold: float) -> np.ndarray:
    return (y_pred == y_req) & (max_conf >= threshold)


def select_topk_per_class(y_req: np.ndarray, requested_conf: np.ndarray, top_k: int, threshold: float | None = None) -> np.ndarray:
    accepted = np.zeros(len(y_req), dtype=bool)
    for cls in sorted(np.unique(y_req).tolist()):
        idx = np.where(y_req == cls)[0]
        if threshold is not None:
            idx = idx[requested_conf[idx] >= threshold]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(-requested_conf[idx])]
        accepted[order[:top_k]] = True
    return accepted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--weights", default="/home/bruno.fonkeng/gencys/artifacts_paper2/audit/real_only_cnn/seed42_real_only_cnn.weights.h5")
    ap.add_argument("--img_shape", default="40,40,1", help="Image shape as H,W,C, e.g. 40,40,1 or 12,12,1")
    ap.add_argument("--num_classes", type=int, default=9)
    ap.add_argument("--dataset_id", default="ustc_tfc2016")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--policy_id", default="confidence_accept_t080")
    ap.add_argument("--audit_csv", default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--mode", choices=["threshold", "topk_per_class"], default="threshold")
    ap.add_argument("--top_k_per_class", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    weights_path = Path(args.weights)
    img_shape = parse_img_shape(args.img_shape)

    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if out_path.resolve() == in_path.resolve():
        raise ValueError("Refusing to overwrite source manifest.")

    X, y_req, resolved_paths, entries, source_manifest = load_manifest_samples(in_path, img_shape)

    model = build_real_only_cnn(img_shape, int(args.num_classes))
    model.load_weights(str(weights_path))

    probs = model.predict(X, batch_size=args.batch_size, verbose=0)
    y_pred = probs.argmax(axis=1).astype("int64")
    max_conf = probs.max(axis=1).astype("float32")
    requested_conf = probs[np.arange(len(y_req)), y_req].astype("float32")

    if args.mode == "threshold":
        accepted = select_threshold(y_req, y_pred, max_conf, float(args.threshold))
    else:
        if args.top_k_per_class is None:
            raise ValueError("--top_k_per_class is required when --mode topk_per_class")
        accepted = select_topk_per_class(y_req, requested_conf, int(args.top_k_per_class), threshold=float(args.threshold))

    selected_entries: List[Dict[str, Any]] = []
    for i, ok in enumerate(accepted):
        if not ok:
            continue
        e = dict(entries[i])
        e["path"] = resolved_paths[i]
        e["label"] = int(y_req[i])
        e["requested_label"] = int(y_req[i])
        e["predicted_label"] = int(y_pred[i])
        e["confidence"] = float(max_conf[i])
        e["requested_class_confidence"] = float(requested_conf[i])
        selected_entries.append(e)

    per_class_counts: Dict[str, int] = {}
    for e in selected_entries:
        label = str(int(e["label"]))
        per_class_counts[label] = per_class_counts.get(label, 0) + 1

    for k in range(int(args.num_classes)):
        per_class_counts.setdefault(str(k), 0)

    out_manifest = {
        "dataset": source_manifest.get("dataset", args.dataset_id),
        "dataset_id": args.dataset_id,
        "seed": source_manifest.get("seed"),
        "source_manifest": str(in_path),
        "paths": selected_entries,
        "num_fake": len(selected_entries),
        "per_class_counts": dict(sorted(per_class_counts.items(), key=lambda kv: int(kv[0]))),
        "budget_per_class": args.top_k_per_class if args.mode == "topk_per_class" else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper4_policy": {
            "paper_id": "paper4",
            "policy_id": args.policy_id,
            "policy_type": "external_real_only_cnn_confidence_acceptance",
            "source_manifest": str(in_path),
            "audit_classifier": "paper2_real_only_cnn",
            "audit_classifier_weights": str(weights_path),
            "dataset_id": args.dataset_id,
            "img_shape": list(img_shape),
            "num_classes": int(args.num_classes),
            "threshold": float(args.threshold),
            "mode": args.mode,
            "top_k_per_class": args.top_k_per_class,
            "num_before": int(len(y_req)),
            "num_after": int(len(selected_entries)),
            "num_rejected": int(len(y_req) - len(selected_entries)),
            "acceptance_rate": float(len(selected_entries) / len(y_req)) if len(y_req) else 0.0,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "description": "Accept or rank synthetic samples using an external real-only CNN audit signal.",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_manifest, f, indent=2)

    audit_path = Path(args.audit_csv) if args.audit_csv else out_path.parent / "audit.csv"
    audit_df = pd.DataFrame({
        "path": resolved_paths,
        "requested_label": y_req.astype(int),
        "predicted_label": y_pred.astype(int),
        "confidence": max_conf.astype(float),
        "requested_class_confidence": requested_conf.astype(float),
        "accepted": accepted.astype(int),
    })
    audit_df.to_csv(audit_path, index=False)

    print("[ok] wrote confidence policy manifest:", out_path)
    print("[ok] wrote audit csv:", audit_path)
    print("policy_id:", args.policy_id)
    print("mode:", args.mode)
    print("threshold:", args.threshold)
    print("num_before:", len(y_req))
    print("num_after:", len(selected_entries))
    print("acceptance_rate:", float(len(selected_entries) / len(y_req)) if len(y_req) else 0.0)
    print("per_class_counts:", out_manifest["per_class_counts"])


if __name__ == "__main__":
    main()