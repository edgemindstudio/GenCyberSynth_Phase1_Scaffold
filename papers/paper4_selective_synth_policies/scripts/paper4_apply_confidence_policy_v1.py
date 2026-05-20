#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def resolve_path(manifest_path: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    cur = manifest_path.parent
    while True:
        cand = (cur / p).resolve()
        if cand.exists():
            return cand
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return (manifest_path.parent / p).resolve()


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((40, 40), Image.NEAREST)
    arr = np.asarray(img).astype("float32")
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr[..., None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--weights", default="/home/bruno.fonkeng/gencys/artifacts_paper2/audit/real_only_cnn/seed42_real_only_cnn.weights.h5")
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

    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if out_path.resolve() == in_path.resolve():
        raise ValueError("Refusing to overwrite source manifest.")

    manifest = json.load(open(in_path))
    entries = manifest.get("paths", [])
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"No manifest paths found in {in_path}")

    X_list: List[np.ndarray] = []
    y_req: List[int] = []
    resolved_paths: List[str] = []

    for e in entries:
        if not isinstance(e, dict):
            continue
        rel = e.get("path")
        label = e.get("label")
        if rel is None or label is None:
            continue
        full = resolve_path(in_path, str(rel))
        if not full.exists():
            raise FileNotFoundError(full)
        X_list.append(load_image(full))
        y_req.append(int(label))
        resolved_paths.append(str(full))

    X = np.stack(X_list, axis=0).astype("float32")
    y_req_arr = np.asarray(y_req, dtype="int64")

    model = build_real_only_cnn((40, 40, 1), 9)
    model.load_weights(str(weights_path))

    probs = model.predict(X, batch_size=args.batch_size, verbose=0)
    y_pred = probs.argmax(axis=1).astype("int64")
    conf = probs.max(axis=1).astype("float32")

    requested_conf = probs[np.arange(len(y_req_arr)), y_req_arr]

    if args.mode == "threshold":
        accepted_mask = (y_pred == y_req_arr) & (conf >= float(args.threshold))
    elif args.mode == "topk_per_class":
        if args.top_k_per_class is None:
            raise ValueError("--top_k_per_class is required when --mode topk_per_class")
        accepted_mask = np.zeros(len(y_req_arr), dtype=bool)
        for cls in sorted(set(y_req_arr.tolist())):
            idx = np.where(y_req_arr == cls)[0]
            order = idx[np.argsort(-requested_conf[idx])]
            keep = order[: int(args.top_k_per_class)]
            accepted_mask[keep] = True
    else:
        raise ValueError(args.mode)
    accepted_entries = [deepcopy(e) for e, keep in zip(entries, accepted_mask) if bool(keep)]

    per_class_counts: Dict[str, int] = {}
    for e in accepted_entries:
        y = str(int(e["label"]))
        per_class_counts[y] = per_class_counts.get(y, 0) + 1

    out_manifest = deepcopy(manifest)
    out_manifest["paths"] = accepted_entries
    out_manifest["num_fake"] = int(len(accepted_entries))
    out_manifest["per_class_counts"] = per_class_counts
    out_manifest["paper4_policy"] = {
        "paper_id": "paper4",
        "policy_id": args.policy_id,
        "policy_type": "external_real_only_cnn_confidence_acceptance",
        "source_manifest": str(in_path),
        "audit_classifier": "paper2_real_only_cnn",
        "audit_classifier_weights": str(weights_path),
        "threshold": float(args.threshold),
        "mode": args.mode,
        "top_k_per_class": args.top_k_per_class,
        "num_before": int(len(entries)),
        "num_after": int(len(accepted_entries)),
        "num_rejected": int(len(entries) - len(accepted_entries)),
        "acceptance_rate": float(len(accepted_entries) / max(1, len(entries))),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Accept synthetic samples only when the Paper 2 real-only CNN predicts the requested class with confidence above threshold."
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_manifest, open(out_path, "w"), indent=2)

    audit_csv = Path(args.audit_csv) if args.audit_csv else out_path.with_name("audit.csv")
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "path", "requested_label", "predicted_label", "confidence", "requested_class_confidence", "accepted"])
        for i, (entry, yp, cf, rcf, keep) in enumerate(zip(entries, y_pred, conf, requested_conf, accepted_mask)):
            w.writerow([i, entry.get("path"), int(y_req_arr[i]), int(yp), float(cf), float(rcf), int(bool(keep))])

    print("[ok] wrote confidence policy manifest:", out_path)
    print("[ok] wrote audit csv:", audit_csv)
    print("policy_id:", args.policy_id)
    print("threshold:", args.threshold)
    print("num_before:", len(entries))
    print("num_after:", len(accepted_entries))
    print("acceptance_rate:", len(accepted_entries) / max(1, len(entries)))
    print("per_class_counts:", per_class_counts)


if __name__ == "__main__":
    main()