# scripts/metrics/local_fid_kid.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local FID/KID with streaming features (OOM-safe).

- Backbone: 'mobilenet' (MobileNetV2, 1280-d, default) or 'inception' (2048-d)
- Images streamed in small batches; only running mean/cov kept in RAM.
- Writes back to the latest model summary: generative.cfid_macro (FID), kid=None.

Args:
  --config CONFIG_YAML
  --artifacts /path/to/artifacts
  --model gan|diffusion|autoregressive|vae|gaussianmixture|restrictedboltzmann|maskedautoflow
  --per_class_cap N
  --total_cap N
  --batch_size N             (default: 16)
  --backbone mobilenet|inception  (default: mobilenet)
  --img_size 160|299         (default: 160 for mobilenet; 299 for inception)
  --streaming {0,1}          (default: 1)

Notes:
- Requires TensorFlow/Keras, PIL, SciPy (for sqrtm).
- CPU-only is enforced by the sbatch wrapper via CUDA_VISIBLE_DEVICES="".
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image  # type: ignore
except Exception as e:
    print(f"[fidkid] PIL import failed: {e}", file=sys.stderr)
    raise

# ---------- IO / helpers ----------
def _load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _read_image_rgb(path: Path, target_size: Tuple[int,int]) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB").resize(target_size, Image.NEAREST)
        arr = np.asarray(img).astype("float32") / 255.0  # [0,1]
        return arr
    except Exception:
        return None

def _as_rgb_from_gray(x: np.ndarray, target_size=(160,160)) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype("float32", copy=False)
    if x.max() > 1.5:
        x = x / 255.0
    if x.ndim == 2:
        x = x[..., None]
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    im = Image.fromarray(np.clip(x*255.0, 0, 255).astype("uint8"))
    im = im.resize(target_size, Image.NEAREST)
    x = np.asarray(im).astype("float32")/255.0
    return x

def _model_encoder(backbone: str, img_size: int):
    import tensorflow as tf
    if backbone == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        base = MobileNetV2(include_top=False, weights="imagenet", pooling="avg",
                           input_shape=(img_size, img_size, 3))
        feat_dim = base.output_shape[-1]  # 1280
        def encode(x: np.ndarray) -> np.ndarray:
            x = x.astype("float32", copy=False)
            x = preprocess_input(x * 255.0)  # maps to [-1,1] for MobileNetV2
            return base(x, training=False).numpy()
        return encode, int(feat_dim)
    elif backbone == "inception":
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        base = InceptionV3(include_top=False, weights="imagenet", pooling="avg",
                           input_shape=(img_size, img_size, 3))
        feat_dim = base.output_shape[-1]  # 2048
        def encode(x: np.ndarray) -> np.ndarray:
            x = x.astype("float32", copy=False)
            x = preprocess_input(x * 255.0)  # inception preprocess
            return base(x, training=False).numpy()
        return encode, int(feat_dim)
    else:
        raise ValueError(f"unknown backbone: {backbone}")

# ---------- streaming stats ----------
class RunningMoments:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.S1 = np.zeros((dim,), dtype=np.float64)
        self.S2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, F: np.ndarray) -> None:
        if F.size == 0: return
        F64 = F.astype(np.float64, copy=False)
        self.n += F64.shape[0]
        self.S1 += F64.sum(axis=0)
        self.S2 += F64.T @ F64

    def mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n <= 1:
            mu = self.S1 / max(self.n, 1)
            cov = np.eye(self.dim, dtype=np.float64)
            return mu, cov
        mu = self.S1 / self.n
        cov = (self.S2 - self.n * np.outer(mu, mu)) / (self.n - 1)
        return mu, cov

def _fid_from_stats(mu_r, cov_r, mu_s, cov_s) -> float:
    from scipy.linalg import sqrtm  # type: ignore
    diff = mu_r - mu_s
    covmean = sqrtm(cov_r @ cov_s)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov_r + cov_s - 2.0 * covmean))

# ---------- data sources ----------
def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    man = json.loads(manifest_path.read_text())
    if "paths" not in man and isinstance(man.get("samples"), list):
        man["paths"] = man["samples"]
    man.setdefault("paths", [])
    return man

def iter_synth_images(manifest: Dict[str, Any], per_class_cap: int, total_cap: int, target_size=(160,160)) -> Iterable[np.ndarray]:
    counts, total = {}, 0
    for it in manifest.get("paths", []):
        if total_cap and total >= total_cap: break
        try:
            y = int(it.get("label", 0))
        except Exception:
            y = 0
        if per_class_cap and counts.get(y,0) >= per_class_cap:
            continue
        p = Path(it["path"])
        arr = _read_image_rgb(p, target_size=target_size)
        if arr is None:
            continue
        counts[y] = counts.get(y,0) + 1
        total += 1
        yield arr

def iter_real_images_from_npy(config: Dict[str,Any], total_cap: int, target_size=(160,160)) -> Iterable[np.ndarray]:
    data_root = (
        config.get("DATA_DIR")
        or (config.get("data", {}).get("root") if isinstance(config.get("data"), dict) else None)
        or "USTC-TFC2016_malware"
    )
    data_dir = Path(data_root)
    files = [data_dir/"train_data.npy", data_dir/"test_data.npy"]
    got = 0
    for f in files:
        if got >= total_cap: break
        if not f.exists():
            continue
        try:
            X = np.load(f, allow_pickle=False)
        except Exception:
            continue
        N = X.shape[0]
        take = min(N, max(0, total_cap - got)) if total_cap else N
        for i in range(take):
            xi = X[i]
            img = _as_rgb_from_gray(xi, target_size=target_size)
            got += 1
            yield img
            if got >= total_cap:
                break

# ---------- core compute ----------
def compute_streaming_fid(
    synth_iter: Iterable[np.ndarray],
    real_iter: Iterable[np.ndarray],
    encode_fn,
    feat_dim: int,
    batch_size: int = 16,
) -> float:
    R = RunningMoments(feat_dim)
    S = RunningMoments(feat_dim)

    def _consume(it, acc: RunningMoments):
        batch: List[np.ndarray] = []
        for x in it:
            batch.append(x)
            if len(batch) >= batch_size:
                F = encode_fn(np.stack(batch, axis=0))
                acc.update(F)
                batch.clear()
        if batch:
            F = encode_fn(np.stack(batch, axis=0))
            acc.update(F)
            batch.clear()

    _consume(real_iter, R)
    _consume(synth_iter, S)

    mu_r, cov_r = R.mean_cov()
    mu_s, cov_s = S.mean_cov()
    return _fid_from_stats(mu_r, cov_r, mu_s, cov_s)

def update_latest_summary(artifacts: Path, model: str, fid_value: Optional[float]) -> None:
    summ_dir = artifacts / model / "summaries"
    summ_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(summ_dir.glob("summary_*.json"))
    latest_path = files[-1] if files else (summ_dir / "summary_manual.json")

    rec = {}
    if latest_path.exists():
        try:
            rec = json.loads(latest_path.read_text())
        except Exception:
            rec = {}

    g = rec.get("generative", {})
    g["cfid_macro"] = float(fid_value) if fid_value is not None else None
    if "kid" not in g:
        g["kid"] = None
    rec["generative"] = g
    rec["metrics.cfid"] = g["cfid_macro"]
    rec["metrics.cfid_macro"] = g["cfid_macro"]
    rec["metrics.kid"] = g["kid"]

    latest_path.write_text(json.dumps(rec, indent=2))
    (summ_dir / "latest.json").write_text(json.dumps(rec, indent=2))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--per_class_cap", type=int, default=0)
    ap.add_argument("--total_cap", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--backbone", choices=["mobilenet","inception"], default="mobilenet")
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--streaming", type=int, default=1)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    artifacts = Path(args.artifacts)

    man_path = artifacts / args.model / "synthetic" / "manifest.json"
    if not man_path.exists():
        print(f"[fidkid] manifest missing: {man_path}", file=sys.stderr)
        sys.exit(2)
    manifest = _load_manifest(man_path)

    # Encoder
    encode, feat_dim = _model_encoder(args.backbone, args.img_size)

    # Iterators (balanced caps)
    synth_it = iter_synth_images(manifest, args.per_class_cap, args.total_cap, target_size=(args.img_size,args.img_size))
    real_cap = args.total_cap if args.total_cap > 0 else 1024
    real_it = iter_real_images_from_npy(cfg, total_cap=real_cap, target_size=(args.img_size,args.img_size))

    print(f"[fidkid] model={args.model} backbone={args.backbone} img={args.img_size} "
          f"per_class_cap={args.per_class_cap} total_cap={args.total_cap} batch={args.batch_size}")
    fid = compute_streaming_fid(synth_it, real_it, encode, feat_dim, batch_size=args.batch_size)
    print(f"[fidkid] FID={fid:.6f}")

    update_latest_summary(artifacts, args.model, fid)

if __name__ == "__main__":
    main()
