# scripts/utils/generate_dummy_embeddings.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate quick 64-D (configurable) embeddings so NN-distance & UMAP plots can run.
- Uses grayscale vectors from image globs and a Gaussian random projection.
- Writes to artifacts/<model>/features/{real,synth}_embeddings.npy (+ optional *_labels.npy)

Example:
python scripts/utils/generate_dummy_embeddings.py \
  --model gan \
  --dim 64 \
  --real-glob "USTC-TFC2016_malware/real/*/*.png" \
  --synth-glob "artifacts/gan/samples/*.png" \
  --limit 2000
"""
from __future__ import annotations
import argparse, os, json, random
from pathlib import Path
from glob import glob
import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(glob_pattern: str):
    if not glob_pattern: return []
    paths = []
    for p in glob(glob_pattern, recursive=True):
        if os.path.splitext(p)[1].lower() in IMG_EXTS:
            paths.append(p)
    return paths

def load_gray_vector(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32) / 255.0  # [0,1]
    return arr.reshape(-1)

def to_labels_from_parent(paths):
    labs = []
    for p in paths:
        try:
            labs.append(int(Path(p).parent.name))
        except Exception:
            labs.append(-1)
    return np.array(labs, dtype=np.int32)

def rp_embed(vectors: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    D = vectors.shape[1]
    W = rng.standard_normal(size=(D, dim)).astype(np.float32)
    E = vectors @ W
    E = (E - E.mean(0, keepdims=True)) / (E.std(0, keepdims=True) + 1e-6)
    return E.astype(np.float32)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def synth_count_from_manifest(manifest_path: Path) -> int:
    if not manifest_path.exists(): return 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data.get("images", []))

def build_embeddings(glob_pattern, out_dir: Path, prefix: str, dim: int, seed: int,
                     random_fallback: int = 0, limit: int = 0):
    paths = list_images(glob_pattern)
    if limit and len(paths) > limit:
        random.Random(seed).shuffle(paths)
        paths = paths[:limit]

    if not paths and random_fallback > 0:
        E = np.random.default_rng(seed).standard_normal(size=(random_fallback, dim)).astype(np.float32)
        L = -1 * np.ones((random_fallback,), dtype=np.int32)
        np.save(out_dir / f"{prefix}_embeddings.npy", E)
        np.save(out_dir / f"{prefix}_labels.npy", L)
        print(f"[warn] No images for '{prefix}'. Wrote random {random_fallback}×{dim} embeddings.")
        return

    if not paths:
        print(f"[warn] No images for '{prefix}' and no fallback size provided. Skipping.")
        return

    vecs = [load_gray_vector(p) for p in paths]
    V = np.vstack(vecs)
    E = rp_embed(V, dim=dim, seed=seed)
    np.save(out_dir / f"{prefix}_embeddings.npy", E)

    L = to_labels_from_parent(paths)
    np.save(out_dir / f"{prefix}_labels.npy", L)

    print(f"[ok] {prefix}: {E.shape[0]} embeddings → {out_dir/(prefix+'_embeddings.npy')}")

def main():
    ap = argparse.ArgumentParser(description="Generate dummy embeddings for NN-distance & UMAP.")
    ap.add_argument("--model", required=True, help="Model name under artifacts/<model>/features")
    ap.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-glob", type=str, default="")
    ap.add_argument("--synth-glob", type=str, default="")
    ap.add_argument("--limit", type=int, default=0, help="Cap per split (0 = no cap)")
    ap.add_argument("--fallback-real-count", type=int, default=0)
    ap.add_argument("--fallback-synth-count", type=int, default=0)
    ap.add_argument("--manifest", type=str, default="", help="Optional synth manifest to infer fallback size")
    args = ap.parse_args()

    out_dir = Path("artifacts") / args.model / "features"
    ensure_dir(out_dir)

    synth_fb = args.fallback_synth_count
    if synth_fb == 0 and args.manifest:
        n = synth_count_from_manifest(Path(args.manifest))
        if n > 0: synth_fb = n

    build_embeddings(args.real_glob,  out_dir, "real",  args.dim, args.seed,
                     random_fallback=args.fallback_real_count,  limit=args.limit)
    build_embeddings(args.synth_glob, out_dir, "synth", args.dim, args.seed,
                     random_fallback=synth_fb,                   limit=args.limit)

if __name__ == "__main__":
    main()
