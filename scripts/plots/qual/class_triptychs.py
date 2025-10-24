# scripts/plots/qual/class_triptychs.py
"""
Class-conditioned triptychs (Real | Synth | NN-Real)

What
----
For selected classes, render a 3-row panel:
  Row 1: Real examples (k images)
  Row 2: Synth examples (k images)
  Row 3: Nearest real neighbor for each synth (in a simple feature space)

Why
---
Visually checks (a) class semantics, (b) that synth samples aren't trivial copies,
and (c) overall realism/variety.

Inputs it can discover automatically
------------------------------------
- Consolidated JSONL: artifacts/summaries/phase1_summaries.jsonl
  (used to locate the latest manifest_path per model).
- Synth manifests: artifacts/<model>/synthetic/manifest.json
  (fallback search if JSONL missing).
- Real images: USTC-TFC2016_malware/real/<class_id>/** (default)

Notes
-----
- If no precomputed feature embeddings are supplied, nearest neighbors are
  computed on a cheap image embedding: grayscale → resize(64x64) → flatten → L2.
- Works for grayscale or RGB; RGB is converted to grayscale.
- Designed to be *robust*: if anything is missing, the script will skip that class
  with a helpful message rather than crash.

Usage
-----
# Single class, 8 examples
python scripts/plots/qual/class_triptychs.py --model gan --class 5 --k 8

# Multiple classes
python scripts/plots/qual/class_triptychs.py --model diffusion --class 0 1 2 --k 6

# Custom real root and JSONL path
python scripts/plots/qual/class_triptychs.py \
  --model vae --class 3 --real-root "USTC-TFC2016_malware/real" \
  --jsonl artifacts/summaries/phase1_summaries.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scripts.plots._common import read_jsonl, savefig, PLOT_OUT


# ---------------------------- basic IO utils ---------------------------------

def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert HxWx{1,3} to float32 grayscale in [0,1]."""
    if arr.ndim == 2:
        x = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        x = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        x = arr[..., 0]
    x = x.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _load_img(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return _to_gray(np.asarray(im))
    except Exception:
        return None


def _cheap_embed(img: np.ndarray, size: int = 64) -> np.ndarray:
    """Grayscale image -> resized vector embedding."""
    im = Image.fromarray((img * 255.0).astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    return x.reshape(-1)


# ----------------- discover synth + real paths per class ---------------------

def _manifest_from_jsonl(df, model: str) -> Optional[Path]:
    if "model" not in df.columns or "manifest_path" not in df.columns:
        return None
    rows = df[df["model"].astype(str) == model]["manifest_path"].dropna()
    if not len(rows):
        return None
    p = Path(rows.iloc[-1])
    return p if p.exists() else None


def _synth_by_class_from_manifest(manifest: Path, class_id: str, limit: int) -> List[Path]:
    """Return up to `limit` image Paths for a given class_id from a flexible manifest schema."""
    try:
        data = json.loads(manifest.read_text())
    except Exception:
        return []

    def _class_of(rec) -> str:
        # accept 'class', 'class_id', 'label' as class indicators
        for k in ("class", "class_id", "label"):
            if isinstance(rec, dict) and k in rec:
                return str(rec[k]).strip()
        return ""

    def _coerce_path(p_like) -> Optional[Path]:
        if p_like is None:
            return None
        p = Path(p_like)
        if not p.is_absolute():
            p = (manifest.parent / p).resolve()
        return p if p.exists() else None

    rec_lists = []
    if isinstance(data, dict):
        for key in ("images", "paths", "samples"):
            if key in data and isinstance(data[key], list):
                rec_lists.append(data[key])
    elif isinstance(data, list):
        rec_lists.append(data)
    else:
        return []

    out: List[Path] = []
    for recs in rec_lists:
        for r in recs:
            if isinstance(r, dict):
                if _class_of(r) != str(class_id):
                    continue
                p = _coerce_path(r.get("path"))
                if p:
                    out.append(p)
            else:
                # bare string path; only keep if parent folder name == class_id
                p = _coerce_path(r)
                if p and p.parent.name == str(class_id):
                    out.append(p)
            if len(out) >= limit:
                return out
    return out[:limit]


def _fallback_synth_by_class(model: str, class_id: str, limit: int) -> List[Path]:
    root = Path(f"artifacts/{model}/synthetic")
    if not root.exists():
        return []
    # first: artifacts/<model>/synthetic/<class_id>/**
    if (root / class_id).exists():
        paths = list((root / class_id).rglob("*"))
        paths = [p for p in paths if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
        return paths[:limit]
    # otherwise search any file under synthetic/ that has class hint in parent folder
    candidates = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            if p.parent.name == class_id:
                candidates.append(p)
    return candidates[:limit]


def _real_by_class(real_root: Path, class_id: str, limit: int) -> List[Path]:
    out: List[Path] = []
    # common structure: real/<class_id>/**
    cdir = real_root / class_id
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if cdir.exists():
        for p in cdir.rglob("*"):
            if p.suffix.lower() in exts:
                out.append(p)
                if len(out) >= limit:
                    return out
    # fallback: search deeper
    for p in real_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and (p.parent.name == class_id):
            out.append(p)
            if len(out) >= limit:
                break
    return out[:limit]


# ------------------------- nearest neighbor matching -------------------------

def _nn_match(
    synth_imgs: List[np.ndarray],
    real_imgs: List[np.ndarray],
    embed_size: int = 64,
) -> List[int]:
    """Return indices of nearest real image for each synth image (L2 in embed space)."""
    if not synth_imgs or not real_imgs:
        return []
    synth_emb = np.stack([_cheap_embed(x, embed_size) for x in synth_imgs], axis=0)  # [S, D]
    real_emb  = np.stack([_cheap_embed(x, embed_size) for x in real_imgs], axis=0)   # [R, D]
    # compute dist^2 using (a-b)^2 = a^2 + b^2 - 2ab
    a2 = (synth_emb ** 2).sum(axis=1, keepdims=True)          # [S,1]
    b2 = (real_emb ** 2).sum(axis=1, keepdims=True).T         # [1,R]
    ab = synth_emb @ real_emb.T                                # [S,R]
    d2 = a2 + b2 - 2 * ab                                      # [S,R]
    nn_idx = np.argmin(d2, axis=1)
    return nn_idx.tolist()


# ------------------------------ plotting grid --------------------------------

def _plot_triptych(
    real_paths: List[Path],
    synth_paths: List[Path],
    nn_paths: List[Path],
    model: str,
    class_id: str,
    k: int,
) -> None:
    # limit to equal columns
    k = min(k, len(real_paths), len(synth_paths), len(nn_paths))
    if k == 0:
        print(f"[skip] {model} class={class_id}: not enough images to render.")
        return

    rows = 3
    cols = k
    fig_h = 3.2  # per row
    fig_w = 2.5 * max(4, min(10, cols)) / 6.0
    plt.figure(figsize=(fig_w * cols / max(cols, 6), fig_h * rows))

    # helper to draw a row
    def draw_row(paths: List[Path], row: int, title: str):
        for i in range(cols):
            ax = plt.subplot(rows, cols, row * cols + i + 1)
            img = _load_img(paths[i])
            if img is None:
                ax.axis("off")
                continue
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(title, fontsize=10, loc="left", pad=8)

    draw_row(real_paths[:k], 0, "Real")
    draw_row(synth_paths[:k], 1, "Synth")
    draw_row(nn_paths[:k],    2, "NN-Real")

    plt.suptitle(f"{model} — class {class_id}: Real | Synth | NN-Real", fontsize=12, y=0.995)
    out = PLOT_OUT / "qual" / f"class_triptych_{model}_c{class_id}_k{k}.png"
    savefig(out, tight=True, dpi=200)
    plt.close()


# ---------------------------------- main -------------------------------------

def _gather_for_class(
    model: str,
    class_id: str,
    k: int,
    df_jsonl,
    real_root: Path,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Return (real_paths, synth_paths, nn_real_paths) sized for plotting."""
    # synth via manifest or fallback
    mani = _manifest_from_jsonl(df_jsonl, model) if df_jsonl is not None else None
    if mani is not None:
        synth_paths = _synth_by_class_from_manifest(mani, class_id=str(class_id), limit=max(k, 32))
    else:
        synth_paths = _fallback_synth_by_class(model, class_id=str(class_id), limit=max(k, 32))

    real_paths = _real_by_class(real_root, class_id=str(class_id), limit=max(k, 256))

    # load images (use more real for NN search, then downselect)
    synth_imgs = [_load_img(p) for p in synth_paths]
    real_imgs  = [_load_img(p) for p in real_paths]
    synth_valid = [(p, x) for p, x in zip(synth_paths, synth_imgs) if x is not None]
    real_valid  = [(p, x) for p, x in zip(real_paths,  real_imgs)  if x is not None]

    if not synth_valid or not real_valid:
        return [], [], []

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(synth_valid)
        rng.shuffle(real_valid)

    # pick up to k synths
    synth_valid = synth_valid[:k]
    s_paths = [p for p, _ in synth_valid]
    s_imgs  = [x for _, x in synth_valid]

    # NN match among (more) real images for stability
    nn_idx = _nn_match(s_imgs, [x for _, x in real_valid], embed_size=64)
    nn_paths = [real_valid[j][0] for j in nn_idx]

    # also pick k real exemplars for the first row (just the first k)
    r_paths = [p for p, _ in real_valid[:k]]

    return r_paths, s_paths, nn_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Render class-conditioned triptychs (Real | Synth | NN-Real).")
    ap.add_argument("--model", required=True, help="Model family (e.g., gan, diffusion, vae, ...).")
    ap.add_argument("--class", dest="classes", nargs="+", required=True, help="One or more class IDs (e.g., 0 1 2).")
    ap.add_argument("--k", type=int, default=8, help="Images per row.")
    ap.add_argument("--jsonl", default="artifacts/summaries/phase1_summaries.jsonl", help="Consolidated JSONL path.")
    ap.add_argument("--real-root", default="USTC-TFC2016_malware/real", help="Folder containing REAL data by class.")
    ap.add_argument("--shuffle", action="store_true", help="Randomly sample examples (seeded).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed used when --shuffle.")
    args = ap.parse_args()

    # JSONL is optional but helps find manifests
    df = read_jsonl(args.jsonl)
    if df is not None and df.empty:
        df = None

    real_root = Path(args.real_root)

    for cls in args.classes:
        r_paths, s_paths, nn_paths = _gather_for_class(
            model=args.model,
            class_id=str(cls),
            k=args.k,
            df_jsonl=df,
            real_root=real_root,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        _plot_triptych(r_paths, s_paths, nn_paths, model=args.model, class_id=str(cls), k=args.k)


if __name__ == "__main__":
    main()
