# scripts/plots/imbalance/simple_stats_sanity.py
"""
Simple stats sanity checks: Real vs Synth

What
----
Low-level, label-free diagnostics comparing REAL and SYNTH samples:
  1) Pixel intensity histogram (normalized)
  2) Average radial Power Spectral Density (PSD)
  3) Edge-density distribution (Sobel gradient > threshold)

Why
---
Quick visual checks that often catch obvious artifacts:
band-limiting, over-smoothing, contrast mismatch, or copy/paste errors.

Inputs
------
- Synth paths are discovered from the per-model manifest if present, else via a
  glob fallback under artifacts/<model>/synthetic/.
- Real paths are inferred from --real-root (default: USTC-TFC2016_malware/real)
  with a glob of **/*.* (override with --real-glob).

Assumptions
-----------
- Images are single-channel or RGB; if RGB, they are converted to grayscale.
- File formats supported by Pillow (png/jpg).

Outputs
-------
PNG figures in artifacts/figures/imbalance/:
  simple_hist_<model>.png
  simple_psd_<model>.png
  simple_edges_<model>.png

Examples
--------
# Run for one model with defaults (samples 2k real + 2k synth if available)
python scripts/plots/imbalance/simple_stats_sanity.py --model gan

# Limit to 500 samples and provide an explicit real glob
python scripts/plots/imbalance/simple_stats_sanity.py \
  --model diffusion --num 500 --real-glob "USTC-TFC2016_malware/real/*/*.png"

# Run for every model found in JSONL
python scripts/plots/imbalance/simple_stats_sanity.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from scripts.plots._common import read_jsonl, savefig, PLOT_OUT


# ----------------------------- IO utilities ----------------------------------

def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert HxWx{1,3} to float32 grayscale in [0,1]."""
    if arr.ndim == 2:
        x = arr
    elif arr.ndim == 3 and arr.shape[2] == 1:
        x = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # RGB to gray: BT.601 luma
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        x = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        # Fallback: flatten last channel
        x = arr.reshape(arr.shape[0], arr.shape[1], -1)[..., 0]
    x = x.astype(np.float32)
    # normalize if likely 0..255
    if x.max() > 1.5:
        x /= 255.0
    return x


def _load_image(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return _to_gray(np.asarray(im))
    except Exception:
        return None


def _manifest_from_jsonl(df: pd.DataFrame, model: str) -> Optional[Path]:
    """Try to fetch manifest_path for the given model from JSONL rows."""
    if "model" not in df.columns or "manifest_path" not in df.columns:
        return None
    rows = df[df["model"].astype(str) == model]["manifest_path"].dropna()
    if not len(rows):
        return None
    p = Path(rows.iloc[-1])
    return p if p.exists() else None


def _synth_paths_from_manifest(manifest_path: Path, limit: int) -> List[Path]:
    """Parse manifest.json and return up to `limit` absolute Paths (best-effort)."""
    try:
        data = json.loads(manifest_path.read_text())
    except Exception:
        return []

    def _coerce_path(p: Path) -> Optional[Path]:
        if not p.is_absolute():
            p = (manifest_path.parent / p).resolve()
        return p if p.exists() else None

    paths: List[Path] = []

    # Case A: dict with known list keys
    if isinstance(data, dict):
        for key in ("images", "paths", "samples"):
            if key in data and isinstance(data[key], list):
                for rec in data[key]:
                    if isinstance(rec, dict) and "path" in rec:
                        p = _coerce_path(Path(rec["path"]))
                        if p:
                            paths.append(p)
                    elif isinstance(rec, (str, Path)):
                        p = _coerce_path(Path(rec))
                        if p:
                            paths.append(p)
                if paths:
                    break  # stop at first populated key

    # Case B: top-level list (of dicts or strings)
    if not paths and isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict) and "path" in rec:
                p = _coerce_path(Path(rec["path"]))
                if p:
                    paths.append(p)
            elif isinstance(rec, (str, Path)):
                p = _coerce_path(Path(rec))
                if p:
                    paths.append(p)

    return paths[:limit]


def _fallback_synth_glob(model: str, limit: int) -> List[Path]:
    root = Path(f"artifacts/{model}/synthetic")
    if not root.exists():
        return []
    # pick common image extensions
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files: List[Path] = []
    for pat in exts:
        files.extend(root.rglob(pat))
        if len(files) >= limit:
            break
    return files[:limit]


def _real_paths_from_glob(real_glob: str, limit: int) -> List[Path]:
    files = [Path(p) for p in sorted(map(str, Path().glob(real_glob)))]
    return files[:limit]


# --------------------------- simple statistics --------------------------------

def _radial_psd(img: np.ndarray) -> np.ndarray:
    """Return azimuthally averaged power spectral density for one image."""
    f = np.fft.fft2(img - img.mean())
    fshift = np.fft.fftshift(f)
    p = (np.abs(fshift) ** 2).astype(np.float64)

    h, w = p.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_int = r.astype(np.int32)
    max_r = r_int.max()
    radial_mean = np.zeros(max_r + 1, dtype=np.float64)
    for R in range(max_r + 1):
        m = r_int == R
        if m.any():
            radial_mean[R] = p[m].mean()
    # avoid zeros/neg for log; small epsilon
    return np.maximum(radial_mean, 1e-12)


def _edge_density(img: np.ndarray, thresh: float = 0.2) -> float:
    """Sobel-like gradient and fraction of pixels above threshold."""
    # kernels
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    # simple valid conv
    def conv2(x, k):
        H, W = x.shape
        out = np.zeros((H - 2, W - 2), dtype=np.float32)
        for i in range(H - 2):
            patch = x[i:i+3, 0:3]
            # slide horizontally with einsum to reduce Python loops
            # build 3x3 windows across width:
            win = np.lib.stride_tricks.as_strided(
                x[i:i+3],
                shape=(3, W - 2, 3),
                strides=(x.strides[0], x.strides[1], x.strides[1])
            )
            # einsum over (3,horiz,3) with (3,3) -> (horiz)
            out[i] = np.einsum("ijk,ij->k", win, k)  # k is 3x3, win is 3x(W-2)x3
        return out
    gx = conv2(img, kx)
    gy = conv2(img, ky)
    mag = np.hypot(gx, gy)
    # normalize gradient magnitude into [0,1] using percentile for robustness
    scale = np.percentile(mag, 99.0)
    if scale <= 1e-9:
        scale = 1.0
    gm = np.clip(mag / scale, 0.0, 1.0)
    return float((gm > thresh).mean())


# ------------------------------ plotting -------------------------------------

def _plot_hist(real_vals: np.ndarray, synth_vals: np.ndarray, model: str, bins: int) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(real_vals, bins=bins, density=True, alpha=0.6, label="Real")
    plt.hist(synth_vals, bins=bins, density=True, alpha=0.6, label="Synth")
    plt.xlabel("Pixel intensity (grayscale, [0,1])")
    plt.ylabel("Density")
    plt.title(f"Pixel intensity distribution — {model}")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(PLOT_OUT / "imbalance" / f"simple_hist_{model}.png")


def _plot_psd(real_psd: np.ndarray, synth_psd: np.ndarray, model: str) -> None:
    plt.figure(figsize=(7, 5))
    r = np.arange(len(real_psd))
    s = np.arange(len(synth_psd))
    # Use log scale on y; x is spatial frequency radius
    plt.plot(r, real_psd, label="Real")
    plt.plot(s, synth_psd, label="Synth")
    plt.yscale("log")
    plt.xlabel("Radial frequency (pixels)")
    plt.ylabel("Power (log scale)")
    plt.title(f"Average radial PSD — {model}")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(PLOT_OUT / "imbalance" / f"simple_psd_{model}.png")


def _plot_edge_density(real_ed: np.ndarray, synth_ed: np.ndarray, model: str) -> None:
    plt.figure(figsize=(7, 5))
    # side-by-side box/violin alternatives; here: hist overlay
    bins = max(10, int(np.sqrt(len(real_ed) + len(synth_ed))))
    plt.hist(real_ed, bins=bins, alpha=0.6, label="Real")
    plt.hist(synth_ed, bins=bins, alpha=0.6, label="Synth")
    plt.xlabel("Edge density (fraction of pixels)")
    plt.ylabel("Count")
    plt.title(f"Edge density distribution — {model}")
    plt.grid(alpha=0.25)
    plt.legend()
    savefig(PLOT_OUT / "imbalance" / f"simple_edges_{model}.png")


# ------------------------------- driver --------------------------------------

def _collect_paths_for_model(
    model: str,
    df: pd.DataFrame,
    num: int,
    real_root: str,
    real_glob: Optional[str],
) -> Tuple[List[Path], List[Path]]:
    """Return (real_paths, synth_paths) limited to `num` each."""
    # Synth
    synth_paths: List[Path] = []
    mani = _manifest_from_jsonl(df, model)
    if mani is not None:
        synth_paths = _synth_paths_from_manifest(mani, limit=num)
    if not synth_paths:
        synth_paths = _fallback_synth_glob(model, limit=num)

    # Real
    if real_glob:
        real_paths = _real_paths_from_glob(real_glob, limit=num)
    else:
        # default: everything under real_root
        rr = Path(real_root)
        # be generous about extensions
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        candidates: List[Path] = []
        for pat in exts:
            candidates.extend(rr.rglob(pat))
            if len(candidates) >= num:
                break
        real_paths = candidates[:num]

    return real_paths, synth_paths


def _load_sample_stack(paths: List[Path]) -> List[np.ndarray]:
    imgs: List[np.ndarray] = []
    for p in paths:
        arr = _load_image(p)
        if arr is not None:
            imgs.append(arr)
    return imgs


def _analyze_and_plot(model: str, real_imgs: List[np.ndarray], synth_imgs: List[np.ndarray], bins: int) -> None:
    if not real_imgs or not synth_imgs:
        print(f"[skip] model={model}: insufficient images (real={len(real_imgs)} synth={len(synth_imgs)})")
        return

    # 1) Pixel hist (sample up to ~100k pixels from each side for speed)
    def sample_vals(stk: List[np.ndarray], cap: int = 100_000) -> np.ndarray:
        flat = np.concatenate([x.reshape(-1) for x in stk])
        if flat.size > cap:
            idx = np.random.default_rng(0).choice(flat.size, size=cap, replace=False)
            flat = flat[idx]
        return flat

    real_vals = sample_vals(real_imgs)
    synth_vals = sample_vals(synth_imgs)
    _plot_hist(real_vals, synth_vals, model=model, bins=bins)

    # 2) PSD: average radial PSD across images (align to min length)
    def avg_psd(stk: List[np.ndarray]) -> np.ndarray:
        psds = [ _radial_psd(x) for x in stk ]
        L = min(len(p) for p in psds)
        if L <= 8:
            L = min(64, min(len(p) for p in psds))  # keep at least some points
        psds = [p[:L] for p in psds]
        return np.mean(psds, axis=0)

    real_psd = avg_psd(real_imgs)
    synth_psd = avg_psd(synth_imgs)
    _plot_psd(real_psd, synth_psd, model=model)

    # 3) Edge-density distribution
    real_ed = np.array([_edge_density(x) for x in real_imgs], dtype=np.float32)
    synth_ed = np.array([_edge_density(x) for x in synth_imgs], dtype=np.float32)
    _plot_edge_density(real_ed, synth_ed, model=model)


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple sanity plots: pixel hist, PSD, edge density (Real vs Synth).")
    ap.add_argument("--jsonl", default="artifacts/summaries/phase1_summaries.jsonl",
                    help="Path to consolidated JSONL for discovering model/manifest.")
    ap.add_argument("--model", default=None, help="Model family to analyze. Omit to run for all.")
    ap.add_argument("--all", action="store_true", help="Run for every model found in JSONL.")
    ap.add_argument("--num", type=int, default=2000, help="Max number of images to load for each split.")
    ap.add_argument("--bins", type=int, default=64, help="Histogram bins for pixel intensity.")
    ap.add_argument("--real-root", default="USTC-TFC2016_malware/real", help="Root folder containing REAL images.")
    ap.add_argument("--real-glob", default=None, help="Explicit glob for REAL images (e.g., '.../real/*/*.png').")
    args = ap.parse_args()

    df = read_jsonl(args.jsonl)
    if df.empty:
        return

    models = []
    if args.all or not args.model:
        if "model" in df.columns:
            models = sorted(df["model"].astype(str).unique())
        else:
            print("[skip] 'model' column not found in JSONL.")
            return
    else:
        models = [args.model]

    for m in models:
        real_paths, synth_paths = _collect_paths_for_model(
            model=m, df=df, num=args.num, real_root=args.real_root, real_glob=args.real_glob
        )
        real_imgs = _load_sample_stack(real_paths)
        synth_imgs = _load_sample_stack(synth_paths)
        _analyze_and_plot(m, real_imgs, synth_imgs, bins=args.bins)


if __name__ == "__main__":
    main()
