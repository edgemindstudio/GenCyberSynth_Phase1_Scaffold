# scripts/plots/diversity/umap_projection.py
#!/usr/bin/env python
"""
UMAP/PCA projection of feature embeddings (real vs synth)

Looks under:
  artifacts/<model>/features/real_class*.npy
  artifacts/<model>/features/synth_class*.npy

Each .npy is expected to be (N, D). We sample up to --max-per-class from each
file, stack, optionally reduce with UMAP (fallback PCA), and scatter plot:
  • color = real vs synth
  • shape = (optionally) class (light annotation)
  • per-model grid with --per-model

Outputs
-------
- artifacts/figures/diversity/umap_projection.png
- artifacts/figures/diversity/umap_projection_per_model.png  (with --per-model)

Notes
-----
- If UMAP isn't installed, PCA is used automatically.
- If no features are found, the script prints [skip] and returns.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# No heavy pandas dependency needed here
try:
    import umap  # type: ignore
    _HAVE_UMAP = True
except Exception:
    _HAVE_UMAP = False

try:
    from sklearn.decomposition import PCA  # type: ignore
except Exception:
    PCA = None  # type: ignore

# Figure root from _common (supports both names)
try:
    from scripts.plots._common import PLOT_OUT_DIR as _FIG_ROOT
except Exception:
    from scripts.plots._common import PLOT_OUT as _FIG_ROOT  # type: ignore[assignment]
FIG_ROOT = Path(_FIG_ROOT)
(FIG_ROOT / "diversity").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

_PAT = re.compile(r"^(real|synth)_class(\d+)\.npy$")

def _scan_feature_dir(model_dir: Path) -> List[Tuple[str, str, Path]]:
    """
    Return list of (source, class_id, path) in a model's features directory.
      source ∈ {"real","synth"}
      class_id is the digits after 'class'
    """
    feats = model_dir / "features"
    out: List[Tuple[str, str, Path]] = []
    if not feats.exists():
        return out
    for p in feats.iterdir():
        if not p.is_file():
            continue
        m = _PAT.match(p.name)
        if m:
            src, cls = m.group(1), m.group(2)
            out.append((src, cls, p))
    return out


def _load_sampled(path: Path, k: int, rng: np.random.Generator) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)
        n = arr.shape[0]
        if k is not None and k > 0 and n > k:
            idx = rng.choice(n, size=k, replace=False)
            arr = arr[idx]
        return arr.astype(np.float32, copy=False)
    except Exception:
        return np.zeros((0, 0), dtype=np.float32)


def _collect_all(feature_root: Path, max_per_class: int, seed: int) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """
    Walk artifacts/*/features/*.npy and return:
      X  : (N, D) stacked features
      ys : list of "real"/"synth" per row
      cs : list of class_id (str) per row
      ms : list of model name per row
    """
    rng = np.random.default_rng(seed)
    Xs: List[np.ndarray] = []
    ys: List[str] = []
    cs: List[str] = []
    ms: List[str] = []

    for model_dir in sorted((feature_root).glob("*")):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        entries = _scan_feature_dir(model_dir)
        if not entries:
            continue

        # bucket by (source, class)
        by_key: Dict[Tuple[str, str], List[Path]] = {}
        for src, cls, p in entries:
            by_key.setdefault((src, cls), []).append(p)

        for (src, cls), paths in by_key.items():
            # Support multiple chunk files; sample from each until we reach approx limit
            remain = max_per_class
            for p in paths:
                take = remain if remain > 0 else None
                chunk = _load_sampled(p, take, rng)
                if chunk.size == 0:
                    continue
                Xs.append(chunk)
                ys.extend([src] * chunk.shape[0])
                cs.extend([cls] * chunk.shape[0])
                ms.extend([model] * chunk.shape[0])
                if remain is not None:
                    remain -= chunk.shape[0]
                    if remain <= 0:
                        break

    if not Xs:
        return np.zeros((0, 0), dtype=np.float32), [], [], []
    X = np.concatenate(Xs, axis=0)
    return X, ys, cs, ms


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def _reduce_2d(X: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> Tuple[np.ndarray, str]:
    if X.size == 0 or X.ndim != 2:
        return np.zeros((0, 2), dtype=np.float32), "empty"

    if _HAVE_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric="euclidean",
            random_state=seed,
            verbose=False,
        )
        Z = reducer.fit_transform(X)
        return Z.astype(np.float32, copy=False), "umap"
    else:
        if PCA is None:
            return np.zeros((0, 2), dtype=np.float32), "none"
        Z = PCA(n_components=2, random_state=seed).fit_transform(X)
        return Z.astype(np.float32, copy=False), "pca"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter(ax, Z: np.ndarray, ys: List[str], cs: List[str], title: str) -> None:
    if Z.size == 0:
        ax.text(0.5, 0.5, "no features", ha="center", va="center")
        ax.set_title(title)
        return
    srcs = np.array(ys)
    # Colors for source
    color_map = {"real": "#1f77b4", "synth": "#ff7f0e"}
    for src in ["real", "synth"]:
        idx = np.where(srcs == src)[0]
        if idx.size == 0:
            continue
        ax.scatter(Z[idx, 0], Z[idx, 1], s=8, alpha=0.75, edgecolors="none",
                   label=src, c=color_map.get(src, "#7f7f7f"))
    ax.legend(frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def plot_combined(X: np.ndarray, ys: List[str], cs: List[str], ms: List[str],
                  *, n_neighbors: int, min_dist: float, seed: int,
                  out: Path) -> None:
    Z, method = _reduce_2d(X, n_neighbors, min_dist, seed)
    plt.figure(figsize=(7.6, 5.0))
    ax = plt.gca()
    _scatter(ax, Z, ys, cs, f"Feature projection ({method}) — all models")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    from scripts.plots._common import savefig
    savefig(out)


def plot_per_model(X: np.ndarray, ys: List[str], cs: List[str], ms: List[str],
                   *, n_neighbors: int, min_dist: float, seed: int,
                   out: Path, cols: int = 3) -> None:
    if len(ms) == 0:
        print("[skip] no features to plot per-model")
        return
    Z, method = _reduce_2d(X, n_neighbors, min_dist, seed)
    if Z.size == 0:
        print("[skip] empty embedding")
        return
    models = sorted(set(ms))
    cols = max(1, int(cols))
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)
    arr_ms = np.array(ms)
    for i, m in enumerate(models):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        idx = np.where(arr_ms == m)[0]
        _scatter(ax, Z[idx], [ys[j] for j in idx], [cs[j] for j in idx], f"{m} ({method})")
    # hide leftover axes
    for i in range(len(models), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")
    plt.tight_layout()
    from scripts.plots._common import savefig
    savefig(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="UMAP/PCA projection of real vs synth feature embeddings")
    ap.add_argument("--feature-root", default="artifacts", help="Root dir that contains <model>/features/")
    ap.add_argument("--max-per-class", type=int, default=200, help="Max samples to load from each (source, class, file)")
    ap.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors (ignored for PCA fallback)")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (ignored for PCA fallback)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling / reducers")
    ap.add_argument("--per-model", action="store_true", help="Also emit a per-model grid figure")
    ap.add_argument("--per-model-cols", type=int, default=3, help="Columns in per-model grid")
    ap.add_argument("--out", default=str(FIG_ROOT / "diversity" / "umap_projection.png"))
    ap.add_argument("--out-per-model", default=str(FIG_ROOT / "diversity" / "umap_projection_per_model.png"))
    args = ap.parse_args()

    root = Path(args.feature_root)
    X, ys, cs, ms = _collect_all(root, max_per_class=args.max_per_class, seed=args.seed)

    if X.size == 0:
        print("[skip] no feature .npy files found under artifacts/*/features")
        return

    plot_combined(X, ys, cs, ms,
                  n_neighbors=args.neighbors, min_dist=args.min_dist, seed=args.seed,
                  out=Path(args.out))

    if args.per_model:
        plot_per_model(X, ys, cs, ms,
                       n_neighbors=args.neighbors, min_dist=args.min_dist, seed=args.seed,
                       out=Path(args.out_per_model), cols=args.per_model_cols)


if __name__ == "__main__":
    main()
