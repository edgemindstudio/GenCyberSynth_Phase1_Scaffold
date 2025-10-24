# scripts/plots/qual/grids_panels.py
"""
Multi-model grid panels (overview collage)

What
----
Loads the per-model preview grids you already generate (e.g.,
artifacts/preview_grids/<model>_grid.png) and assembles them into a single,
clean collage with titles. This is the quick, paper-ready overview panel.

Why
---
Reviewers love a “family photo.” This lets you compare visual characteristics
across model families at a glance without re-sampling images.

Inputs (auto-discovered)
------------------------
- Preview grids (preferred): artifacts/preview_grids/<model>_grid.png

If a given model’s grid is missing, the script will skip it (with a warning) so
you can still build a partial collage.

Usage
-----
# Default models, 3 columns
python scripts/plots/qual/grids_panels.py

# Custom set and layout
python scripts/plots/qual/grids_panels.py \
  --models gan diffusion vae maskedautoflow \
  --cols 2 --out overview_grids.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scripts.plots._common import savefig, PLOT_OUT


PREVIEW_DIR = Path("artifacts/preview_grids")
DEFAULT_MODELS = [
    "gan",
    "diffusion",
    "vae",
    "autoregressive",
    "maskedautoflow",
    "restrictedboltzmann",
    "gaussianmixture",
]


def _find_model_grids(models: List[str]) -> List[Tuple[str, Path]]:
    """
    Return list of (model, path) for preview grids that exist.
    """
    found = []
    for m in models:
        p = PREVIEW_DIR / f"{m}_grid.png"
        if p.exists():
            found.append((m, p))
        else:
            print(f"[warn] missing preview grid for model={m}: {p}")
    return found


def _load_png(path: Path) -> np.ndarray:
    """
    Load PNG/JPG with PIL and return float array in [0,1].
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32)
        if arr.max() > 1.5:
            arr /= 255.0
        return np.clip(arr, 0.0, 1.0)


def _plot_collage(entries: List[Tuple[str, Path]], cols: int, title: str, out_name: str) -> None:
    """
    Create a collage with `cols` columns from list of (model, image_path).
    Each cell shows the grid image with a small model title above it.
    """
    n = len(entries)
    if n == 0:
        print("[skip] No grids found to plot.")
        return

    cols = max(1, cols)
    rows = int(np.ceil(n / cols))

    # Size heuristics: scale by image aspect but cap total figure size
    # Load one image to infer base size
    sample_img = _load_png(entries[0][1])
    img_h, img_w = sample_img.shape[:2]
    aspect = img_w / max(1, img_h)

    # Choose per-cell size that looks good on paper/slides
    cell_w = 4.0  # inches
    cell_h = cell_w / max(0.5, aspect)
    fig_w = cell_w * cols
    fig_h = cell_h * rows + 0.4 * rows  # extra space for small titles per cell

    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.subplots_adjust(hspace=0.35, wspace=0.08, top=0.95)

    for idx, (model, path) in enumerate(entries):
        r = idx // cols
        c = idx % cols
        ax = plt.subplot(rows, cols, idx + 1)
        img = _load_png(path)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(model, fontsize=11, pad=6, loc="left")

    if title:
        fig.suptitle(title, fontsize=13)

    out_path = PLOT_OUT / "qual" / out_name
    savefig(out_path, tight=True, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble per-model preview grids into a single collage.")
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Model families to include (default: {' '.join(DEFAULT_MODELS)})",
    )
    ap.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns in the collage (default: 3).",
    )
    ap.add_argument(
        "--title",
        default="Preview Grids by Model",
        help="Figure title (default: 'Preview Grids by Model').",
    )
    ap.add_argument(
        "--out",
        default="overview_grids.png",
        help="Output filename under artifacts/figures/qual/ (default: overview_grids.png).",
    )
    args = ap.parse_args()

    entries = _find_model_grids(args.models)
    _plot_collage(entries, cols=args.cols, title=args.title, out_name=args.out)


if __name__ == "__main__":
    main()
