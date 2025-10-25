# scripts/utils/npy_to_real_pngs.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npy_to_real_pngs.py
-------------------
Materialize PNG images by class from USTC-TFC2016 .npy arrays so downstream
plotters that expect on-disk "real" images can run.

Accepts common array layouts:
- (H, W), (H, W, 1), (1, H, W)
- flat length H*W
- color-like shapes (H, W, 3/4) or (3/4, H, W) → averaged to grayscale

Handles data ranges:
- [-1, 1] → scaled to [0, 255]
- [0, 1]  → scaled to [0, 255]
- [0, 255] assumed already bytes

Output layout:
<out_root>/<class_id>/*.png

Usage (example)
---------------
python scripts/utils/npy_to_real_pngs.py \
  --root USTC-TFC2016_malware \
  --out  USTC-TFC2016_malware/real \
  --split both \
  --limit-per-class 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def as_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert many possible array shapes into a 2D uint8 grayscale image.

    Accepts:
      (H,W), (H,W,1), (1,H,W), flat (H*W,),
      (H,W,3/4), (3/4,H,W)  → averaged to grayscale.

    Returns:
      np.ndarray of shape (H, W), dtype=uint8
    """
    arr = np.asarray(img)

    # Remove trivial singleton dims first (e.g., (1,H,W,1) → (H,W))
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        # Assume square if flat
        L = arr.shape[0]
        s = int(np.sqrt(L))
        if s * s != L:
            raise ValueError(f"Flat vector length {L} is not a perfect square.")
        arr = arr.reshape(s, s)

    elif arr.ndim == 3:
        # Handle channel-first or channel-last
        # Try channel-first (C,H,W)
        if arr.shape[0] in (1, 3, 4):
            C = arr.shape[0]
            if C == 1:
                arr = arr[0]
            else:
                # average first 3 channels
                arr = arr[:3].mean(axis=0)
        # Else try channel-last (H,W,C)
        elif arr.shape[-1] in (1, 3, 4):
            C = arr.shape[-1]
            if C == 1:
                arr = arr[..., 0]
            else:
                arr = arr[..., :3].mean(axis=-1)
        else:
            # Unknown 3D layout → best-effort squeeze again or take first slice
            arr = np.squeeze(arr)
            if arr.ndim == 3:
                arr = arr[..., 0]

    # Expect 2D now
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D after processing, got {arr.shape}.")

    # Normalize to [0,255] if needed
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= 1.0 and mn >= -1.0:
        # Could be [-1,1] or [0,1]
        if mn < 0.0:
            arr = (arr + 1.0) * 127.5
        else:
            arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_split(
    data_path: Path,
    labels_path: Path,
    out_root: Path,
    limit_per_class: Optional[int],
    offset: int,
    prefix: str,
) -> Dict[int, int]:
    """
    Write images for one split (train/test).

    Returns:
      dict[class_id] -> written_count
    """
    X = np.load(data_path)
    y = np.load(labels_path)
    if len(X) != len(y):
        raise ValueError(
            f"Data/labels length mismatch: {len(X)} vs {len(y)} "
            f"({data_path.name} / {labels_path.name})"
        )

    classes = np.unique(y.astype(int)).tolist()
    per_class_written: Dict[int, int] = {int(c): 0 for c in classes}

    for i, (img, lab) in enumerate(zip(X, y)):
        c = int(lab)
        if limit_per_class is not None and per_class_written[c] >= limit_per_class:
            continue

        out_dir = out_root / str(c)
        ensure_dir(out_dir)

        # Pillow infers "L" mode from 2D uint8; no explicit mode arg (avoid deprecation)
        im = Image.fromarray(as_uint8(img))

        # Unique filename to avoid collisions across splits
        fn = out_dir / f"{prefix}_{c}_{offset + i:07d}.png"
        im.save(fn)

        per_class_written[c] += 1

    return per_class_written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Materialize real PNGs by class from USTC-TFC2016 .npy splits."
    )
    ap.add_argument(
        "--root",
        default="USTC-TFC2016_malware",
        help="Folder containing train/test .npy files",
    )
    ap.add_argument(
        "--out",
        default="USTC-TFC2016_malware/real",
        help="Output root: <out>/<class_id>/*.png",
    )
    ap.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which splits to export",
    )
    ap.add_argument(
        "--limit-per-class",
        type=int,
        default=2000,
        help="Max PNGs per class per split (use a smaller value to save disk/time).",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Validate expected files exist
    expected = []
    if args.split in ("train", "both"):
        expected += [root / "train_data.npy", root / "train_labels.npy"]
    if args.split in ("test", "both"):
        expected += [root / "test_data.npy", root / "test_labels.npy"]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required .npy files:\n  " + "\n  ".join(missing))

    written_by_split: Dict[str, Dict[int, int]] = {}
    offset = 0

    if args.split in ("train", "both"):
        w = save_split(
            root / "train_data.npy",
            root / "train_labels.npy",
            out_root,
            args.limit_per_class,
            offset,
            "train",
        )
        written_by_split["train"] = w
        offset += sum(w.values())

    if args.split in ("test", "both"):
        w = save_split(
            root / "test_data.npy",
            root / "test_labels.npy",
            out_root,
            args.limit_per_class,
            offset,
            "test",
        )
        written_by_split["test"] = w

    # Pretty summary
    total: Dict[int, int] = {}
    for split_name, per_cls in written_by_split.items():
        for c, n in per_cls.items():
            total[c] = total.get(c, 0) + n

    print("[ok] PNG export complete.")
    for c in sorted(total):
        parts = []
        for split_name in ("train", "test"):
            n = written_by_split.get(split_name, {}).get(c, 0)
            parts.append(f"{split_name}:{n}")
        print(f"  class {c}: {total[c]} ({' | '.join(parts)})")

    print(f"[hint] Real root for plots: {out_root.resolve()}")


if __name__ == "__main__":
    main()
