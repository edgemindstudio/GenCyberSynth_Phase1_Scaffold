# adapters/diffusion_adapter.py

"""
DiffusionAdapter
----------------
Production-ready adapter that generates class-conditional samples with the
*diffusion* backend and writes a manifest the evaluator can consume.

It tries the following in order:

1) Preferred path: use your local diffusion implementation
   - Builds the ε-prediction UNet via `diffusion.models.build_diffusion_model`.
   - Loads weights if a checkpoint is found (best → last → legacy).
   - Samples S images per class via `diffusion.sample.sample_batch`.
   - Saves PNGs under:  {artifacts}/diffusion/synthetic/<class>/<seed>/...
   - Writes a JSON manifest to: {artifacts}/diffusion/synthetic/manifest.json

2) Fallback: if anything critical fails (imports, build, etc.), it will still
   emit a stub manifest so the pipeline doesn’t crash. The stub contains zero
   paths and empty per-class counts, plus a warning printed to stdout.

Config keys (with safe defaults)
--------------------------------
IMG_SHAPE: [40, 40, 1]
NUM_CLASSES: 9
SAMPLES_PER_CLASS: 25
SEED: 42
paths:
  artifacts: "artifacts"
diffusion:                # (all optional)
  steps: 200              # reverse steps T for sampling previews
  base_filters: 64
  depth: 2
  time_emb_dim: 128
  lr: 2e-4
  beta_1: 0.9
ARTIFACTS:
  diffusion_checkpoints: "artifacts/diffusion/checkpoints"   # override location
DATA_DIR or data.root: string written back into the manifest as "dataset".
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .base import Adapter

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """
    Fetch a nested config value by dotted path, e.g. "paths.artifacts".
    """
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    img = np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)
    return img


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    """
    Save a single HxW[xC] image in [0,1] as PNG. Tries Pillow, falls back to matplotlib.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        x = img01
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
            mode = "L"
        elif x.ndim == 3 and x.shape[-1] == 3:
            mode = "RGB"
        else:
            x = x.squeeze()
            mode = "L"
        Image.fromarray(_to_uint8(x), mode=mode).save(out_path)
    except Exception:
        # Minimal fallback using matplotlib (slower; avoids adding hard deps)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(1.6, 1.6))
        if img01.ndim == 3 and img01.shape[-1] == 1:
            plt.imshow(img01[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(np.clip(img01, 0.0, 1.0))
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()


# ---------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------
class DiffusionAdapter(Adapter):
    """Adapter that calls the local diffusion sampler to emit a manifest."""
    name = "diffusion"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate S images per class using the diffusion backend and
        return the manifest dictionary that was written to disk.
        """
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "diffusion"
        synth_root = _ensure_dir(model_root / "synthetic")

        # Basic knobs (with robust fallbacks)
        H, W, C = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        K = int(_cfg_get(config, "NUM_CLASSES", 9))
        S = int(_cfg_get(config, "SAMPLES_PER_CLASS", 25))
        SEED = int(config.get("SEED", 42))

        # Diffusion hyperparams (sampling-side)
        T = int(_cfg_get(config, "diffusion.steps", 200))
        base_filters = int(_cfg_get(config, "diffusion.base_filters", 64))
        depth = int(_cfg_get(config, "diffusion.depth", 2))
        time_emb_dim = int(_cfg_get(config, "diffusion.time_emb_dim", 128))
        lr = float(_cfg_get(config, "diffusion.lr", 2e-4))
        beta_1 = float(_cfg_get(config, "diffusion.beta_1", 0.9))

        # Checkpoints
        default_ckpt_dir = artifacts_root / "diffusion" / "checkpoints"
        ckpt_dir = Path(_cfg_get(config, "ARTIFACTS.diffusion_checkpoints", default_ckpt_dir))
        candidates = [
            ckpt_dir / "DIFF_best.weights.h5",
            ckpt_dir / "DIFF_last.weights.h5",
            ckpt_dir / "diffusion_best.h5",   # legacy
            ckpt_dir / "diffusion_last.h5",   # legacy
        ]
        weights_path = next((p for p in candidates if p.exists()), None)

        # Dataset tag for the manifest
        dataset = _cfg_get(config, "data.root", config.get("DATA_DIR", "USTC-TFC2016_40x40_gray"))

        # Manifest scaffold
        manifest: Dict[str, Any] = {
            "dataset": dataset,
            "seed": SEED,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "per_class_counts": {str(k): 0 for k in range(K)},
            "paths": [],  # list of {"path": "...", "label": int}
        }

        # Attempt the full sampling path; on failure, emit a stub manifest
        try:
            # Imports are scoped so that this adapter remains importable even
            # if diffusion packages aren’t installed yet.
            import tensorflow as tf
            from diffusion.models import build_diffusion_model  # type: ignore
            from diffusion.sample import sample_batch          # type: ignore

            # Build model
            model = build_diffusion_model(
                img_shape=(H, W, C),
                num_classes=K,
                base_filters=base_filters,
                depth=depth,
                time_emb_dim=time_emb_dim,
                learning_rate=lr,
                beta_1=beta_1,
            )

            # Ensure variables exist before loading weights (Keras 3 safety)
            _ = model(
                [
                    tf.zeros((1, H, W, C), dtype=tf.float32),
                    tf.one_hot([0], depth=K, dtype=tf.float32),
                    tf.constant([0], dtype=tf.int32),
                ],
                training=False,
            )

            if weights_path:
                try:
                    model.load_weights(str(weights_path))
                    print(f"[diffusion] Loaded checkpoint: {weights_path}")
                except Exception as e:
                    print(f"[diffusion][warn] Failed to load {weights_path.name}: {e}\n"
                          f"→ continuing with randomly initialized weights.")
            else:
                print(f"[diffusion][warn] no DDPM checkpoint in {ckpt_dir}; using random weights.")

            # Deterministic sampling
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            # Generate per class
            for k in range(K):
                class_ids = np.full((S,), k, dtype=np.int32)
                imgs01, _ = sample_batch(
                    model,
                    num_samples=S,
                    num_classes=K,
                    img_shape=(H, W, C),
                    T=T,
                    alpha_hat=None,       # let sampler build a linear schedule
                    class_ids=class_ids,
                    seed=SEED + k,        # small offset per class
                )

                cls_dir = synth_root / str(k) / str(SEED)
                _ensure_dir(cls_dir)
                for j in range(S):
                    out_path = cls_dir / f"diff_{j:05d}.png"
                    _save_png(imgs01[j], out_path)
                    manifest["paths"].append({"path": str(out_path.resolve()), "label": int(k)})

                manifest["per_class_counts"][str(k)] = int(S)

        except Exception as e:
            # Fallback: emit a stub manifest and clearly warn
            print(f"[diffusion][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[diffusion] Emitting a stub manifest so the pipeline can proceed.")

        # Persist manifest
        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[diffusion] Wrote manifest → {man_path}")

        return manifest
