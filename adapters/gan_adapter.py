# adapters/gan_adapter.py

"""
GANAdapter
----------
Adapter that invokes the local *gan* sampler to generate class-conditional images
and emits a manifest that the evaluator can consume.

Happy path
----------
- Imports `gan.sample.synth`.
- Resolves output dir: {artifacts}/gan/synthetic
- Calls: synth(cfg, output_root, seed) → manifest dict
- Writes manifest to: {artifacts}/gan/synthetic/manifest.json

Fallback
--------
If import or sampling fails, a stub manifest (no images) is written so the
pipeline can continue, and a clear warning is printed.

Config keys (with safe defaults)
--------------------------------
IMG_SHAPE: [40, 40, 1]
NUM_CLASSES: 9
SAMPLES_PER_CLASS: 25
SEED: 42                # preferred single seed
random_seeds: [42, ...] # legacy fallback (first element used)
paths:
  artifacts: "artifacts"
DATA_DIR or data.root: persisted into the manifest as "dataset".
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import Adapter


# ------------------------------
# Small utilities
# ------------------------------
def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """Fetch a nested config value by dotted path, e.g. 'paths.artifacts'."""
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------
# Adapter
# ------------------------------
class GANAdapter(Adapter):
    """Adapter that calls the local GAN sampler to emit a manifest."""
    name = "gan"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "gan"
        synth_root = _ensure_dir(model_root / "synthetic")

        # Minimal knobs (mainly for stub manifest)
        H, W, C = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        K = int(_cfg_get(config, "NUM_CLASSES", 9))

        # Seed: prefer SEED, else first from random_seeds, else 42
        if "SEED" in config:
            seed = int(config["SEED"])
        else:
            seed = int(_cfg_get(config, "random_seeds", [42])[0])

        dataset = _cfg_get(config, "data.root", config.get("DATA_DIR", "USTC-TFC2016_40x40_gray"))

        # Default (stub) manifest structure; will be replaced on success
        manifest: Dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "per_class_counts": {str(k): 0 for k in range(K)},
            "paths": [],  # {"path": "...", "label": int}
        }

        try:
            # Keep import local so adapter is importable even if GAN package isn't present yet
            from gan.sample import synth as gan_synth  # type: ignore

            # Deterministic sampling
            np.random.seed(seed)
            try:
                import tensorflow as tf  # set TF seed if available
                tf.random.set_seed(seed)
            except Exception:
                pass

            print(f"[gan] HWC={H,W,C}  K={K}  seed={seed}")
            man = gan_synth(config, str(synth_root), seed=seed)

            # Normalize to plain dict & use it as our manifest
            manifest = dict(man)

        except Exception as e:
            # Fallback: emit stub manifest and warn clearly
            print(f"[gan][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[gan] Emitting a stub manifest so the pipeline can proceed.")

        # Persist manifest (always write something)
        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[gan] Wrote manifest → {man_path}")

        return manifest
