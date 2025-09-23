# adapters/restrictedboltzmann_adapter.py

"""
RBMAdapter
----------
Adapter that invokes the local *restrictedboltzmann* (RBM) sampler to generate
class-conditional samples and writes a manifest the evaluator can consume.

Happy path
----------
- Imports `restrictedboltzmann.sample.synth`.
- Resolves output dir: {artifacts}/restrictedboltzmann/synthetic
- Calls: synth(cfg, output_root, seed) → manifest dict
- Writes manifest to: {artifacts}/restrictedboltzmann/synthetic/manifest.json

Fallback
--------
If import or sampling fails, a stub manifest (no images) is written so the
pipeline can continue, and a clear warning is printed.

Config keys (safe defaults)
---------------------------
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
class RBMAdapter(Adapter):
    """Adapter that calls the local Restricted Boltzmann Machine sampler."""
    name = "restrictedboltzmann"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "restrictedboltzmann"
        synth_root = _ensure_dir(model_root / "synthetic")

        # Minimal knobs (used for stub and logging)
        H, W, C = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        K = int(_cfg_get(config, "NUM_CLASSES", 9))

        # Seed preference: SEED, else first from random_seeds, else 42
        if "SEED" in config:
            seed = int(config["SEED"])
        else:
            seed = int(_cfg_get(config, "random_seeds", [42])[0])

        dataset = _cfg_get(config, "data.root", config.get("DATA_DIR", "USTC-TFC2016_40x40_gray"))

        # Default stub manifest; replaced on success
        manifest: Dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "per_class_counts": {str(k): 0 for k in range(K)},
            "paths": [],  # each: {"path": "...", "label": int}
        }

        try:
            # Import locally so the adapter remains importable if package not present yet
            from restrictedboltzmann.sample import synth as rbm_synth  # type: ignore

            # Deterministic runs
            np.random.seed(seed)
            try:
                import tensorflow as tf  # if available, sync TF RNG as well
                tf.random.set_seed(seed)
            except Exception:
                pass

            print(f"[restrictedboltzmann] HWC={H,W,C}  K={K}  seed={seed}")
            man = rbm_synth(config, str(synth_root), seed=seed)

            # Normalize to plain dict & use as manifest
            manifest = dict(man)

        except Exception as e:
            print(f"[restrictedboltzmann][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[restrictedboltzmann] Emitting a stub manifest so the pipeline can proceed.")

        # Persist manifest (always write something)
        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[restrictedboltzmann] Wrote manifest → {man_path}")

        return manifest
