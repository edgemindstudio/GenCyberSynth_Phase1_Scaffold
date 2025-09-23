# adapters/maskedautoflow_adapter.py

"""
MAFAdapter
----------
Adapter that invokes the local *maskedautoflow* (MAF) sampler to generate
class-conditional samples and writes a manifest the evaluator can consume.

Happy path
----------
- Imports `maskedautoflow.sample.synth`.
- Resolves output dir: {artifacts}/maskedautoflow/synthetic
- Calls: synth(cfg, output_root, seed) → manifest dict
- Writes manifest to: {artifacts}/maskedautoflow/synthetic/manifest.json

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
class MAFAdapter(Adapter):
    """Adapter that calls the local Masked Auto-Flow sampler to emit a manifest."""
    name = "maskedautoflow"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "maskedautoflow"
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
            from maskedautoflow.sample import synth as maf_synth  # type: ignore

            # Deterministic runs
            np.random.seed(seed)
            try:
                import tensorflow as tf  # if available, sync TF RNG as well
                tf.random.set_seed(seed)
            except Exception:
                pass

            print(f"[maskedautoflow] HWC={H,W,C}  K={K}  seed={seed}")
            man = maf_synth(config, str(synth_root), seed=seed)

            # Normalize to plain dict & use as manifest
            manifest = dict(man)

        except Exception as e:
            print(f"[maskedautoflow][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[maskedautoflow] Emitting a stub manifest so the pipeline can proceed.")

        # Persist manifest (always write something)
        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[maskedautoflow] Wrote manifest → {man_path}")

        return manifest
