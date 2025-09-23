# adapters/autoregressive_adapter.py

"""
AutoregressiveAdapter
---------------------
Adapter that calls the local *autoregressive* sampler to generate class-conditional
images and write a manifest the evaluator can consume.

Happy path
----------
- Imports `autoregressive.sample.synth`.
- Resolves output directory: {artifacts}/autoregressive/synthetic
- Calls: synth(cfg, output_root, seed)  → manifest dict
- Persists manifest to: {artifacts}/autoregressive/synthetic/manifest.json

Fallback
--------
If import or sampling fails, it emits a stub manifest (no images) so the pipeline
doesn’t break, and prints a clear warning.

Config keys (with safe defaults)
--------------------------------
IMG_SHAPE: [40, 40, 1]
NUM_CLASSES: 9
SAMPLES_PER_CLASS: 25
SEED: 42               # preferred single seed
random_seeds: [42, ...]  # legacy fall-back (first element used)
paths:
  artifacts: "artifacts"
DATA_DIR or data.root: used as 'dataset' tag inside the manifest.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import Adapter


# ------------------------------
# Small config/file utilities
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
class AutoregressiveAdapter(Adapter):
    """Adapter that calls the local AR PixelCNN-style sampler to emit a manifest."""
    name = "autoregressive"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "autoregressive"
        synth_root = _ensure_dir(model_root / "synthetic")

        # Basic knobs with robust fallbacks (kept mainly for stub manifest)
        H, W, C = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        K = int(_cfg_get(config, "NUM_CLASSES", 9))

        # Seed: prefer SEED, else first from random_seeds, else 42
        if "SEED" in config:
            seed = int(config["SEED"])
        else:
            seed = int(_cfg_get(config, "random_seeds", [42])[0])

        dataset = _cfg_get(config, "data.root", config.get("DATA_DIR", "USTC-TFC2016_40x40_gray"))

        # Default manifest scaffold (will be overwritten on success)
        manifest: Dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "per_class_counts": {str(k): 0 for k in range(K)},
            "paths": [],  # {"path": "...", "label": int}
        }

        try:
            # Imports are scoped so the adapter itself remains importable
            # even if the AR package isn’t present yet.
            from autoregressive.sample import synth as ar_synth  # type: ignore

            # Deterministic sampling
            np.random.seed(seed)
            try:
                import tensorflow as tf  # ensure TF seed set if available
                tf.random.set_seed(seed)
            except Exception:
                pass

            # Call the project’s AR synth entrypoint
            print(f"[autoregressive] HWC={H,W,C}  K={K}  seed={seed}")
            man = ar_synth(config, str(synth_root), seed=seed)

            # Persist (normalize to plain dict for safety)
            manifest = dict(man)

        except Exception as e:
            # Fallback: emit a stub manifest and clearly warn
            print(f"[autoregressive][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[autoregressive] Emitting a stub manifest so the pipeline can proceed.")

        # Write manifest to disk (always)
        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[autoregressive] Wrote manifest → {man_path}")

        return manifest
