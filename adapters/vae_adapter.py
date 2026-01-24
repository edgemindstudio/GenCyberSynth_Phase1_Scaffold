# adapters/vae_adapter.py

"""
VAEAdapter
----------
Adapter that invokes the local *vae* sampler to generate class-conditional samples
and writes a manifest the evaluator can consume.

Happy path
----------
- Imports `vae.sample.synth`.
- Resolves output dir: {artifacts}/vae/synthetic
- Calls: synth(cfg, output_root, seed) → manifest dict
- Writes manifest to: {artifacts}/vae/synthetic/manifest.json

Fallback
--------
If import or sampling fails, a stub manifest (no images) is written so the
pipeline can continue, with a clear warning.

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


class VAEAdapter(Adapter):
    """Adapter that calls the local VAE sampler."""
    name = "vae"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "vae"
        synth_root = _ensure_dir(model_root / "synthetic")

        # For stub + logging
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
            # Local import keeps adapter importable if package isn't present yet
            from vae.sample import synth as vae_synth  # type: ignore

            # Deterministic runs
            np.random.seed(seed)
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
            except Exception:
                pass

            print(f"[vae] HWC={H,W,C}  K={K}  seed={seed}")
            man = vae_synth(config, str(synth_root), seed=seed)

            # Normalize to a plain dict in case a custom mapping is returned
            manifest = dict(man)

        except Exception as e:
            print(f"[vae][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[vae] Emitting a stub manifest so the pipeline can proceed.")

        # --- Ensure manifest metadata is consistent with what was actually written ----
        paths = manifest.get("paths") or manifest.get("samples") or []

        # Normalize "samples" -> "paths" so downstream tools are consistent
        if "paths" not in manifest and isinstance(manifest.get("samples"), list):
            manifest["paths"] = manifest["samples"]
            paths = manifest["paths"]

        # Coerce path entries to the expected simple shape (best-effort)
        norm_paths = []
        for it in (paths or []):
            if not isinstance(it, dict):
                continue
            p = it.get("path")
            y = it.get("label")
            try:
                y_int = int(y)
            except Exception:
                continue
            if isinstance(p, Path):
                p = str(p)
            norm_paths.append({"path": p, "label": y_int})
        manifest["paths"] = norm_paths

        # Per-class counts from ACTUAL written paths
        pcc: Dict[str, int] = {}
        for it in (manifest.get("paths") or []):
            try:
                y = str(int(it.get("label")))
            except Exception:
                continue
            pcc[y] = pcc.get(y, 0) + 1

        # Keep keys for all classes (stable schema), but overwrite with real counts
        manifest["per_class_counts"] = {str(k): int(pcc.get(str(k), 0)) for k in range(K)}

        # Totals + budget (derived from manifest truth)
        manifest["num_fake"] = len(manifest.get("paths") or [])
        
        vals = [v for v in manifest["per_class_counts"].values() if isinstance(v, int) and v > 0]
        manifest["budget_per_class"] = min(vals) if vals else None
        # -----------------------------------------------------------------------------

        # Persist manifest (always write something)
        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[vae] Wrote manifest → {man_path}")

        return manifest
