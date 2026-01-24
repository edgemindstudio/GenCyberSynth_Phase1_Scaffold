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


def _normalize_manifest(manifest: Dict[str, Any], *, num_classes: int) -> Dict[str, Any]:
    """
    Normalize schema + add stable derived fields:
      - Normalize "samples" -> "paths"
      - Ensure paths is a list[{"path": str, "label": int}]
      - Ensure per_class_counts has keys "0"..."{K-1}"
      - Derive:
          num_fake         = len(paths) if paths else sum(per_class_counts)
          budget_per_class = min(per_class_counts) if available else None
    """
    if not isinstance(manifest, dict):
        manifest = {}

    # Normalize "samples" -> "paths"
    if "paths" not in manifest and isinstance(manifest.get("samples"), list):
        manifest["paths"] = manifest["samples"]

    # Ensure paths exists and normalize entries
    raw_paths = manifest.get("paths")
    if not isinstance(raw_paths, list):
        raw_paths = []

    norm_paths = []
    for it in raw_paths:
        if not isinstance(it, dict):
            continue
        p = it.get("path")
        y = it.get("label")
        if isinstance(p, Path):
            p = str(p)
        if not isinstance(p, str) or not p:
            continue
        try:
            y_int = int(y)
        except Exception:
            continue
        norm_paths.append({"path": p, "label": y_int})
    manifest["paths"] = norm_paths

    # per_class_counts: prefer existing if valid, else derive from paths
    pcc_in = manifest.get("per_class_counts")
    pcc: Dict[str, int] = {}

    if isinstance(pcc_in, dict) and len(pcc_in) > 0:
        for k, v in pcc_in.items():
            try:
                kk = str(int(k))
                vv = int(v)
            except Exception:
                continue
            if 0 <= int(kk) < num_classes and vv >= 0:
                pcc[kk] = vv
    else:
        for it in manifest["paths"]:
            try:
                kk = str(int(it["label"]))
            except Exception:
                continue
            pcc[kk] = pcc.get(kk, 0) + 1

    # Stabilize keys for all classes
    manifest["per_class_counts"] = {str(k): int(pcc.get(str(k), 0)) for k in range(num_classes)}

    # Derived: num_fake
    if len(manifest["paths"]) > 0:
        manifest["num_fake"] = int(len(manifest["paths"]))
    else:
        manifest["num_fake"] = int(sum(manifest["per_class_counts"].values()))

    # Derived: budget_per_class
    vals = [int(v) for v in manifest["per_class_counts"].values() if v is not None]
    manifest["budget_per_class"] = (min(vals) if vals and min(vals) > 0 else (min(vals) if vals else None))

    return manifest


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

        # Basic knobs with robust fallbacks
        H, W, C = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        K = int(_cfg_get(config, "NUM_CLASSES", 9))

        # Seed: prefer SEED, else first from random_seeds, else 42
        if "SEED" in config:
            seed = int(config["SEED"])
        else:
            seed = int(_cfg_get(config, "random_seeds", [42])[0])

        dataset = _cfg_get(config, "data.root", config.get("DATA_DIR", "USTC-TFC2016_40x40_gray"))

        # Default manifest scaffold (stub)
        manifest: Dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "per_class_counts": {str(k): 0 for k in range(K)},
            "paths": [],  # {"path": "...", "label": int}
        }

        try:
            from autoregressive.sample import synth as ar_synth  # type: ignore

            # Deterministic sampling
            np.random.seed(seed)
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
            except Exception:
                pass

            print(f"[autoregressive] HWC={H,W,C}  K={K}  seed={seed}")
            man = ar_synth(config, str(synth_root), seed=seed)
            manifest = dict(man) if isinstance(man, dict) else dict(manifest)

        except Exception as e:
            print(f"[autoregressive][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[autoregressive] Emitting a stub manifest so the pipeline can proceed.")

        # Normalize + add stable derived fields (ALWAYS)
        manifest = _normalize_manifest(manifest, num_classes=K)
        manifest.setdefault("dataset", dataset)
        manifest.setdefault("seed", seed)
        manifest.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

        # Write manifest to disk (always)
        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[autoregressive] Wrote manifest → {man_path}")

        return manifest
