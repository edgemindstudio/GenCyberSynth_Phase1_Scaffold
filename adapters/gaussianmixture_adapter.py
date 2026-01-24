# adapters/gaussianmixture_adapter.py

"""
GMMAdapter
----------
Adapter that invokes the local *gaussianmixture* sampler to generate
class-conditional samples and writes a manifest the evaluator can consume.

Happy path
----------
- Imports `gaussianmixture.sample.synth`.
- Resolves output dir: {artifacts}/gaussianmixture/synthetic
- Calls: synth(cfg, output_root, seed) → manifest dict
- Writes manifest to: {artifacts}/gaussianmixture/synthetic/manifest.json

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
    cur: Any = cfg
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
      - Ensure paths is list[{"path": str, "label": int}]
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

    # Ensure paths exists + normalize entries
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

    # Derived: budget_per_class (conservative)
    vals = [int(v) for v in manifest["per_class_counts"].values() if v is not None]
    manifest["budget_per_class"] = (min(vals) if vals and min(vals) > 0 else (min(vals) if vals else None))

    return manifest


# ------------------------------
# Adapter
# ------------------------------
class GMMAdapter(Adapter):
    """Adapter that calls the local Gaussian Mixture sampler to emit a manifest."""
    name = "gaussianmixture"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "gaussianmixture"
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
            # Local import so adapter remains importable even if package not vendored yet
            from gaussianmixture.sample import synth as gmm_synth  # type: ignore

            # Deterministic runs
            np.random.seed(seed)
            try:
                import tensorflow as tf  # if available, sync TF RNG as well
                tf.random.set_seed(seed)
            except Exception:
                pass

            print(f"[gaussianmixture] HWC={H,W,C}  K={K}  seed={seed}")
            man = gmm_synth(config, str(synth_root), seed=seed)

            # Normalize to plain dict & use as manifest
            manifest = dict(man)

        except Exception as e:
            print(f"[gaussianmixture][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[gaussianmixture] Emitting a stub manifest so the pipeline can proceed.")

        # Normalize + derived fields (ALWAYS)
        manifest = _normalize_manifest(manifest, num_classes=K)
        manifest.setdefault("dataset", dataset)
        manifest.setdefault("seed", seed)
        manifest.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

        # Persist manifest (always write something)
        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[gaussianmixture] Wrote manifest → {man_path}")

        return manifest
