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


def _normalize_manifest(manifest: Dict[str, Any], *, num_classes: int) -> Dict[str, Any]:
    """
    Normalize schema + add stable derived fields:
      - Normalize "samples" -> "paths"
      - Ensure paths is a list[{"path": str, "label": int}]
      - Ensure per_class_counts has keys "0"..."{K-1}"
      - Derive:
          num_fake        = len(paths) if paths else sum(per_class_counts)
          budget_per_class = min(per_class_counts) if available else None
    """
    if not isinstance(manifest, dict):
        manifest = {}

    # Normalize "samples" -> "paths"
    if "paths" not in manifest and isinstance(manifest.get("samples"), list):
        manifest["paths"] = manifest["samples"]

    # Ensure paths exists
    raw_paths = manifest.get("paths")
    if not isinstance(raw_paths, list):
        raw_paths = []
    # Normalize each entry
    norm_paths = []
    for it in raw_paths:
        if not isinstance(it, dict):
            continue
        p = it.get("path")
        y = it.get("label")
        # path
        if isinstance(p, Path):
            p = str(p)
        if not isinstance(p, str) or not p:
            continue
        # label
        try:
            y_int = int(y)
        except Exception:
            continue
        norm_paths.append({"path": p, "label": y_int})
    manifest["paths"] = norm_paths

    # per_class_counts: prefer existing if it looks valid, otherwise derive from paths
    pcc_in = manifest.get("per_class_counts")
    pcc: Dict[str, int] = {}

    if isinstance(pcc_in, dict) and len(pcc_in) > 0:
        # Cast values to int best-effort, keep only known classes
        for k, v in pcc_in.items():
            try:
                kk = str(int(k))
                vv = int(v)
            except Exception:
                continue
            if 0 <= int(kk) < num_classes and vv >= 0:
                pcc[kk] = vv
    else:
        # Derive from paths
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
        # If sampler didn’t record paths, fall back to per_class_counts totals
        manifest["num_fake"] = int(sum(manifest["per_class_counts"].values()))

    # Derived: budget_per_class

    # For class-restricted synthesis, non-target classes may correctly have count 0.

    # Infer the requested budget from positive class counts when available.

    vals = [int(v) for v in manifest["per_class_counts"].values() if v is not None]

    positive_vals = [v for v in vals if v > 0]

    if positive_vals:

        manifest["budget_per_class"] = int(min(positive_vals))

    else:

        manifest["budget_per_class"] = 0 if vals else None
    return manifest


# ------------------------------
# Adapter
# ------------------------------
class GANAdapter(Adapter):
    """Adapter that calls the local GAN sampler to emit a manifest."""
    name = "gan"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "gan"

        # Seed: prefer SEED, else first from random_seeds, else 42
        if "SEED" in config:
            seed = int(config["SEED"])
        else:
            seed = int(_cfg_get(config, "random_seeds", [42])[0])

        # Keep synthetic root UN-scoped; gan.sample.synth() will add <config_id>/seed<seed>
        base_synth_root = Path(
            _cfg_get(
                config,
                "ARTIFACTS.gan_synthetic",
                model_root / "synthetic",
            )
        )
        base_synth_root = _ensure_dir(base_synth_root)

        # Minimal knobs (mainly for stub manifest)
        H, W, C = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        K = int(_cfg_get(config, "NUM_CLASSES", 9))

        dataset = _cfg_get(config, "data.root", config.get("DATA_DIR", "USTC-TFC2016_40x40_gray"))

        # Default (stub) manifest structure; replaced on success
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
            # man = gan_synth(config, str(synth_root), seed=seed)
            man = gan_synth(config, str(base_synth_root), seed=seed)

            # Normalize to plain dict & use it as our manifest
            manifest = dict(man) if isinstance(man, dict) else dict(manifest)

        except Exception as e:
            # Fallback: emit stub manifest and warn clearly
            print(f"[gan][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[gan] Emitting a stub manifest so the pipeline can proceed.")

        # Normalize + add stable derived fields (ALWAYS)
        manifest = _normalize_manifest(manifest, num_classes=K)
        # Ensure minimal required fields still exist
        manifest.setdefault("dataset", dataset)
        manifest.setdefault("seed", seed)
        manifest.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

        # Canonical per-config manifest path
        rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}
        cfg_id = rm.get("config_id") or "default"
        out_dir = _ensure_dir(base_synth_root / cfg_id / f"seed{seed}")

        man_path = out_dir / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[gan] Wrote manifest → {man_path}")

        # Optional convenience pointer for eval/back-compat: "latest" manifest
        try:
            latest_path = base_synth_root / "manifest.json"
            with open(latest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"[gan] Also wrote latest manifest → {latest_path}")
        except Exception:
            pass

        return manifest
