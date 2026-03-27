# adapters/restrictedboltzmann_adapter.py

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import Adapter


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
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
    if not isinstance(manifest, dict):
        manifest = {}

    # Normalize "samples" -> "paths"
    if "paths" not in manifest and isinstance(manifest.get("samples"), list):
        manifest["paths"] = manifest["samples"]

    # Normalize paths entries
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
            kk = str(int(it["label"]))
            pcc[kk] = pcc.get(kk, 0) + 1

    # Stabilize keys for all classes
    manifest["per_class_counts"] = {str(k): int(pcc.get(str(k), 0)) for k in range(num_classes)}

    # Derived fields
    manifest["num_fake"] = int(len(manifest["paths"])) if isinstance(manifest["paths"], list) else 0

    vals = [int(v) for v in manifest["per_class_counts"].values() if v is not None]
    manifest["budget_per_class"] = (min(vals) if vals else None)

    return manifest


class RBMAdapter(Adapter):
    name = "restrictedboltzmann"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "restrictedboltzmann"
        synth_root = _ensure_dir(model_root / "synthetic")

        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "restrictedboltzmann"

        seed = int(config["SEED"]) if "SEED" in config else int(_cfg_get(config, "random_seeds", [42])[0])

        base_synth_root = Path(
            _cfg_get(
                config,
                "ARTIFACTS.restrictedboltzmann_synthetic",
                model_root / "synthetic",
            )
        )

        synth_root = _ensure_dir(
            base_synth_root if base_synth_root.name.startswith("seed")
            else base_synth_root / f"seed{seed}"
        )

        H, W, C = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        K = int(_cfg_get(config, "NUM_CLASSES", 9))

        dataset = _cfg_get(config, "data.root", config.get("DATA_DIR", "USTC-TFC2016_40x40_gray"))

        manifest: Dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "per_class_counts": {str(k): 0 for k in range(K)},
            "paths": [],
        }

        try:
            from restrictedboltzmann.sample import synth as rbm_synth  # type: ignore

            np.random.seed(seed)
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
            except Exception:
                pass

            print(f"[restrictedboltzmann] HWC={H,W,C}  K={K}  seed={seed}")
            man = rbm_synth(config, str(synth_root), seed=seed)
            manifest = dict(man)

        except Exception as e:
            print(f"[restrictedboltzmann][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[restrictedboltzmann] Emitting a stub manifest so the pipeline can proceed.")

        # Normalize ONCE (right before writing)
        manifest = _normalize_manifest(manifest, num_classes=K)
        manifest.setdefault("dataset", dataset)
        manifest.setdefault("seed", seed)
        manifest.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[restrictedboltzmann] Wrote manifest → {man_path}")

        return manifest
