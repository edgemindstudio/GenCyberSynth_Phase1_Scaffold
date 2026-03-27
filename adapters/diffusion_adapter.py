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
   - Writes a JSON manifest to: man_path = synth_root / "manifest.json"

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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import Adapter


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """Fetch a nested config value by dotted path, e.g. "paths.artifacts"."""
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
    return np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)


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


# ---------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------
class DiffusionAdapter(Adapter):
    """Adapter that calls the local diffusion sampler to emit a manifest."""
    name = "diffusion"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts"))
        model_root = artifacts_root / "diffusion"

        SEED = int(config.get("SEED", 42))

        base_synth_root = Path(
            _cfg_get(
                config,
                "ARTIFACTS.diffusion_synthetic",
                model_root / "synthetic",
            )
        )
        synth_root = _ensure_dir(
            base_synth_root if base_synth_root.name.startswith("seed") else base_synth_root / f"seed{SEED}"
        )

        # Basic knobs (with robust fallbacks)
        H, W, C = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        K = int(_cfg_get(config, "NUM_CLASSES", 9))
        S = int(_cfg_get(config, "SAMPLES_PER_CLASS", 25))

        # Diffusion hyperparams (sampling-side)
        T = int(_cfg_get(config, "diffusion.steps", 200))
        base_filters = int(_cfg_get(config, "diffusion.base_filters", 64))
        depth = int(_cfg_get(config, "diffusion.depth", 2))
        time_emb_dim = int(_cfg_get(config, "diffusion.time_emb_dim", 128))
        lr = float(_cfg_get(config, "diffusion.lr", 2e-4))
        beta_1 = float(_cfg_get(config, "diffusion.beta_1", 0.9))

        # Checkpoints
        default_ckpt_dir = artifacts_root / "diffusion" / "checkpoints"
        base_ckpt_dir = Path(_cfg_get(config, "ARTIFACTS.diffusion_checkpoints", default_ckpt_dir))
        ckpt_dir = base_ckpt_dir if base_ckpt_dir.name.startswith("seed") else base_ckpt_dir / f"seed{SEED}"

        print(f"[diffusion] ckpt_dir={ckpt_dir}")

        candidates = [
            ckpt_dir / "DDPM_best.weights.h5",
            ckpt_dir / "DDPM_last.weights.h5",
            ckpt_dir / "DIFF_best.weights.h5",
            ckpt_dir / "DIFF_last.weights.h5",
            ckpt_dir / "diffusion_best.h5",  # legacy
            ckpt_dir / "diffusion_last.h5",  # legacy
        ]

        weights_path = next((p for p in candidates if p.exists()), None)

        dataset = _cfg_get(config, "data.root", config.get("DATA_DIR", "USTC-TFC2016_40x40_gray"))

        # Stub manifest (will be replaced on success)
        manifest: Dict[str, Any] = {
            "dataset": dataset,
            "seed": SEED,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "per_class_counts": {str(k): 0 for k in range(K)},
            "paths": [],
        }

        try:
            import tensorflow as tf
            from diffusion.models import build_diffusion_model  # type: ignore
            from diffusion.sample import sample_batch  # type: ignore

            model = build_diffusion_model(
                img_shape=(H, W, C),
                num_classes=K,
                base_filters=base_filters,
                depth=depth,
                time_emb_dim=time_emb_dim,
                learning_rate=lr,
                beta_1=beta_1,
            )

            # Keras 3: create variables before load_weights
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
                    print(
                        f"[diffusion][warn] Failed to load {weights_path.name}: {e}\n"
                        f"→ continuing with randomly initialized weights."
                    )
            else:
                print(f"[diffusion][warn] no DDPM checkpoint in {ckpt_dir}; using random weights.")

            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            for k in range(K):
                class_ids = np.full((S,), k, dtype=np.int32)
                imgs01, _ = sample_batch(
                    model,
                    num_samples=S,
                    num_classes=K,
                    img_shape=(H, W, C),
                    T=T,
                    alpha_hat=None,
                    class_ids=class_ids,
                    seed=SEED + k,
                )

                cls_dir = synth_root / str(k) / str(SEED)
                _ensure_dir(cls_dir)

                for j in range(S):
                    out_path = cls_dir / f"diff_{j:05d}.png"
                    _save_png(imgs01[j], out_path)
                    manifest["paths"].append({"path": str(out_path.resolve()), "label": int(k)})

                manifest["per_class_counts"][str(k)] = int(S)

        except Exception as e:
            print(f"[diffusion][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[diffusion] Emitting a stub manifest so the pipeline can proceed.")

        # Normalize + derived fields (ALWAYS)
        manifest = _normalize_manifest(manifest, num_classes=K)
        manifest.setdefault("dataset", dataset)
        manifest.setdefault("seed", SEED)
        manifest.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

        # Persist manifest
        man_path = synth_root / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[diffusion] Wrote manifest → {man_path}")

        return manifest