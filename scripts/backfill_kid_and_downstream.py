#!/usr/bin/env python3
"""
scripts/backfill_kid_and_downstream.py

Backfills missing KID + downstream utility metrics into
artifacts/<model>/summaries/summary_*.json so your existing pipeline
(normalize -> jsonl -> collect_scores -> csv) shows real values.

Aligned with gcs-core/gcs_core/val_common.py schema:
- Writes metrics.kid + generative.kid + metrics.kid (flattened legacy)
- Writes metrics.downstream.* (macro_f1, macro_auprc, balanced_acc, precision, recall)
- Writes metrics.gen_precision / metrics.gen_recall (downstream macro precision/recall)
- Writes utility_real_only / utility_real_plus_synth blocks including macro_precision/macro_recall
- Writes metrics_meta stamp so you can tell backfilled vs runtime
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path

import numpy as np
from PIL import Image

# ---- Add repo root + gcs-core to import path ----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gcs-core"))

from common.data import load_dataset_npy  # noqa: E402
from gcs_core import val_common  # noqa: E402


# ---- Config ----
MODELS = [
    "gan",
    "vae",
    "diffusion",
    "autoregressive",
    "maskedautoflow",
    "restrictedboltzmann",
    "gaussianmixture",
]

ART = Path("artifacts")
DATA_DIR = Path("/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware_nhwc")

IMG_SHAPE = (40, 40, 1)
NUM_CLASSES = 9
VAL_FRACTION = 0.1

SEED = 42
# used for downstream + KID inputs
MANIFEST_CAP_PER_CLASS = int(os.getenv("MANIFEST_CAP_PER_CLASS", "200")) 
FID_CAP_PER_CLASS = 200         # passed into compute_all_metrics #was 50 before
EPOCHS = 20                     # it was 5 before and now bump to 20 for paper-grade runs

KID_SUBSET = 200
KID_DEGREE = 3

PHASE1_SUMMARY_NAME  = os.environ.get("PHASE1_SUMMARY_NAME", "paper1.json").strip()
PHASE1_MANIFEST_NAME = os.environ.get("PHASE1_MANIFEST_NAME", "paper1_manifest.json").strip()

# ---------------------------
# Helpers
# ---------------------------

# def _find_summaries(model: str) -> list[Path]:
    # sdir = ART / model / "summaries"
    # if not sdir.exists():
        # return []

    # summary_name = os.environ.get("PHASE1_SUMMARY_NAME", "").strip()

    ## If user pins a single file (paper1.json / latest.json), patch ONLY that
    # if summary_name:
        # p = sdir / summary_name
        # return [p] if p.exists() else []

    ## Otherwise default: patch all historical summaries
    # return sorted(sdir.glob("summary_*.json"))
    
def _find_summaries(model: str) -> list[Path]:
    sdir = ART / model / "summaries"
    if not sdir.exists():
        return []
    locked = sdir / PHASE1_SUMMARY_NAME
    if locked.exists():
        return [locked]              # <-- ONLY patch paper1.json (or whatever name you set)
    return sorted(sdir.glob("summary_*.json"))



# def _manifest_path(model: str) -> Path:
    # manifest_name = os.environ.get("PHASE1_MANIFEST_NAME", "manifest.json").strip()
    # return ART / model / "synthetic" / manifest_name

def _manifest_path(model: str) -> Path:
    # Prefer frozen manifest for paper builds
    p = ART / model / "synthetic" / PHASE1_MANIFEST_NAME
    if p.exists():
        return p
    return ART / model / "synthetic" / "manifest.json"


def _load_manifest_xy(manifest_path: Path, cap_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Loads grayscale PNGs pointed to by manifest.json and returns (x, y_int)."""
    d = json.loads(manifest_path.read_text())
    items = d.get("paths", [])
    if not items:
        raise RuntimeError(f"manifest has no 'paths': {manifest_path}")

    rng = np.random.default_rng(seed)
    byc: dict[int, list[str]] = {}
    for it in items:
        byc.setdefault(int(it["label"]), []).append(it["path"])

    xs, ys = [], []
    for c, plist in sorted(byc.items()):
        rng.shuffle(plist)
        for p in plist[:cap_per_class]:
            im = Image.open(p).convert("L")
            arr = (np.asarray(im, dtype=np.float32) / 255.0)[..., None]  # HWC
            xs.append(arr)
            ys.append(c)

    x = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return x, y


def _poly_mmd2_unbiased(A: np.ndarray, B: np.ndarray, degree: int = 3, gamma: float | None = None, coef0: float = 1.0) -> float:
    """Unbiased polynomial-kernel MMD^2 (KID-style estimator)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m, d = A.shape
    n, _ = B.shape
    if m < 2 or n < 2:
        return float("nan")
    if gamma is None:
        gamma = 1.0 / d

    Kaa = (gamma * (A @ A.T) + coef0) ** degree
    Kbb = (gamma * (B @ B.T) + coef0) ** degree
    Kab = (gamma * (A @ B.T) + coef0) ** degree

    np.fill_diagonal(Kaa, 0.0)
    np.fill_diagonal(Kbb, 0.0)

    return float(Kaa.sum() / (m * (m - 1)) + Kbb.sum() / (n * (n - 1)) - 2.0 * Kab.mean())


def _kid_cpu_fallback(real_01: np.ndarray, fake_01: np.ndarray, subset: int, seed: int) -> float:
    """CPU-only fallback: polynomial MMD^2 on flattened pixel vectors."""
    rng = np.random.default_rng(seed)
    r_idx = rng.choice(len(real_01), size=min(subset, len(real_01)), replace=False)
    f_idx = rng.choice(len(fake_01), size=min(subset, len(fake_01)), replace=False)

    A = real_01[r_idx].reshape(len(r_idx), -1)
    B = fake_01[f_idx].reshape(len(f_idx), -1)
    return _poly_mmd2_unbiased(A, B, degree=KID_DEGREE)


def _kid_inception_if_available(real_01: np.ndarray, fake_01: np.ndarray, subset: int, seed: int) -> float | None:
    """Inception-based KID using the same Inception backbone as FID (if model loads)."""
    try:
        import tensorflow as tf
    except Exception:
        return None

    try:
        inc = val_common._get_inception()
    except Exception:
        inc = None

    if inc is None:
        return None

    rng = np.random.default_rng(seed)
    r_idx = rng.choice(len(real_01), size=min(subset, len(real_01)), replace=False)
    f_idx = rng.choice(len(fake_01), size=min(subset, len(fake_01)), replace=False)

    r = tf.image.resize(real_01[r_idx], (299, 299))
    f = tf.image.resize(fake_01[f_idx], (299, 299))
    r = tf.image.grayscale_to_rgb(r)
    f = tf.image.grayscale_to_rgb(f)

    r = tf.keras.applications.inception_v3.preprocess_input(r * 255.0)
    f = tf.keras.applications.inception_v3.preprocess_input(f * 255.0)

    a = inc.predict(r, verbose=0)
    b = inc.predict(f, verbose=0)

    return _poly_mmd2_unbiased(a, b, degree=KID_DEGREE)


def _set(d: dict, path: list[str], value):
    """Set nested dict value, creating dicts along the way."""
    cur = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def _get_macro_pr(util_block: dict) -> tuple[float | None, float | None]:
    """
    Robustly retrieve macro precision/recall from util dict, supporting both layouts:
    - preferred: util_block["macro_precision"], util_block["macro_recall"]
    - fallback: util_block["per_class"]["macro_precision"], ["macro_recall"]
    """
    if not isinstance(util_block, dict):
        return None, None

    mp = util_block.get("macro_precision")
    mr = util_block.get("macro_recall")

    if (mp is None) or (mr is None):
        pc = util_block.get("per_class")
        if isinstance(pc, dict):
            mp = mp if mp is not None else pc.get("macro_precision")
            mr = mr if mr is not None else pc.get("macro_recall")

    return mp, mr


# ---------------------------
# Main
# ---------------------------

def main():
    xtr, ytr, xv, yv, xt, yt = load_dataset_npy(
        DATA_DIR,
        img_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        val_fraction=VAL_FRACTION,
    )

    print(f"[INFO] loaded real: train={xtr.shape}, val={xv.shape}, test={xt.shape}")

    for model in MODELS:
        try:
            summaries = _find_summaries(model)
            print(f"[INFO] {model}: patching summaries {[p.name for p in summaries]}")
    
            if not summaries:
                print(f"[SKIP] {model}: no summaries")
                continue
    
            manifest = _manifest_path(model)
            print(f"[INFO] {model}: using manifest {manifest.name}")
            
            if not manifest.exists():
                print(f"[SKIP] {model}: no manifest at {manifest}")
                continue
    
            x_syn, y_syn = _load_manifest_xy(manifest, cap_per_class=MANIFEST_CAP_PER_CLASS, seed=SEED)
            print(f"[INFO] {model}: loaded synth {x_syn.shape} from {manifest}")
    
            # Compute downstream utility via gcs_core
            out = val_common.compute_all_metrics(
                img_shape=IMG_SHAPE,
                x_train_real=xtr, y_train_real=ytr,
                x_val_real=xv,   y_val_real=yv,
                x_test_real=xt,  y_test_real=yt,
                x_synth=x_syn,   y_synth=y_syn,
                fid_cap_per_class=FID_CAP_PER_CLASS,
                seed=SEED,
                epochs=EPOCHS,
            )
    
            util_R = out.get("real_only")
            util_RS = out.get("real_plus_synth")
            if not isinstance(util_R, dict) or not isinstance(util_RS, dict):
                raise RuntimeError(f"{model}: invalid util blocks from compute_all_metrics")
    
            mp_R, mr_R = _get_macro_pr(util_R)
            mp_RS, mr_RS = _get_macro_pr(util_RS)
    
            # KID inputs (val real vs synth)
            real_01 = val_common.to_01_hwc(xv, IMG_SHAPE)
            fake_01 = val_common.to_01_hwc(x_syn, IMG_SHAPE)
    
            kid = _kid_inception_if_available(real_01, fake_01, subset=KID_SUBSET, seed=SEED)
            if kid is None:
                kid = _kid_cpu_fallback(real_01, fake_01, subset=KID_SUBSET, seed=SEED)
                kid_mode = "cpu_fallback_poly_mmd_pixels_v1"
            else:
                kid_mode = "inception_poly_mmd_v1"
    
            # Patch ALL summaries for this model
            for sp in summaries:
                s = json.loads(sp.read_text())
    
                # ---- meta stamp: lets you detect backfill vs runtime ----
                s.setdefault("metrics_meta", {})
                s["metrics_meta"].update({
                    "computed_at": "backfill",
                    "metrics_version": "phase1_backfill_v2",
                    "kid_mode": kid_mode,
                    "gen_pr_mode": "downstream_macro",
                })
    
                # ---- utility blocks (human-readable) ----
                s["utility_real_only"] = {
                    "macro_f1": util_R.get("macro_f1"),
                    "macro_auprc": util_R.get("macro_auprc"),
                    "bal_acc": util_R.get("bal_acc"),
                    "balanced_acc": util_R.get("bal_acc"),
                    "macro_precision": mp_R,
                    "macro_recall": mr_R,
                }
                s["utility_real_plus_synth"] = {
                    "macro_f1": util_RS.get("macro_f1"),
                    "macro_auprc": util_RS.get("macro_auprc"),
                    "bal_acc": util_RS.get("bal_acc"),
                    "balanced_acc": util_RS.get("bal_acc"),
                    "macro_precision": mp_RS,
                    "macro_recall": mr_RS,
                }
    
                # ---- structured downstream block (what your collector can read) ----
                _set(s, ["metrics", "downstream", "macro_f1"], util_RS.get("macro_f1"))
                _set(s, ["metrics", "downstream", "macro_auprc"], util_RS.get("macro_auprc"))
                _set(s, ["metrics", "downstream", "balanced_acc"], util_RS.get("bal_acc"))
                s["metrics.downstream.balanced_acc"] = util_RS.get("bal_acc")
                _set(s, ["metrics", "downstream", "precision"], mp_RS)
                _set(s, ["metrics", "downstream", "recall"], mr_RS)
    
                # ---- gen_precision / gen_recall (your “gen_*” columns) ----
                _set(s, ["metrics", "gen_precision"], mp_RS)
                _set(s, ["metrics", "gen_recall"], mr_RS)
    
                # ---- legacy flattened keys (keeps older scripts working) ----
                s["metrics.kid"] = float(kid)
                s["metrics.gen_precision"] = mp_RS
                s["metrics.gen_recall"] = mr_RS
                s["metrics.downstream.macro_f1"] = util_RS.get("macro_f1")
                s["metrics.downstream.precision"] = mp_RS
                s["metrics.downstream.recall"] = mr_RS
    
                # ---- KID fields (structured + legacy) ----
                _set(s, ["metrics", "kid"], float(kid))
                _set(s, ["generative", "kid"], float(kid))
    
                sp.write_text(json.dumps(s, indent=2, sort_keys=True))
    
            # Pretty print
            mp_show = float(mp_RS) if mp_RS is not None else float("nan")
            mr_show = float(mr_RS) if mr_RS is not None else float("nan")
    
            print(
                f"[OK] {model}: patched {len(summaries)} summaries "
                f"(kid={kid:.6g} | macro_f1={util_RS.get('macro_f1'):.4f} | bal_acc={util_RS.get('bal_acc'):.4f} "
                f"| macro_P={mp_show:.4f} | macro_R={mr_show:.4f})"
            )
            
        except Exception as e:
            print(f"[FAIL] {model}: {e}")
            continue


if __name__ == "__main__":
    main()
