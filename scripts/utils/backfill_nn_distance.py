# scripts/utils/backfill_nn_distance.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill NN distance stats (synth→nearest-real) into JSONL with many aliases so plotters can find them.

Writes for each line where obj["model"] == <model>:
  diversity.nn_distance = {"mean","min","p50","p90"}           # nested object
Plus flat aliases:
  diversity.nn_distance.mean, diversity.nn_distance_min, ...
  diversity.nn_distance_mean, diversity.nn_distance_min, ...
  metrics.nn_distance.mean, metrics.nn_distance_min, ...
  metrics.nn_distance_mean, metrics.nn_distance_min, ...
  nn_distance.mean, nn_distance_min, ...
  nn_distance_mean, nn_distance_min, ...

Usage:
python scripts/utils/backfill_nn_distance.py \
  --in artifacts/summaries/phase1_summaries.counts.jsonl \
  --out artifacts/summaries/phase1_summaries.nn.jsonl
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

def load_embeddings(root: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    re = root / "real_embeddings.npy"
    se = root / "synth_embeddings.npy"
    if not (re.exists() and se.exists()):
        return None
    real = np.load(re).astype(np.float32)   # (Nr, D)
    synth = np.load(se).astype(np.float32)  # (Ns, D)
    if real.ndim != 2 or synth.ndim != 2 or real.shape[1] != synth.shape[1]:
        raise ValueError(f"Bad shapes: real={real.shape} synth={synth.shape} under {root}")
    return real, synth

def pairwise_sqdist(A: np.ndarray, B: np.ndarray, chunk: int = 2048) -> np.ndarray:
    mins = []
    for i in range(0, A.shape[0], chunk):
        a = A[i:i+chunk]
        aa = np.sum(a*a, axis=1, keepdims=True)
        bb = np.sum(B*B, axis=1, keepdims=True).T
        d2 = aa + bb - 2.0 * (a @ B.T)
        d2 = np.maximum(d2, 0.0)
        mins.append(np.sqrt(np.min(d2, axis=1)))
    return np.concatenate(mins, axis=0)

def summarize(dist: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(dist)),
        "min":  float(np.min(dist)),
        "p50":  float(np.percentile(dist, 50.0)),
        "p90":  float(np.percentile(dist, 90.0)),
    }

def discover_models(artifacts_root: Path) -> Dict[str, Path]:
    out = {}
    if not artifacts_root.exists():
        return out
    for model_dir in artifacts_root.iterdir():
        feats = model_dir / "features"
        if feats.is_dir():
            out[model_dir.name] = feats
    return out

def write_aliases(obj: dict, stats: Dict[str, float]) -> None:
    # nested object
    div = obj.setdefault("diversity", {})
    div["nn_distance"] = stats
    # flat aliases across common namespaces
    for k, v in stats.items():
        # diversity.*
        obj[f"diversity.nn_distance.{k}"] = v
        obj[f"diversity.nn_distance_{k}"] = v
        # metrics.*
        obj[f"metrics.nn_distance.{k}"] = v
        obj[f"metrics.nn_distance_{k}"] = v
        # top-level simple
        obj[f"nn_distance.{k}"] = v
        obj[f"nn_distance_{k}"] = v

def main():
    ap = argparse.ArgumentParser(description="Backfill NN distance stats into JSONL (write many alias fields).")
    ap.add_argument("--in",  dest="inp",  required=True, help="Input JSONL (e.g., artifacts/summaries/phase1_summaries.counts.jsonl)")
    ap.add_argument("--out", dest="out",  required=True, help="Output JSONL path")
    ap.add_argument("--artifacts-root", default="artifacts", help="Root containing <model>/features/")
    args = ap.parse_args()

    models = discover_models(Path(args.artifacts_root))
    stats_by_model: Dict[str, Dict[str, float]] = {}
    for model, feats in models.items():
        maybe = load_embeddings(feats)
        if maybe is None:
            continue
        real, synth = maybe
        if len(synth) == 0 or len(real) == 0:
            continue
        d = pairwise_sqdist(synth, real)
        stats_by_model[model] = summarize(d)
        print(f"[ok] {model}: NN distance stats {stats_by_model[model]}")

    wrote = 0
    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            model = str(obj.get("model", ""))
            if model in stats_by_model:
                write_aliases(obj, stats_by_model[model])
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[ok] wrote {wrote} line(s) → {args.out}")
    missing = [m for m in models if m not in stats_by_model]
    if missing:
        print(f"[warn] No stats for: {', '.join(missing)} (check embeddings exist).")

if __name__ == "__main__":
    main()
