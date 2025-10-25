# scripts/utils/backfill_counts.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill per-class counts into a consolidated JSONL so the imbalance plots work.
- Accepts a REAL images root organized as <real-root>/<class_id>/**.{png,jpg,...}
- Optionally accepts a synth manifest: {"images":[{"path": "...", "class": <int>}, ...]}
- Writes aliases the plots already read:
    counts.real_per_class
    counts.synth_per_class
    counts.real_plus_synth_per_class
Usage:
python scripts/utils/backfill_counts.py \
  --in artifacts/summaries/phase1_summaries.jsonl \
  --out artifacts/summaries/phase1_summaries.counts.jsonl \
  --real-root USTC-TFC2016_malware/real \
  --synth-manifest artifacts/gan/synth_manifest.json \
  --num-classes 9
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from glob import glob
from collections import Counter, defaultdict
from typing import Dict

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def count_real_per_class(real_root: Path) -> Dict[int, int]:
    if not real_root or not real_root.exists():
        return {}
    counts: Dict[int, int] = {}
    for sub in sorted(p for p in real_root.iterdir() if p.is_dir()):
        if not re.fullmatch(r"\d+", sub.name):
            continue
        n = 0
        for ext in IMG_EXTS:
            n += len(glob(str(sub / f"**/*{ext}"), recursive=True))
        counts[int(sub.name)] = n
    return counts

def count_synth_from_manifest(manifest_path: Path) -> Dict[int, int]:
    if not manifest_path or not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    ctr = Counter()
    for item in images:
        cls = item.get("class", item.get("label", item.get("y")))
        try:
            ctr[int(cls)] += 1
        except Exception:
            # try reading parent dir name as class id
            p = Path(item.get("path", ""))
            try:
                ctr[int(p.parent.name)] += 1
            except Exception:
                pass
    return dict(ctr)

def merge_sum(a: Dict[int, int], b: Dict[int, int]) -> Dict[int, int]:
    out = defaultdict(int)
    for k, v in a.items(): out[int(k)] += int(v)
    for k, v in b.items(): out[int(k)] += int(v)
    return dict(out)

def main():
    ap = argparse.ArgumentParser(description="Backfill per-class counts into summaries JSONL.")
    ap.add_argument("--in",  dest="inp", required=True, help="Input JSONL (e.g., artifacts/summaries/phase1_summaries.jsonl)")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    ap.add_argument("--real-root", type=str, default="", help="Real images root (<root>/<class_id>/**)")
    ap.add_argument("--synth-manifest", type=str, default="", help="Synth manifest JSON with images[].class")
    ap.add_argument("--num-classes", type=int, default=None, help="Optional clamp to [0..C-1]")
    args = ap.parse_args()

    real_counts  = count_real_per_class(Path(args.real_root)) if args.real_root else {}
    synth_counts = count_synth_from_manifest(Path(args.synth_manifest)) if args.synth_manifest else {}
    real_plus    = merge_sum(real_counts, synth_counts)

    if args.num_classes is not None:
        def clamp(d):
            return {int(k): int(v) for k, v in d.items() if 0 <= int(k) < args.num_classes}
        real_counts  = clamp(real_counts)
        synth_counts = clamp(synth_counts)
        real_plus    = clamp(real_plus)

    wrote = 0
    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            obj = json.loads(line)
            counts = obj.setdefault("counts", {})
            if "real_per_class" not in counts and real_counts:
                counts["real_per_class"] = real_counts
            if "synth_per_class" not in counts and synth_counts:
                counts["synth_per_class"] = synth_counts
            if "real_plus_synth_per_class" not in counts and (real_plus or (real_counts and synth_counts)):
                counts["real_plus_synth_per_class"] = real_plus
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[ok] wrote {wrote} line(s) → {args.out}")
    if not real_counts and not synth_counts:
        print("[warn] No counts derived. Provide --real-root and/or --synth-manifest.")

if __name__ == "__main__":
    main()
