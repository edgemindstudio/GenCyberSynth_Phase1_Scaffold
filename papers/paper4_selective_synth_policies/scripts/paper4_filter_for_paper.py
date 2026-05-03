#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEEP_SEEDS = {"42", "43", "44"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    with args.in_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    if not rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=r.fieldnames or [])
            w.writeheader()
        print(f"[ok] wrote {args.out_csv} (0 rows)")
        return

    keep = []
    for row in rows:
        paper_id = (row.get("paper_id") or "").strip()
        cfg = (row.get("config_id") or "").strip()
        seed = (row.get("seed") or "").strip()

        # Keep paper4 only (defensive)
        if paper_id and paper_id != "paper4":
            continue

        # Drop smoke
        if "smoke" in cfg:
            continue

        # Keep only canonical seeds
        if seed and seed not in KEEP_SEEDS:
            continue

        keep.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in keep:
            w.writerow(row)

    print(f"[ok] wrote {args.out_csv} ({len(keep)} rows)")


if __name__ == "__main__":
    main()