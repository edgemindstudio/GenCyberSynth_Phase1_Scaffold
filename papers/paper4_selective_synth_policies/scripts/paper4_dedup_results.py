#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


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

    # Dedup key: (model, config_id, seed, manifest_path)
    best: Dict[Tuple[str, str, str, str], Dict] = {}
    for row in rows:
        key = (
            (row.get("model") or "").strip(),
            (row.get("config_id") or "").strip(),
            (row.get("seed") or "").strip(),
            (row.get("manifest_path") or "").strip(),
        )
        cur = best.get(key)
        if cur is None:
            best[key] = row
            continue
        # keep most recently modified summary_file
        if _mtime(row.get("summary_file", "")) >= _mtime(cur.get("summary_file", "")):
            best[key] = row

    deduped: List[Dict] = list(best.values())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in deduped:
            w.writerow(row)

    print(f"[ok] wrote {args.out_csv} ({len(deduped)} rows)")


if __name__ == "__main__":
    main()