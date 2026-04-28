#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    with args.in_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    # Drop smoke + anything missing regime fields (imbalance/budget/method)
    keep = []
    for row in rows:
        cfg = (row.get("config_id") or "").strip()
        if "smoke" in cfg:
            continue
        if (row.get("imbalance") or "").strip() == "" and (row.get("synth_budget_per_class") or "").strip() == "":
            continue
        keep.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
        w.writeheader()
        for row in keep:
            w.writerow(row)

    print(f"[ok] wrote {args.out_csv} ({len(keep)} rows)")

if __name__ == "__main__":
    main()