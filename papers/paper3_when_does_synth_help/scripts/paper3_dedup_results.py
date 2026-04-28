#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
from datetime import datetime

def parse_ts_from_summary_file(p: str) -> datetime:
    # expects .../summary_YYYYMMDD_HHMMSS.json
    name = Path(p).name
    stem = name.replace("summary_", "").replace(".json", "")
    return datetime.strptime(stem, "%Y%m%d_%H%M%S")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    with args.in_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # primary key for Paper 3 analysis
    def key(row):
        return (row.get("dataset_id"), row.get("model"), row.get("config_id"), row.get("seed"))

    best = {}
    for row in rows:
        k = key(row)
        ts = parse_ts_from_summary_file(row["summary_file"])
        if (k not in best) or (ts > best[k][0]):
            best[k] = (ts, row)

    dedup = [v[1] for v in sorted(best.values(), key=lambda x: x[0])]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dedup[0].keys()))
        w.writeheader()
        for row in dedup:
            w.writerow(row)

    print(f"[ok] wrote {args.out_csv} ({len(dedup)} rows)")

if __name__ == "__main__":
    main()