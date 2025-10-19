# scripts/jsonl_to_csv.py

#!/usr/bin/env python3
"""
Convert consolidated JSONL -> tiny CSV table for README/report previews.

Usage:
  python scripts/jsonl_to_csv.py \
      [--src artifacts/summaries/phase1_summaries.jsonl] \
      [--dst artifacts/phase1_scores.csv]
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="artifacts/summaries/phase1_summaries.jsonl")
    ap.add_argument("--dst", default="artifacts/phase1_scores.csv")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise SystemExit(f"[ERROR] Missing JSONL: {src}. Run: make summaries-jsonl")

    rows = []
    for i, line in enumerate(src.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"[ERROR] Bad JSON on line {i}: {e}") from e

        metrics = d.get("metrics") or {}
        counts = d.get("counts") or {}

        rows.append({
            "run_id": d.get("run_id"),
            "model": d.get("model"),
            "num_fake": counts.get("num_fake"),
            "ms_ssim": metrics.get("ms_ssim"),
            "cfid": metrics.get("cfid"),
        })

    if not rows:
        raise SystemExit("[ERROR] No rows parsed from JSONL; is it empty?")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {dst} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
