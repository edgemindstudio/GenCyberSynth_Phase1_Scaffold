#!/usr/bin/env bash
set -euo pipefail
export CFG=configs/paper1_config.yaml
export GCS_WORKDIR=/home/bruno.fonkeng/gencys
export GCS_ARTIFACTS=/home/bruno.fonkeng/gencys/artifacts
export GCS_DATA=/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware_nhwc

python scripts/backfill_kid_and_downstream.py
python scripts/normalize_summaries.py
bash scripts/build_jsonl.sh
python scripts/jsonl_to_csv.py
python scripts/phase1_report.py
