#!/usr/bin/env bash
set -euo pipefail
python app/main.py eval --simulate --model-name TEMPLATE --fid-cap 200 --json runs/summary.jsonl --console runs/console.txt
