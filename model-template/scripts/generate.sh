#!/usr/bin/env bash
set -euo pipefail
python app/main.py synth --simulate --model-name TEMPLATE --per-class 1000 --out artifacts/synthetic
