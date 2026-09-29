#!/usr/bin/env bash
set -euo pipefail

: "${PHASE1_DATASET:?PHASE1_DATASET is required}"
: "${PHASE1_SEED:?PHASE1_SEED is required}"
: "${TRUSTFORGE_ARTIFACTS_ROOT:?TRUSTFORGE_ARTIFACTS_ROOT is required}"

PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

"${PYTHON_BIN}" tools/check_phase1_integrity.py \
  --dataset "${PHASE1_DATASET}" \
  --seed "${PHASE1_SEED}"

"${PYTHON_BIN}" tools/build_phase1_scores.py \
  --dataset "${PHASE1_DATASET}" \
  --seed "${PHASE1_SEED}"

echo "[phase1_gate] OK"
