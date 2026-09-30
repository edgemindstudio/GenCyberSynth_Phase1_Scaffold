#!/usr/bin/env bash
# scripts/build_jsonl.sh
#
# Transitional Paper 1 JSONL compatibility wrapper.
#
# Canonical mode:
#   PHASE1_DATASET and PHASE1_SEED are both set.
#   Authority comes from TrustForge canonical linkage.
#
# Legacy compatibility mode:
#   Neither variable is set.
#   Historical/CI glob collection behavior is preserved.
#
# A partial scientific identity fails closed.

set -euo pipefail

OUT_JSONL="${OUT_JSONL:-artifacts/summaries/phase1_summaries.jsonl}"
SCHEMA_PATH="${SCHEMA_PATH:-gcs-core/gcs_core/schemas/eval_summary.lite.schema.json}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON:-python}"

dataset="${PHASE1_DATASET:-}"
seed="${PHASE1_SEED:-}"

if [[ -n "${dataset}" || -n "${seed}" ]]; then
  if [[ -z "${dataset}" ]]; then
    echo "PHASE1_DATASET is required when PHASE1_SEED is set" >&2
    exit 2
  fi
  if [[ -z "${seed}" ]]; then
    echo "PHASE1_SEED is required when PHASE1_DATASET is set" >&2
    exit 2
  fi

  echo "Building canonical Paper 1 JSONL compatibility export…"
  echo "[canonical] dataset=${dataset} seed=${seed}"

  cmd=(
    "${PYTHON_BIN}"
    scripts/summaries_to_jsonl.py
    --canonical
    --repo-root "${REPO_ROOT}"
    --dataset "${dataset}"
    --seed "${seed}"
    --out "${OUT_JSONL}"
  )

  if [[ -f "${SCHEMA_PATH}" ]]; then
    echo "Using schema: ${SCHEMA_PATH}"
    cmd+=(--schema "${SCHEMA_PATH}")
  else
    echo "Schema not found → skipping JSON Schema validation"
  fi

  if [[ "${RESET_JSONL:-0}" == "1" ]]; then
    cmd+=(--reset)
  fi

  set +e
  PYTHONPATH="${REPO_ROOT}/src" "${cmd[@]}"
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 || "${rc}" -eq 2 ]]; then
    echo "Built ${OUT_JSONL}"
    exit 0
  fi

  echo "ERROR: canonical summaries_to_jsonl.py failed (exit=${rc})" >&2
  exit "${rc}"
fi

echo "Building consolidated JSONL in legacy compatibility mode…"

run_pass () {
  local label="$1"
  local glob="$2"

  if compgen -G "${glob}" >/dev/null; then
    echo "[${label}] ${glob}"

    cmd=(
      "${PYTHON_BIN}"
      scripts/summaries_to_jsonl.py
      --glob "${glob}"
      --out "${OUT_JSONL}"
    )

    if [[ -f "${SCHEMA_PATH}" ]]; then
      echo "Using schema: ${SCHEMA_PATH}"
      cmd+=(--schema "${SCHEMA_PATH}")
    else
      echo "Schema not found → skipping JSON Schema validation (fast path)"
    fi

    if [[ "${RESET_JSONL:-0}" == "1" ]]; then
      cmd+=(--reset)
    fi

    set +e
    "${cmd[@]}"
    rc=$?
    set -e

    if [[ "${rc}" -eq 0 || "${rc}" -eq 2 ]]; then
      echo "Built ${OUT_JSONL}"
      return 0
    fi

    echo "ERROR: summaries_to_jsonl.py failed (exit=${rc})" >&2
    return "${rc}"
  fi

  return 1
}

run_pass "pass1" "artifacts/*/summaries/summary_*.json" \
|| run_pass "pass2" "phase1-artifacts-raw/artifacts/*/summaries/summary_*.json" \
|| run_pass "pass3" "phase1-artifacts-raw/*/summaries/summary_*.json" \
|| {
  echo "No per-model summaries found under any known path." >&2
  echo "Workspace snapshot (top-level):"
  ls -la || true
  echo "Tree under ./phase1-artifacts-raw (if present):"
  ls -la phase1-artifacts-raw || true
  exit 2
}
