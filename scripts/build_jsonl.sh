#!/usr/bin/env bash
# scripts/build_jsonl.sh
# Consolidate per-model summary_*.json → artifacts/summaries/phase1_summaries.jsonl
# Works both locally and in CI (handles several artifact layouts).

set -euo pipefail

OUT_JSONL="${OUT_JSONL:-artifacts/summaries/phase1_summaries.jsonl}"
SCHEMA_PATH="${SCHEMA_PATH:-gcs-core/gcs_core/schemas/eval_summary.lite.schema.json}"

echo "Building consolidated JSONL…"

# Always initialize as an array (safe under set -u)
declare -a schema_arg=()
if [[ -f "${SCHEMA_PATH}" ]]; then
  echo "Using schema: ${SCHEMA_PATH}"
  schema_arg=(--schema "${SCHEMA_PATH}")
else
  echo "Schema not found → skipping JSON Schema validation (fast path)"
fi

# Safe expansion for empty arrays under bash 3.x + set -u:
#   "${schema_arg[@]:-}" expands to nothing if the array is empty,
#   avoiding the "unbound variable" error.
_safe_schema_expansion() {
  # shellcheck disable=SC2128
  printf '%s\n' "${schema_arg[@]:-}"
}

run_pass () {
  local label="$1"
  local glob="$2"
  if compgen -G "${glob}" >/dev/null; then
    echo "[${label}] ${glob}"
    python scripts/summaries_to_jsonl.py \
      --glob "${glob}" \
      --out "${OUT_JSONL}" \
      $(_safe_schema_expansion) \
      --reset
    echo "Built ${OUT_JSONL}"
    return 0
  fi
  return 1
}

# Try, in order:
#  1) Local runner outputs
#  2) Downloaded artifact with an 'artifacts/' prefix
#  3) Downloaded artifact flattened one level (no 'artifacts/' prefix)
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
