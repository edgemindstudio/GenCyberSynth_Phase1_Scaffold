#!/usr/bin/env bash
# scripts/build_jsonl.sh
set -euo pipefail

OUT_JSONL="artifacts/summaries/phase1_summaries.jsonl"
SCHEMA_PATH="gcs-core/gcs_core/schemas/eval_summary.lite.schema.json"
DOWNLOADED_PREFIX="phase1-artifacts-raw"

schema_arg=""
if [ -f "$SCHEMA_PATH" ]; then
  echo "Using schema: $SCHEMA_PATH"
  schema_arg="--schema $SCHEMA_PATH"
else
  echo "Schema not found → skipping JSON Schema validation (fast path)"
fi

# Pass 1: local artifacts produced on this runner
if compgen -G "artifacts/*/summaries/summary_*.json" >/dev/null; then
  echo "[pass1] artifacts/*/summaries/summary_*.json"
  python scripts/summaries_to_jsonl.py \
    --glob "artifacts/*/summaries/summary_*.json" \
    --out "$OUT_JSONL" $schema_arg --reset

# Pass 2: CI-downloaded bundle (artifact prefix folder)
elif compgen -G "$DOWNLOADED_PREFIX/artifacts/*/summaries/summary_*.json" >/dev/null; then
  echo "[pass2] $DOWNLOADED_PREFIX/artifacts/*/summaries/summary_*.json"
  python scripts/summaries_to_jsonl.py \
    --glob "$DOWNLOADED_PREFIX/artifacts/*/summaries/summary_*.json" \
    --out "$OUT_JSONL" $schema_arg --reset

else
  echo "No per-model summaries found under either path." >&2
  exit 2
fi

echo "Built $OUT_JSONL"
