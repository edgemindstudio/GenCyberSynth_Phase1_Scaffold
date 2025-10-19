#!/usr/bin/env bash
#==============================================================================
# GenCyberSynth — Phase 3 Talon Scale-Up
# Slurm Array: (diffusion, cdcgan, cvae) × seeds (42, 43, 44)
#
# This script:
#   1) (optional) trains a model        -> summary_*.json
#   2) synthesizes per-class counts     -> summary_*.json  (records counts)
#   3) evaluates Real vs Real+Synth     -> summary_*.json  (records metrics & deltas)
#   4) consolidates JSON -> JSONL       -> artifacts/summaries/phase1_summaries.jsonl
#
# Requirements:
#   - app/main.py implements: train|synth|eval (each writing one summary_*.json)
#   - scripts/summaries_to_jsonl.py exists (Phase 1)
#   - (Recommended) Apptainer SIF at artifacts/containers/gcs-dev.sif
#
# Usage (on Talon):
#   module load apptainer || true
#   export USE_APPTAINER=1
#   export APPTAINER_IMG=$PWD/artifacts/containers/gcs-dev.sif
#   sbatch slurm/phase3_array.sbatch
#
# Tip:
#   make submit-array   # wraps sbatch with mkdirs (see Makefile helper)
#   make monitor        # watch queue
#   make tailf          # follow latest logs
#==============================================================================

###############################
# Slurm directives (Talon)
###############################
#SBATCH -J gcs_phase3                     # Job name
#SBATCH -p talon-short                    # Partition: talon-short|talon-long|talon-gpu32
#SBATCH -N 1                              # Nodes
#SBATCH -c 8                              # CPU cores per task
#SBATCH --mem=16G                         # Memory per node
#SBATCH -t 02:00:00                       # Max time (HH:MM:SS)
#SBATCH -o logs/%x_%A_%a.out              # STDOUT (logs/gcs_phase3_JOBID_TASKID.out)
#SBATCH -e logs/%x_%A_%a.err              # STDERR
#SBATCH --array=0-8                       # 3 models × 3 seeds = 9 tasks (0..8)

# --- GPU variant (uncomment if you need GPUs) ---
##SBATCH -p talon-gpu32
##SBATCH --gpus=1

set -euo pipefail

###############################
# User knobs (edit as needed)
###############################

# Models and seeds to sweep
MODELS=("diffusion" "cdcgan" "cvae")
SEEDS=(42 43 44)

# Core entrypoints and config
APP="app/main.py"
CFG="configs/config.yaml"
PYTHON=${PYTHON:-python}

# Per-class synthetic counts (JSON string)
# Example: boost minority class 5× (adjust for your class IDs and strategy)
PER_CLASS='{"0":400,"1":400,"2":400,"3":400,"4":400,"5":2000,"6":400,"7":400,"8":400}'

# Where summary_*.json lines are written per model
SUMMARIES_ROOT="artifacts"
GLOBAL_JSONL="artifacts/summaries/phase1_summaries.jsonl"

# Containerization (recommended on Talon)
USE_APPTAINER=${USE_APPTAINER:-0}
APPTAINER_IMG=${APPTAINER_IMG:-artifacts/containers/gcs-dev.sif}

###############################
# Derived variables
###############################
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_SEEDS=${#SEEDS[@]}
MIDX=$(( TASK_ID / NUM_SEEDS ))
SIDX=$(( TASK_ID % NUM_SEEDS ))

MODEL=${MODELS[$MIDX]}
SEED=${SEEDS[$SIDX]}

MODEL_SUMMARIES_DIR="${SUMMARIES_ROOT}/${MODEL}/summaries"

###############################
# Utilities
###############################
log()   { printf '%s %s\n' "$(date +'%F %T')" "$*" ; }
die()   { log "[FATAL]" "$*"; exit 1; }

# Run a command either natively or inside Apptainer, with Slurm CPU binding
run_py () {
  if [[ "$USE_APPTAINER" -eq 1 ]]; then
    srun -c "$SLURM_CPUS_ON_NODE" apptainer exec --cleanenv "$APPTAINER_IMG" "$@"
  else
    srun -c "$SLURM_CPUS_ON_NODE" "$@"
  fi
}

# Clean on abnormal termination
cleanup() {
  local code=$?
  if [[ $code -ne 0 ]]; then
    log "[ERROR] job=$SLURM_JOB_ID task=$TASK_ID model=$MODEL seed=$SEED exited with code $code"
  fi
}
trap cleanup EXIT

###############################
# Pre-flight checks
###############################
mkdir -p logs "$MODEL_SUMMARIES_DIR" "artifacts/summaries"

log "[INFO] job=$SLURM_JOB_ID task=$TASK_ID model=$MODEL seed=$SEED"
log "[INFO] partition=$SLURM_JOB_PARTITION cpus=$SLURM_CPUS_ON_NODE mem=${SLURM_MEM_PER_NODE:-NA}"

# Optional: basic file existence checks
[[ -f "$APP" ]] || die "Missing $APP"
[[ -f "$CFG" ]] || die "Missing $CFG"
if [[ "$USE_APPTAINER" -eq 1 ]]; then
  [[ -f "$APPTAINER_IMG" ]] || die "Missing Apptainer image: $APPTAINER_IMG"
fi

###############################
# 1) (Optional) Train
###############################
# Uncomment if your protocol includes training before synth/eval
# log "[STEP] train model=$MODEL seed=$SEED"
# run_py "$PYTHON" "$APP" train \
#   --model "$MODEL" --config "$CFG" --seed "$SEED" \
#   --out   "${MODEL_SUMMARIES_DIR}"

###############################
# 2) Synthesize with per-class counts
###############################
log "[STEP] synth model=$MODEL seed=$SEED"
run_py "$PYTHON" "$APP" synth \
  --model "$MODEL" --config "$CFG" --seed "$SEED" \
  --out   "${MODEL_SUMMARIES_DIR}" \
  --per-class "$PER_CLASS"

###############################
# 3) Evaluate (Real vs Real+Synth) and record metrics/deltas
###############################
log "[STEP] eval model=$MODEL seed=$SEED"
run_py "$PYTHON" "$APP" eval \
  --model "$MODEL" --config "$CFG" --seed "$SEED" \
  --out   "${MODEL_SUMMARIES_DIR}"

###############################
# 4) Consolidate per-run JSON → global JSONL
###############################
# Note: summaries_to_jsonl.py should be idempotent; --reset rebuilds the file.
log "[STEP] consolidate → ${GLOBAL_JSONL}"
$PYTHON scripts/summaries_to_jsonl.py \
  --glob "artifacts/*/summaries/summary_*.json" \
  --out  "${GLOBAL_JSONL}" \
  --reset || log "[WARN] consolidation failed; continuing (will aggregate later)"

###############################
# 5) Epilogue
###############################
# (Optional) Print quick stats
LINES=$(wc -l < "${GLOBAL_JSONL}" 2>/dev/null || echo 0)
log "[DONE] model=$MODEL seed=$SEED | jsonl_lines=${LINES}"
exit 0
