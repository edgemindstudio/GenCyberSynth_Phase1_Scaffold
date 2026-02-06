#!/usr/bin/env bash
#SBATCH -J gcs-tune
#SBATCH -o gcs-tune.%A_%a.out
#SBATCH -e gcs-tune.%A_%a.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 1-00:00:00
#SBATCH --array=0-41%6

set -euo pipefail

module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

echo "[slurm] job_id=$SLURM_JOB_ID task_id=$SLURM_ARRAY_TASK_ID submit_dir=$SLURM_SUBMIT_DIR host=$(hostname)"
cd "$SLURM_SUBMIT_DIR"

# Quiet / deterministic-ish CPU behavior
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

# --- Grid definition ---
models=(gan vae diffusion autoregressive restrictedboltzmann gaussianmixture maskedautoflow)
cfgs=(A B)
seeds=(42 43 44)

N_MODELS=${#models[@]}   # 7
N_CFGS=${#cfgs[@]}       # 2
N_SEEDS=${#seeds[@]}     # 3

# Map task_id -> (model, cfg, seed)
tid=$SLURM_ARRAY_TASK_ID
midx=$(( tid / (N_CFGS * N_SEEDS) ))
rem=$(( tid % (N_CFGS * N_SEEDS) ))
cidx=$(( rem / N_SEEDS ))
sidx=$(( rem % N_SEEDS ))

MODEL="${models[$midx]}"
CFG_TAG="${cfgs[$cidx]}"               # A or B
SEED="${seeds[$sidx]}"

CONFIG_ID="${MODEL}_${CFG_TAG}"
BPC="${BPC:-200}"                      # tuning budget per class
CAP="${CAP:-200}"                      # evaluator cap per class

CFG="${CFG:-configs/config.yaml}"
ARTS="${ARTS:-${SCRATCH:-$HOME}/gencys/artifacts}"

echo "[grid] MODEL=$MODEL CONFIG_ID=$CONFIG_ID SEED=$SEED BPC=$BPC CAP=$CAP"
echo "[paths] CFG=$CFG ARTS=$ARTS PWD=$(pwd)"
python -V

# --- Overrides file (unique per task) ---
OV="/tmp/ov_${SLURM_JOB_ID}_${tid}.yaml"
cat > "$OV" <<YAML
SEED: ${SEED}
synth:
  n_per_class: ${BPC}
evaluator:
  per_class_cap: ${CAP}
run_meta:
  config_id: ${CONFIG_ID}
  slurm_job_id: "${SLURM_JOB_ID}"
  slurm_array_task_id: "${SLURM_ARRAY_TASK_ID}"
YAML

echo "[ov] $OV"
cat "$OV"

# --- Run pipeline (train optional; synth+eval required) ---
DO_TRAIN="${DO_TRAIN:-1}"
DO_SYNTH="${DO_SYNTH:-1}"
DO_EVAL="${DO_EVAL:-1}"

if [[ "$DO_TRAIN" == "1" ]]; then
  set +e
  python -u -m app.main train --model "$MODEL" --config "$CFG" --overrides "$OV" --artifacts "$ARTS"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[warn] train($MODEL) rc=$rc; continuing."
  fi
fi

if [[ "$DO_SYNTH" == "1" ]]; then
  python -u -m app.main synth --model "$MODEL" --config "$CFG" --overrides "$OV" --artifacts "$ARTS"
fi

if [[ "$DO_EVAL" == "1" ]]; then
  python -u -m app.main eval  --model "$MODEL" --config "$CFG" --overrides "$OV" --artifacts "$ARTS"
fi

# Quick proof printed to logs (optional)
LATEST="$ARTS/$MODEL/summaries/latest.json"
if [[ -f "$LATEST" ]]; then
  echo "[latest] $(jq -r '.run_id' "$LATEST")"
  echo "[latest] config_id=$(jq -r '.run_meta.config_id' "$LATEST")"
  echo "[latest] budget_pc=$(jq -r '.run_meta.budget_per_class' "$LATEST")"
fi
