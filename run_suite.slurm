#!/usr/bin/env bash
#SBATCH -J gcs-suite
#SBATCH -o gcs-suite.%A_%a.out
#SBATCH -e gcs-suite.%A_%a.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 1-00:00:00
# Run all 7 families; cap concurrency to 3 at once:
#SBATCH --array=0-6%3

set -euo pipefail

# --- ENV (modules & conda) ---
module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

echo "[slurm] job_id=$SLURM_JOB_ID submit_dir=$SLURM_SUBMIT_DIR host=$(hostname)"
cd "$SLURM_SUBMIT_DIR" || { echo "Cannot cd to submit dir"; exit 1; }

# Sanity: show where python is and ensure we’re in repo root
which python
python -c "import os; print('[py] cwd=', os.getcwd())"
python -c "import eval.runner, scripts; print('[py] runner=', eval.runner.__file__)"


# --- Make TF quiet on CPU-only nodes ---
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3

export TF_ENABLE_ONEDNN_OPTS=0     # avoid CPU kernel variance
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

cd "$SLURM_SUBMIT_DIR"

# Models in order; index by SLURM_ARRAY_TASK_ID
models=(gan vae diffusion autoregressive restrictedboltzmann gaussianmixture maskedautoflow)
MODEL="${models[$SLURM_ARRAY_TASK_ID]}"

CFG="${CFG:-configs/config.yaml}"
ARTS="${SCRATCH:-$HOME}/gencys/artifacts"

DO_TRAIN="${DO_TRAIN:-1}"
DO_SYNTH="${DO_SYNTH:-1}"
DO_EVAL="${DO_EVAL:-1}"

echo "[info] TASK=${SLURM_ARRAY_TASK_ID} MODEL=${MODEL}"
echo "[info] CFG=${CFG}"
echo "[info] ARTS=${ARTS}"
echo "[info] PWD=$(pwd)"
python -V

# --- TRAIN (best-effort; don't kill the array task if trainer returns nonzero) ---
if [[ "$DO_TRAIN" == "1" ]]; then
  set +e
  python -u -m app.main train --model "$MODEL" --config "$CFG" --artifacts "$ARTS"
  train_rc=$?
  set -e
  if [[ $train_rc -ne 0 ]]; then
    echo "[warn] train($MODEL) rc=${train_rc}; continuing."
  fi
fi

# --- SYNTH (required to produce manifest) ---
if [[ "$DO_SYNTH" == "1" ]]; then
  python -u -m app.main synth --model "$MODEL" --config "$CFG" --artifacts "$ARTS"
fi

# --- EVAL (run metrics on latest manifest) ---
if [[ "$DO_EVAL" == "1" ]]; then
  python -u -m app.main eval  --model "$MODEL" --config "$CFG" --artifacts "$ARTS"
fi

