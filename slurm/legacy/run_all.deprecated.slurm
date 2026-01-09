#!/usr/bin/env bash
#SBATCH -J gcs-suite
#SBATCH -o gcs-suite.%A_%a.out
#SBATCH -e gcs-suite.%A_%a.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --array=0-6

set -euo pipefail

# ---- Modules / Conda --------------------------------------------------------
module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

# Keep single-threaded math libs from oversubscribing
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# Force CPU (safe on login-submitted batch jobs)
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export XLA_FLAGS="--xla_gpu_cuda_data_dir="

cd "$SLURM_SUBMIT_DIR"
echo "PWD=$(pwd)"
echo "HOST=$(hostname)"
python -V

# ---- Config knobs (override via --export=ALL,KEY=VAL) -----------------------
CONFIG="${CFG:-configs/config.yaml}"
SYN_PER_CLASS="${SYN_PER_CLASS:-500}"   # ramp this: 500 -> 1000 -> 2000
SEEDS="${SEEDS:-42}"                    # space-separated, e.g. "42 43 44"
DO_TRAIN="${DO_TRAIN:-1}"               # 1=train, 0=skip
DO_SYNTH="${DO_SYNTH:-1}"               # 1=synthesize
DO_EVAL="${DO_EVAL:-1}"                 # 1=evaluate

# Array of models in this sweep
models=(gan vae diffusion autoregressive restrictedboltzmann gaussianmixture maskedautoflow)
MODEL="${models[$SLURM_ARRAY_TASK_ID]}"

echo "TASK=$SLURM_ARRAY_TASK_ID MODEL=$MODEL"
echo "CONFIG=$CONFIG SYN_PER_CLASS=$SYN_PER_CLASS SEEDS='$SEEDS'"
echo "DO_TRAIN=$DO_TRAIN DO_SYNTH=$DO_SYNTH DO_EVAL=$DO_EVAL"

# ---- Train (resume if ckpts exist) ------------------------------------------
if [[ "$DO_TRAIN" == "1" ]]; then
  echo "[train] $MODEL"
  # Prefer the unified CLI if present; fallback to per-model train.py
  if python -m app.main --help >/dev/null 2>&1; then
    python -m app.main train --model "$MODEL" --config "$CONFIG" || true
  else
    SCRIPT="$MODEL/train.py"
    if [[ -f "$SCRIPT" ]]; then
      set +e
      python "$SCRIPT" --config "$CONFIG"
      rc=$?; set -e
      if [[ $rc -ne 0 ]]; then
        echo "[train] retry positional config"
        python "$SCRIPT" "$CONFIG"
      fi
    else
      echo "[train][warn] no trainer at $SCRIPT — skipping"
    fi
  fi
fi

# ---- Synthesize (per seed) --------------------------------------------------
if [[ "$DO_SYNTH" == "1" ]]; then
  echo "[synth] $MODEL × seeds: $SEEDS  (per-class=$SYN_PER_CLASS)"
  for s in $SEEDS; do
    python -m app.main synth \
      --model "$MODEL" \
      --config "$CONFIG" \
      --seed "$s" \
      --syn-per-class "$SYN_PER_CLASS"
  done
fi

# ---- Evaluate (per seed; writes JSON summaries) -----------------------------
if [[ "$DO_EVAL" == "1" ]]; then
  echo "[eval] $MODEL × seeds: $SEEDS"
  for s in $SEEDS; do
    python -m app.main eval \
      --model "$MODEL" \
      --config "$CONFIG" \
      --seed "$s"
  done
fi

echo "DONE: $MODEL"
