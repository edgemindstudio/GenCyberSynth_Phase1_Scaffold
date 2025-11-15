#!/usr/bin/env bash
#SBATCH -J gcs-all
#SBATCH -o gcs-all.%A_%a.out
#SBATCH -e gcs-all.%A_%a.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --array=0-6

set -euo pipefail

module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

# Keep single-threaded libs from oversubscribing
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export XLA_FLAGS="--xla_gpu_cuda_data_dir="  # keep XLA from looking for CUDA


cd "$SLURM_SUBMIT_DIR"
echo "PWD=$(pwd)"
echo "HOST=$(hostname)"
python -V

models=(gan vae diffusion autoregressive restrictedboltzmann gaussianmixture maskedautoflow)
MODEL="${models[$SLURM_ARRAY_TASK_ID]}"
SCRIPT="${MODEL}/train.py"
CONFIG="configs/config.yaml"

echo "TASK=$SLURM_ARRAY_TASK_ID MODEL=$MODEL"
echo "SCRIPT=$SCRIPT"
echo "CONFIG=$CONFIG"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: Trainer not found: $SCRIPT"
  find . -maxdepth 3 -type f -iname "train*.py" | sed 's|^\./||'
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Config not found: $CONFIG"
  ls -lah configs || true
  exit 1
fi

# Some trainers may not use --config; if this flag fails instantly, retry with positional config
set +e
python "$SCRIPT" --config "$CONFIG"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "Retrying without --config (positional)..."
  python "$SCRIPT" "$CONFIG"
fi
