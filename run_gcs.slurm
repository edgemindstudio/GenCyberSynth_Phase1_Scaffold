#!/usr/bin/env bash
#SBATCH -J gcs-train
#SBATCH -o gcs.%j.out
#SBATCH -e gcs.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH -t 02:00:00

set -euo pipefail
module purge
module load slurm/slurm/24.05.7

export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export XLA_FLAGS="--xla_gpu_cuda_data_dir="  # keep XLA from looking for CUDA


# Activate conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

# Run from the directory where sbatch was invoked
cd "$SLURM_SUBMIT_DIR"
echo "PWD=$(pwd)"
python -V

# Choose your real script & config here:
SCRIPT="scripts/train.py"
CONFIG="configs/phase1.yaml"

if [[ -f "$SCRIPT" ]]; then
  echo "Running: python $SCRIPT --config $CONFIG"
  python "$SCRIPT" --config "$CONFIG"
else
  echo "Missing $SCRIPT. Listing candidates..."
  find . -maxdepth 3 -type f -iname "train*.py" -o -name "*main*.py" | sed 's|^\./||'
  echo "Listing scripts/ and configs/ for reference:"
  ls -lah scripts || true
  ls -lah configs  || true
  exit 1
fi
