#!/usr/bin/env bash
#SBATCH -J gcs-ar-train
#SBATCH -o gcs-ar-train.%j.out
#SBATCH -e gcs-ar-train.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 48:00:00

set -euo pipefail

module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

# CPU-only; keep TF quiet and predictable on CPU
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

cd "$SLURM_SUBMIT_DIR"
python -V
echo "[info] starting AR full training…"
python -m app.main train --model autoregressive --config configs/config.yaml
echo "[done] training"
