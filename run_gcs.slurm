#!/usr/bin/env bash
#SBATCH -J gcs-suite
#SBATCH -o gcs-suite.%A_%a.out
#SBATCH -e gcs-suite.%A_%a.err
#SBATCH -p talon
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --array=0-6

set -euo pipefail

module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export XLA_FLAGS="--xla_gpu_cuda_data_dir="

cd "$SLURM_SUBMIT_DIR"
echo "PWD=$(pwd)"
echo "HOST=$(hostname)"
python -V

CONFIG="${CFG:-configs/config.yaml}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_SYNTH="${DO_SYNTH:-1}"
DO_EVAL="${DO_EVAL:-1}"

models=(gan vae diffusion autoregressive maskedautoflow restrictedboltzmann gaussianmixture)
MODEL="${models[$SLURM_ARRAY_TASK_ID]}"

echo "TASK=$SLURM_ARRAY_TASK_ID MODEL=$MODEL"
echo "CONFIG=$CONFIG"
echo "DO_TRAIN=$DO_TRAIN DO_SYNTH=$DO_SYNTH DO_EVAL=$DO_EVAL"

if [[ "$DO_TRAIN" == "1" ]]; then
  echo "[train] $MODEL"
  srun --cpu-bind=cores python -m app.main train --model "$MODEL" --config "$CONFIG"
fi

if [[ "$DO_SYNTH" == "1" ]]; then
  echo "[synth] $MODEL"
  srun --cpu-bind=cores python -m app.main synth --model "$MODEL" --config "$CONFIG"
fi

if [[ "$DO_EVAL" == "1" ]]; then
  echo "[eval] $MODEL"
  srun --cpu-bind=cores python -m app.main eval --model "$MODEL" --config "$CONFIG"
fi

echo "DONE: $MODEL"
