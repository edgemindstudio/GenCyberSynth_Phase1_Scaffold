#!/usr/bin/env bash
#SBATCH -J gcs-ar
#SBATCH -o gcs-ar.%j.out
#SBATCH -e gcs-ar.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 16:00:00
set -euo pipefail
module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310
export CUDA_VISIBLE_DEVICES=""   # silence GPU probing on CPU nodes
export TF_CPP_MIN_LOG_LEVEL=1
cd "$SLURM_SUBMIT_DIR"
python -m app.main train --model autoregressive --config configs/config.yaml
python -m app.main synth --model autoregressive --config configs/config.yaml --artifacts "${SCRATCH:-$HOME}/gencys/artifacts"
python -m app.main eval  --model autoregressive --config configs/config.yaml --artifacts "${SCRATCH:-$HOME}/gencys/artifacts"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
