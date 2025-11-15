#!/usr/bin/env bash
#SBATCH -J gcs-rbm
#SBATCH -o gcs-rbm.%j.out
#SBATCH -e gcs-rbm.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 08:00:00
set -euo pipefail
module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=1
cd "$SLURM_SUBMIT_DIR"
python -m app.main train --model restrictedboltzmann --config configs/config.yaml
python -m app.main synth --model restrictedboltzmann --config configs/config.yaml --artifacts "${SCRATCH:-$HOME}/gencys/artifacts"
python -m app.main eval  --model restrictedboltzmann --config configs/config.yaml --artifacts "${SCRATCH:-$HOME}/gencys/artifacts"
