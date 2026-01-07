#!/usr/bin/env bash
#SBATCH -J gcs-gmm
#SBATCH -o gcs-gmm.%j.out
#SBATCH -e gcs-gmm.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
set -euo pipefail
module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310
export CUDA_VISIBLE_DEVICES=""
cd "$SLURM_SUBMIT_DIR"
python -m app.main train --model gaussianmixture --config configs/config.yaml
python -m app.main synth --model gaussianmixture --config configs/config.yaml --artifacts "${SCRATCH:-$HOME}/gencys/artifacts"
python -m app.main eval  --model gaussianmixture --config configs/config.yaml --artifacts "${SCRATCH:-$HOME}/gencys/artifacts"
