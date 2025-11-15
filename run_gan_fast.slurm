#!/usr/bin/env bash
#SBATCH -J gcs-gan-fast
#SBATCH -o gcs-gan-fast.%j.out
#SBATCH -e gcs-gan-fast.%j.err
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

# CPU only (silence TF CUDA probing on CPU nodes)
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=1

cd "$SLURM_SUBMIT_DIR"

# Train quickly on CPU with the merged config
python -m app.main train --model gan --config configs/config.gan_fast.yaml

# Synthesize and evaluate using same fast config + standard artifacts root
python -m app.main synth --model gan --config configs/config.gan_fast.yaml \
  --artifacts "${SCRATCH:-$HOME}/gencys/artifacts"

python -m app.main eval  --model gan --config configs/config.gan_fast.yaml \
  --artifacts "${SCRATCH:-$HOME}/gencys/artifacts"
