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

# Activate conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

# Work from the submission dir
cd "$SLURM_SUBMIT_DIR"
echo "PWD=$(pwd)"
python -V

# Pick the model: gan | vae | diffusion | autoregressive | restrictedboltzmann | gaussianmixture | maskedautoflow
MODEL="${MODEL:-gan}"
SCRIPT="${MODEL}/train.py"
CONFIG="configs/config.yaml"

echo "MODEL=${MODEL}"
echo "SCRIPT=${SCRIPT}"
echo "CONFIG=${CONFIG}"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: Trainer not found: $SCRIPT"
  echo "Available trainers:"
  find . -maxdepth 3 -type f -iname "train*.py" | sed 's|^\./||'
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Config not found: $CONFIG"
  ls -lah configs || true
  exit 1
fi

# Some repos use a different arg than --config; detect it:
if grep -qE -- '--config' "$SCRIPT"; then
  ARGS=(--config "$CONFIG")
else
  ARGS=("$CONFIG")
fi

echo "Running: python $SCRIPT ${ARGS[*]}"
python "$SCRIPT" "${ARGS[@]}"
