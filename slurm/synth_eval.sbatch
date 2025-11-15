#!/bin/bash
#SBATCH -p talon-large
#SBATCH -t 04:00:00
#SBATCH -J gcs-synth-eval
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -D /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold

set -euxo pipefail

# env
source ~/env-setup.sh 2>/dev/null || true
REPO="/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# models
MODELS=(autoregressive gan diffusion vae gaussianmixture restrictedboltzmann maskedautoflow)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}

# paths
CFG="$REPO/configs/config.yaml"
ARTS="${SCRATCH:-$HOME}/gencys/artifacts"

echo "PWD=$(pwd)"
which python
python -V
ls -l app/main.py
echo "Running \$M"
python -m app.main synth --model "$M" --config "$CFG" --artifacts "$ARTS"
python -m app.main eval  --model "$M" --config "$CFG" --artifacts "$ARTS"
