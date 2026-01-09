#!/bin/bash
#SBATCH -p talon-large
#SBATCH -t 04:00:00
#SBATCH -J gcs-eval-only
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --array=0-6
#SBATCH -o slurm/logs/%x_%A_%a.out
#SBATCH -e slurm/logs/%x_%A_%a.err
#SBATCH -D /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold
set -euo pipefail

source ~/env-setup.sh 2>/dev/null || true
REPO="/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Force CPU mode & suppress TF CUDA init spam
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3

MODELS=(gan diffusion autoregressive vae gaussianmixture restrictedboltzmann maskedautoflow)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}

CFG="$REPO/configs/config_2000.yaml"
ARTS="/home/bruno.fonkeng/gencys/artifacts"

echo "[eval-only] model=$M cfg=$CFG arts=$ARTS host=$(hostname) $(date)"
python -m app.main eval --model "$M" --config "$CFG" --artifacts "$ARTS"
