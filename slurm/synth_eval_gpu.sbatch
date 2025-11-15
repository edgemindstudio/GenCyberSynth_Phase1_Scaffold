#!/bin/bash
#SBATCH -p talon-gpu32
#SBATCH -t 02:00:00
#SBATCH -J gcs-synth-eval-gpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -D /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold
set -euxo pipefail
source ~/env-setup.sh 2>/dev/null || true
REPO="/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
MODELS=(gan diffusion)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
CFG="$REPO/configs/config.yaml"
ARTS="${SCRATCH:-$HOME}/gencys/artifacts"
echo "GPU synth+eval -> $M"
python -m app.main synth --model "$M" --config "$CFG" --artifacts "$ARTS"
python -m app.main eval  --model "$M" --config "$CFG" --artifacts "$ARTS"
