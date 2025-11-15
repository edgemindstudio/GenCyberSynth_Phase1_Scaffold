#!/bin/bash
#SBATCH -p talon-large
#SBATCH -t 04:00:00
#SBATCH -J gcs-synth-eval-cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -D /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold
set -euxo pipefail
source ~/env-setup.sh 2>/dev/null || true
REPO="/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
MODELS=(autoregressive vae gaussianmixture restrictedboltzmann maskedautoflow)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
CFG="$REPO/configs/config.yaml"
ARTS="${SCRATCH:-$HOME}/gencys/artifacts"
echo "CPU synth+eval -> $M"
python -m app.main synth --model "$M" --config "$CFG" --artifacts "$ARTS"
python -m app.main eval  --model "$M" --config "$CFG" --artifacts "$ARTS"
