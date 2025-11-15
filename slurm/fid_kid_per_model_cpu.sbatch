#!/bin/bash
#SBATCH -J gcs-fidkid
#SBATCH -p talon-large
#SBATCH --array=0-6
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:20:00
#SBATCH -o slurm/logs/gcs-fidkid_%A_%a.out
#SBATCH -e slurm/logs/gcs-fidkid_%A_%a.err

set -euo pipefail

REPO="/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"
CFG="$REPO/configs/config.yaml"
ARTS="/home/bruno.fonkeng/gencys/artifacts"

MODELS=(gan diffusion autoregressive vae gaussianmixture restrictedboltzmann maskedautoflow)
M="${MODELS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$REPO/slurm/logs" "$REPO/logs"
cd "$REPO"

# Hard-disable GPU; keep threading tiny to avoid RAM spikes
export CUDA_VISIBLE_DEVICES=-1
export TF_FORCE_CPU=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-1}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TF_CPP_MIN_LOG_LEVEL=2

# Backbone defaults (stay consistent with your summaries)
BACKBONE=${BACKBONE:-mobilenet}
IMG_SIZE=${IMG_SIZE:-160}

# Small caps to prove stability; can bump later
PER_CLASS_CAP=${PER_CLASS_CAP:-6}
TOTAL_CAP=${TOTAL_CAP:-54}
BATCH=${BATCH:-16}

echo "[fidkid-per-model] $(date) host=$(hostname) model=$M caps: per_class=$PER_CLASS_CAP total=$TOTAL_CAP batch=$BATCH backbone=$BACKBONE img=$IMG_SIZE"

python -u scripts/metrics/local_fid_kid.py \
  --config "$CFG" \
  --artifacts "$ARTS" \
  --model "$M" \
  --per_class_cap "$PER_CLASS_CAP" \
  --total_cap "$TOTAL_CAP" \
  --backbone "$BACKBONE" \
  --img_size "$IMG_SIZE" \
  --batch "$BATCH"

echo "[fidkid-per-model] done $(date) model=$M"
