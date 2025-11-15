#!/bin/bash
#SBATCH -J gcs-fidkid
#SBATCH -p talon-lar
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o slurm/logs/gcs-fidkid_%j_%t.out
#SBATCH -e slurm/logs/gcs-fidkid_%j_%t.err

set -euo pipefail
REPO="/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"
CFG="$REPO/configs/config.yaml"
ARTS="/home/bruno.fonkeng/gencys/artifacts"
mkdir -p "$REPO/slurm/logs" "$REPO/logs"
cd "$REPO"

# Force CPU and tame threads/memory
export CUDA_VISIBLE_DEVICES=""
export TF_ENABLE_ONEDNN_OPTS=0      # save RAM vs oneDNN
export TF_NUM_INTRAOP_THREADS=6
export TF_NUM_INTEROP_THREADS=6
export OMP_NUM_THREADS=6

# conservative caps (raise later when stable)
PER_CLASS_CAP="${PER_CLASS_CAP:-30}"
TOTAL_CAP="${TOTAL_CAP:-270}"

MODELS=(gan diffusion autoregressive vae gaussianmixture restrictedboltzmann maskedautoflow)

echo "[fidkid] start $(date) host=$(hostname) caps: per_class=$PER_CLASS_CAP total=$TOTAL_CAP"
for M in "${MODELS[@]}"; do
  echo "[fidkid] === model: $M ==="
  python -u scripts/metrics/local_fid_kid.py \
    --config "$CFG" \
    --artifacts "$ARTS" \
    --model "$M" \
    --per_class_cap "$PER_CLASS_CAP" \
    --total_cap "$TOTAL_CAP"
done
echo "[fidkid] done $(date)"
