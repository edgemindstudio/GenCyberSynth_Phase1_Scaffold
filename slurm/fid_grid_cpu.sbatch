#!/bin/bash
#SBATCH -J gcs-fidgrid
#SBATCH -p talon-large
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -a 0-41
#SBATCH -o slurm/logs/gcs-fidgrid_%A_%a.out
#SBATCH -e slurm/logs/gcs-fidgrid_%A_%a.err

set -euo pipefail

# ---- Environment: force CPU, quiet TF logs, tame threading ----
export CUDA_VISIBLE_DEVICES=""
export TF_FORCE_CPU=1
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-1}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}
export BATCH=${BATCH:-16}

MODELS=(gan diffusion autoregressive vae gaussianmixture restrictedboltzmann maskedautoflow)  # 7
BACKBONES=(mobilenet inception)   # 2
CAPS=(6 10 15)                    # 3  -> total combos = 7*2*3 = 42 (array 0..41)

mi=${SLURM_ARRAY_TASK_ID}

midx=$(( mi % ${#MODELS[@]} ))
tmp=$(( mi / ${#MODELS[@]} ))
bidx=$(( tmp % ${#BACKBONES[@]} ))
cidx=$(( tmp / ${#BACKBONES[@]} ))

MODEL=${MODELS[$midx]}
BACKBONE=${BACKBONES[$bidx]}
PER_CLASS_CAP=${CAPS[$cidx]}
# you have 9 classes; keep total proportional
TOTAL_CAP=$(( PER_CLASS_CAP * 9 ))

# img size per backbone
if [[ "$BACKBONE" == "mobilenet" ]]; then
  IMG_SIZE=160
else
  IMG_SIZE=299
fi

echo "[fidgrid] $(date) host=$(hostname) model=$MODEL backbone=$BACKBONE img=$IMG_SIZE per_class=$PER_CLASS_CAP total=$TOTAL_CAP batch=$BATCH"

# Run
python -u scripts/metrics/local_fid_kid.py \
  --model "$MODEL" \
  --backbone "$BACKBONE" \
  --img_size "$IMG_SIZE" \
  --per_class_cap "$PER_CLASS_CAP" \
  --total_cap "$TOTAL_CAP" \
| tee >(grep -Eo '\[fidkid\] FID=[0-9.]+')

# Parse FID from stdout and append a row to artifacts/summaries/fid_grid_runs.csv
mkdir -p artifacts/summaries
FID=$(grep -Eo 'FID=[0-9.]+' "slurm/logs/gcs-fidgrid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" | tail -n1 | cut -d= -f2 || true)
echo "job_id,task_id,model,backbone,img_size,per_class_cap,total_cap,batch,fid" > artifacts/summaries/_fid_grid_header.tmp
echo "${SLURM_ARRAY_JOB_ID},${SLURM_ARRAY_TASK_ID},${MODEL},${BACKBONE},${IMG_SIZE},${PER_CLASS_CAP},${TOTAL_CAP},${BATCH},${FID}" >> artifacts/summaries/fid_grid_runs.csv

echo "[fidgrid] done $(date) model=$MODEL backbone=$BACKBONE FID=${FID:-NA}"
