#!/usr/bin/env bash
#SBATCH -J gcs-maf
#SBATCH -o gcs-maf.%j.out
#SBATCH -e gcs-maf.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 04:00:00

set -euo pipefail
module purge
module load slurm/slurm/24.05.7
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

# CPU-only & quieter TF logs
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

cd "$SLURM_SUBMIT_DIR"

CFG="configs/config.yaml"
ARTS="${SCRATCH:-$HOME}/gencys/artifacts"

echo "[info] MODEL=maskedautoflow"
python -V
echo "[info] CFG=${CFG}"
echo "[info] ARTS=${ARTS}"

# Train (best-effort), then synth + eval
set +e
python -m app.main train --model maskedautoflow --config "${CFG}"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "[warn] train returned rc=$rc, continuing to synth/eval."
fi

python -m app.main synth --model maskedautoflow --config "${CFG}" --artifacts "${ARTS}"
python -m app.main eval  --model maskedautoflow --config "${CFG}" --artifacts "${ARTS}"
echo "[done] maskedautoflow"
