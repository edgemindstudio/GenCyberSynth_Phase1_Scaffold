#!/usr/bin/env bash
#SBATCH -J gcs_matrix_gpu
#SBATCH -p talon-gpu32        # adjust to your GPU partition
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

# Optional: load site modules (uncomment/tune for Talon)
# module load cuda/12.1 || true

MODELS="${MODELS:-gan diffusion vae}"   # start with GPU-benefiting adapters
SEEDS="${SEEDS:-42 43 44}"
SYN_PER_CLASS="${SYN_PER_CLASS:-1000}"

REPO_DIR="${REPO_DIR:-$PWD}"
WORKDIR_BASE="${WORKDIR_BASE:-$SCRATCH/gencys/runs}"
REAL_ROOT="${REAL_ROOT:-$REPO_DIR/USTC-TFC2016_malware/real}"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$WORKDIR_BASE/.venv-gpu}"

echo "[info] node=$(hostname) job=$SLURM_JOB_ID (GPU)"
mkdir -p "$WORKDIR_BASE"
cd "$REPO_DIR"

if [ ! -d "$VENV_DIR" ]; then
  $PYTHON -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  if [ -f requirements.ci.txt ]; then
    pip install -r requirements.ci.txt
  else
    pip install -r requirements.txt
  fi
else
  source "$VENV_DIR/bin/activate"
fi

make setup

for m in $MODELS; do
  for s in $SEEDS; do
    RUN_DIR="${WORKDIR_BASE}/${m}_seed${s}_$(date +%Y%m%d_%H%M%S)"
    echo "[run] model=$m seed=$s -> $RUN_DIR"
    mkdir -p "$RUN_DIR"

    # If your app uses TF/PyTorch GPU, it should auto-detect GPU
    make onepass MODEL="$m" SEED="$s" SYN_PER_CLASS="$SYN_PER_CLASS" REAL_ROOT="$REAL_ROOT"

    make summaries-jsonl || true
    make grids || true
    python scripts/jsonl_to_csv.py || true

    mkdir -p "$RUN_DIR/artifacts"
    rsync -a --delete "artifacts/" "$RUN_DIR/artifacts/"
  done
done

echo "[done] Matrix complete (GPU)."
