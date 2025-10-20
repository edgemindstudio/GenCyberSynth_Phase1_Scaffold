#!/usr/bin/env bash
#SBATCH -J gcs_matrix_cpu
#SBATCH -p cpu                 # Talon CPU partition; adjust if different
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

# -------- Tunables (override with sbatch --export=ALL,VAR=...) -------------
MODELS="${MODELS:-gan diffusion vae autoregressive maskedautoflow restrictedboltzmann gaussianmixture}"
SEEDS="${SEEDS:-42 43 44}"
SYN_PER_CLASS="${SYN_PER_CLASS:-1000}"

# Paths
REPO_DIR="${REPO_DIR:-$PWD}"                         # submit from repo root by default
WORKDIR_BASE="${WORKDIR_BASE:-$SCRATCH/gencys/runs}" # per-user scratch
REAL_ROOT="${REAL_ROOT:-$REPO_DIR/USTC-TFC2016_malware/real}"

# Python/venv
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$WORKDIR_BASE/.venv}"
# ---------------------------------------------------------------------------

echo "[info] node=$(hostname) job=$SLURM_JOB_ID"
echo "[info] MODELS=[$MODELS]"
echo "[info] SEEDS=[$SEEDS]"
echo "[info] REPO_DIR=$REPO_DIR"
echo "[info] WORKDIR_BASE=$WORKDIR_BASE"
echo "[info] REAL_ROOT=$REAL_ROOT"

mkdir -p "$WORKDIR_BASE"
cd "$REPO_DIR"

# Bootstrap venv once under WORKDIR_BASE (reused across runs)
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] venv -> $VENV_DIR"
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

# Prepare repo (creates summaries dir etc.)
make setup

# Matrix: model × seed
for m in $MODELS; do
  for s in $SEEDS; do
    RUN_DIR="${WORKDIR_BASE}/${m}_seed${s}_$(date +%Y%m%d_%H%M%S)"
    echo "[run] model=$m seed=$s -> $RUN_DIR"
    mkdir -p "$RUN_DIR"

    # One pass: train -> synth -> eval (uses your existing CLI)
    make onepass MODEL="$m" SEED="$s" SYN_PER_CLASS="$SYN_PER_CLASS" REAL_ROOT="$REAL_ROOT"

    # Consolidate (idempotent)
    make summaries-jsonl || true

    # Useful visuals/CSV (don’t fail the job if missing)
    make grids || true
    python scripts/jsonl_to_csv.py || true

    # Snapshot artifacts per run
    mkdir -p "$RUN_DIR/artifacts"
    rsync -a --delete "artifacts/" "$RUN_DIR/artifacts/"
  done
done

echo "[done] Matrix complete."
echo "[hint] Latest consolidated files in repo ./artifacts/:"
echo " - summaries/phase1_summaries.jsonl"
echo " - phase1_scores.csv"
echo " - preview_grids/*_grid.png"
