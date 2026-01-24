#!/bin/bash
#SBATCH --job-name=gcs_backfill_p1
#SBATCH --partition=talon-gpu32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/backfill_p1_%j.out
#SBATCH --error=slurm/backfill_p1_%j.err

set -euo pipefail

cd /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

export PHASE1_SUMMARY_NAME=paper1.json
export PHASE1_MANIFEST_NAME=paper1_manifest.json
export MANIFEST_CAP_PER_CLASS=200

# Optional: make TF calmer
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/backfill_kid_and_downstream.py
