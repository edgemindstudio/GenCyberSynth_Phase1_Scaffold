#!/bin/bash
#SBATCH --job-name=paper2_template
#SBATCH --partition=talon
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=papers/paper2_conditional_generation_done_right/logs/%x.%A_%a.out
#SBATCH --error=papers/paper2_conditional_generation_done_right/logs/%x.%A_%a.err

set -euo pipefail

echo "===== Paper 2 Slurm Job Start ====="
echo "Job name: ${SLURM_JOB_NAME:-NA}"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node list: ${SLURM_JOB_NODELIST:-NA}"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo "==================================="

cd "$SLURM_SUBMIT_DIR"

mkdir -p papers/paper2_conditional_generation_done_right/logs
mkdir -p papers/paper2_conditional_generation_done_right/results/raw
mkdir -p papers/paper2_conditional_generation_done_right/manifests

echo "Current git branch:"
git branch --show-current || true

echo "Current git commit:"
git rev-parse HEAD || true

echo "Python:"
which python || true
python --version || true

echo "Conda environment:"
echo "${CONDA_DEFAULT_ENV:-NA}"

echo "This is a Paper 2 Slurm-safe template."
echo "Replace this section with the actual Paper 2 train/synth/eval command."

echo "End time: $(date)"
echo "===== Paper 2 Slurm Job End ====="