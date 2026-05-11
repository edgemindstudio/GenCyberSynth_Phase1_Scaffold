#!/bin/bash

#SBATCH --job-name=p2_multiclf

#SBATCH --output=papers/paper2_conditional_generation_done_right/slurm/logs/p2_multiclf_%j.out

#SBATCH --error=papers/paper2_conditional_generation_done_right/slurm/logs/p2_multiclf_%j.err

#SBATCH --time=10:00:00

#SBATCH --cpus-per-task=16

#SBATCH --mem=64G



set -euo pipefail



echo "[job] Paper 2 multi-classifier b100 audit"

echo "[job] Host: $(hostname)"

echo "[job] Start: $(date)"

echo "[job] CWD before cd: $(pwd)"



if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then

  source "$HOME/miniforge3/etc/profile.d/conda.sh"

  conda activate gcs_py310

fi



cd /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold



echo "[job] CWD: $(pwd)"

echo "[job] Git commit: $(git rev-parse --short HEAD)"

echo "[job] Python: $(which python)"

echo "[job] Python version: $(python --version)"



echo "[stage] Compile multi-classifier audit script"

python -m py_compile papers/paper2_conditional_generation_done_right/scripts/audit_conditioning_multiclf_v1.py



echo "[stage] Run multi-classifier audit"

python papers/paper2_conditional_generation_done_right/scripts/audit_conditioning_multiclf_v1.py --max-per-class 100 --classifiers resnet_cnn linear_svm_flat --resnet-epochs 30 --batch-size 128



echo "[stage] Show comparison table"

cat papers/paper2_conditional_generation_done_right/results/tables/paper2_audit_classifier_comparison.csv



echo "[job] Done: $(date)"
