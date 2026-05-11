#!/bin/bash

#SBATCH --job-name=p2_cic_s43

#SBATCH --output=papers/paper2_conditional_generation_done_right/slurm/logs/p2_cic_s43_%j.out

#SBATCH --error=papers/paper2_conditional_generation_done_right/slurm/logs/p2_cic_s43_%j.err

#SBATCH --time=08:00:00

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G



set -euo pipefail



echo "[job] Paper 2 CICMalDroid fake-class full seed43"

echo "[job] Host: $(hostname)"

echo "[job] Start: $(date)"



if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then

  source "$HOME/miniforge3/etc/profile.d/conda.sh"

  conda activate gcs_py310

fi



cd /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold



echo "[job] CWD: $(pwd)"

echo "[job] Git commit: $(git rev-parse --short HEAD)"

echo "[job] Python: $(which python)"

echo "[job] Python version: $(python --version)"



CONFIG="papers/paper2_conditional_generation_done_right/configs/paper2_cicmaldroid_fakeclass_seed43.yaml"

ARTIFACTS="/home/bruno.fonkeng/gencys/artifacts_paper2"

DATA_DIR="/home/bruno.fonkeng/gencys/data/CICMalDroid2020_pipeline_paper1"

MANIFEST="/home/bruno.fonkeng/gencys/artifacts_paper2/gan/synthetic/paper2_cicmaldroid_fakeclass_seed43/seed43/manifest.json"

TAG="paper2_cicmaldroid_fakeclass_seed43"



echo "[stage] Compile touched files"

python -m py_compile papers/paper2_conditional_generation_done_right/scripts/audit_conditioning_v2.py gan/train.py gan/train_acgan.py gan/sample.py



echo "[stage] Train full"

python -m app.main train --model gan --config "${CONFIG}" --artifacts "${ARTIFACTS}"



echo "[stage] Synthesize full"

python -m app.main synth --model gan --config "${CONFIG}" --artifacts "${ARTIFACTS}"



echo "[stage] Manifest check"

python - <<'PY'

import json

from pathlib import Path



p = Path("/home/bruno.fonkeng/gencys/artifacts_paper2/gan/synthetic/paper2_cicmaldroid_fakeclass_seed43/seed43/manifest.json")

if not p.exists():

    raise FileNotFoundError(p)



m = json.load(open(p))

print("manifest:", p)

print("num_fake:", m.get("num_fake"))

print("budget_per_class:", m.get("budget_per_class"))

print("per_class_counts:", m.get("per_class_counts"))

print("paths:", len(m.get("paths", [])))



if len(m.get("paths", [])) != 500:

    raise RuntimeError(f"Expected 500 synthetic paths, got {len(m.get('paths', []))}")

PY



echo "[stage] Eval full"

python -m app.main eval --model gan --config "${CONFIG}" --artifacts "${ARTIFACTS}"



echo "[stage] External conditioning audit full"

python papers/paper2_conditional_generation_done_right/scripts/audit_conditioning_v2.py --manifest "${MANIFEST}" --tag "${TAG}" --max-per-class 100 --data-dir "${DATA_DIR}" --dataset-id cicmaldroid2020



echo "[stage] Audit summary"

cat papers/paper2_conditional_generation_done_right/results/tables/paper2_gan_conditioning_summary_${TAG}.json | grep -E "dataset_id|data_dir|max_synth_per_class|total_synthetic_audited|overall_conditioning_accuracy|overall_conditioning_failure_rate|leakage_count|leakage_rate"



echo "[stage] Predicted-label histogram"

cat papers/paper2_conditional_generation_done_right/results/tables/paper2_gan_predicted_label_histogram_${TAG}.csv



echo "[job] Done: $(date)"
