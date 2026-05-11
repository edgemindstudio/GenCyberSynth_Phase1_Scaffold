#!/bin/bash
#SBATCH --job-name=p2_b100_s43
#SBATCH --output=papers/paper2_conditional_generation_done_right/slurm/logs/p2_b100_s43_%j.out
#SBATCH --error=papers/paper2_conditional_generation_done_right/slurm/logs/p2_b100_s43_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "[job] Paper 2 b100 seed43 synth + audit"
echo "[job] Host: $(hostname)"
echo "[job] Start: $(date)"
echo "[job] CWD before cd: $(pwd)"

cd /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold

echo "[job] CWD: $(pwd)"
echo "[job] Git commit: $(git rev-parse --short HEAD)"
echo "[job] Python: $(which python)"
echo "[job] Python version: $(python --version)"

CONFIG="papers/paper2_conditional_generation_done_right/configs/paper2_acgan_fakeclass_b100_seed43.yaml"
ARTIFACTS="/home/bruno.fonkeng/gencys/artifacts_paper2"
MANIFEST="/home/bruno.fonkeng/gencys/artifacts_paper2/gan/synthetic/paper2_acgan_fakeclass_b100_seed43/seed43/manifest.json"
TAG="paper2_acgan_fakeclass_b100_seed43"

echo "[job] Config: ${CONFIG}"
echo "[job] Artifacts: ${ARTIFACTS}"
echo "[job] Manifest: ${MANIFEST}"
echo "[job] Tag: ${TAG}"

echo "[stage] Check checkpoint symlink"
ls -lh /home/bruno.fonkeng/gencys/artifacts_paper2/gan/checkpoints/paper2_acgan_fakeclass_b100_seed43
ls -lh /home/bruno.fonkeng/gencys/artifacts_paper2/gan/checkpoints/paper2_acgan_fakeclass_b100_seed43/seed43

echo "[stage] Synthesis"
python -m app.main synth --model gan --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Manifest check"
python - <<'PY'
import json
from pathlib import Path

p = Path("/home/bruno.fonkeng/gencys/artifacts_paper2/gan/synthetic/paper2_acgan_fakeclass_b100_seed43/seed43/manifest.json")
if not p.exists():
    raise FileNotFoundError(f"Missing manifest: {p}")

m = json.load(open(p))
print("manifest:", p)
print("num_fake:", m.get("num_fake"))
print("budget_per_class:", m.get("budget_per_class"))
print("per_class_counts:", m.get("per_class_counts"))
print("paths:", len(m.get("paths", [])))

expected = 900
actual = len(m.get("paths", []))
if actual != expected:
    raise RuntimeError(f"Expected {expected} paths but found {actual}")
PY

echo "[stage] External conditioning audit"
python papers/paper2_conditional_generation_done_right/scripts/audit_conditioning_v2.py --manifest "${MANIFEST}" --tag "${TAG}" --max-per-class 100

echo "[stage] Audit summary"
cat papers/paper2_conditional_generation_done_right/results/tables/paper2_gan_conditioning_summary_${TAG}.json | grep -E "total_synthetic_audited|overall_conditioning_accuracy|overall_conditioning_failure_rate|leakage_count|leakage_rate|minority_failure_rate" || true

echo "[stage] Predicted-label histogram"
cat papers/paper2_conditional_generation_done_right/results/tables/paper2_gan_predicted_label_histogram_${TAG}.csv

echo "[job] Done: $(date)"