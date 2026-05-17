#!/bin/bash

#SBATCH --job-name=p3_diff_smoke
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/p3_diff_smoke_s42_b25_%j.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/p3_diff_smoke_s42_b25_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "[job] Paper 3 Diffusion synth/eval smoke seed42 b25"
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTEROP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=1

CONFIG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_balanced_b25.yaml"
ARTIFACTS="/home/bruno.fonkeng/gencys/artifacts_paper3"
MANIFEST="/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/synthetic/paper3_diffusion_aug_balanced_b25/seed42/manifest.json"

echo "[stage] Compile touched files"
python -m py_compile app/main.py adapters/diffusion_adapter.py diffusion/train.py diffusion/sample.py diffusion/pipeline.py diffusion/models.py eval/runner.py

echo "[stage] Synthesize diffusion seed42 b25"
python -m app.main synth --model diffusion --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Manifest check"
python - <<'PY'
import json
from pathlib import Path

p = Path("/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/synthetic/paper3_diffusion_aug_balanced_b25/seed42/manifest.json")

if not p.exists():
    raise FileNotFoundError(p)

m = json.load(open(p))

print("manifest:", p)
print("num_fake:", m.get("num_fake"))
print("budget_per_class:", m.get("budget_per_class"))
print("per_class_counts:", m.get("per_class_counts"))
print("paths:", len(m.get("paths", [])))

expected = 9 * 25
if len(m.get("paths", [])) != expected:
    raise RuntimeError(f"Expected {expected} synthetic paths, got {len(m.get('paths', []))}")

counts = m.get("per_class_counts", {})
bad = {k: v for k, v in counts.items() if int(v) != 25}
if bad:
    raise RuntimeError(f"Expected 25 per class, got bad counts: {bad}")
PY

echo "[stage] Eval diffusion seed42 b25"
python -m app.main eval --model diffusion --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Latest diffusion summary check"
python - <<'PY'
import json, glob
from pathlib import Path

paths = sorted(glob.glob("/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/summaries/summary_*.json"))
if not paths:
    raise RuntimeError("No diffusion summaries found.")

p = paths[-1]
s = json.load(open(p))

print("latest_summary:", p)
print("model:", s.get("model"))
print("config_id:", s.get("config_id"), (s.get("run_meta") or {}).get("config_id"))
print("seed:", (s.get("run_meta") or {}).get("seed", s.get("seed")))
print("deltas:", s.get("deltas_RS_minus_R") or (s.get("utility") or {}).get("deltas_RS_minus_R"))
print("manifest_path:", s.get("manifest_path") or (s.get("run_meta") or {}).get("manifest_path"))
PY

echo "[job] Done: $(date)"