#!/bin/bash
#SBATCH --job-name=p3d_s44_b100
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/p3d_s44_b100_%j.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/p3d_s44_b100_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "[job] Diffusion synth+eval seed44 budget100"
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

CONFIG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_balanced_b100_seed44.yaml"
ARTIFACTS="/home/bruno.fonkeng/gencys/artifacts_paper3"
MANIFEST="/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/synthetic/paper3_diffusion_aug_balanced_b100/seed44/manifest.json"
EXPECTED_CID="paper3_diffusion_aug_balanced_b100"
EXPECTED_SEED=44
EXPECTED_BUDGET=100
EXPECTED_NUM_FAKE=900

echo "[stage] Synthesize"
python -m app.main synth --model diffusion --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Manifest check"
python - <<PY
import json
from pathlib import Path

p = Path("/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/synthetic/paper3_diffusion_aug_balanced_b100/seed44/manifest.json")
if not p.exists():
    raise FileNotFoundError(p)

m = json.load(open(p))
print("manifest:", p)
print("num_fake:", m.get("num_fake"))
print("budget_per_class:", m.get("budget_per_class"))
print("per_class_counts:", m.get("per_class_counts"))
print("paths:", len(m.get("paths", [])))

expected_num_fake = 100 * 9
if int(m.get("num_fake", -1)) != expected_num_fake:
    raise RuntimeError(f"Expected num_fake={expected_num_fake}, got {m.get('num_fake')}")

if len(m.get("paths", [])) != expected_num_fake:
    raise RuntimeError(f"Expected {expected_num_fake} synthetic paths, got {len(m.get('paths', []))}")
PY

echo "[stage] Evaluate"
python -m app.main eval --model diffusion --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Latest diffusion summary check"
python - <<PY
import json, glob

files = sorted(glob.glob("/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/summaries/summary_*.json"))
hits = []
for p in files:
    try:
        s = json.load(open(p))
    except Exception:
        continue
    rm = s.get("run_meta") or {}
    cid = s.get("config_id") or rm.get("config_id")
    seed = rm.get("seed", s.get("seed"))
    if cid == "paper3_diffusion_aug_balanced_b100" and int(seed) == 44:
        hits.append((p, s))

if not hits:
    raise RuntimeError("No matching diffusion summary found")

p, s = hits[-1]
rm = s.get("run_meta") or {}
print("latest_summary:", p)
print("model:", s.get("model"))
print("config_id:", s.get("config_id"), rm.get("config_id"))
print("seed:", rm.get("seed", s.get("seed")))
print("deltas:", s.get("deltas_RS_minus_R") or (s.get("utility") or {}).get("deltas_RS_minus_R"))
print("manifest_path:", s.get("manifest_path"))
PY

echo "[job] Done: $(date)"
