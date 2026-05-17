#!/bin/bash
#SBATCH --job-name=p3dm_s44_b2000
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/p3dm_s44_b2000_%j.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/p3dm_s44_b2000_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

echo "[job] Paper 3 diffusion minority-heavy synth+eval seed44 b2000"
echo "[job] Host: $(hostname)"
echo "[job] Start: $(date)"

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate gcs_py310
fi

cd "/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"

echo "[job] CWD: $(pwd)"
echo "[job] Git commit: $(git rev-parse --short HEAD)"
echo "[job] Python: $(which python)"
echo "[job] Python version: $(python --version)"
echo "[job] Config: papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b2000_seed44.yaml"

echo "[stage] Synthesize diffusion minority-heavy seed44 b2000"
python -m app.main synth --model diffusion --config "papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b2000_seed44.yaml" --artifacts "/home/bruno.fonkeng/gencys/artifacts_paper3"

echo "[stage] Manifest check"
python - <<'PY'
import json
from pathlib import Path

p = Path("/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/synthetic/paper3_diffusion_aug_minority_heavy_c4c7_b2000/seed44/manifest.json")
if not p.exists():
    raise FileNotFoundError(p)

m = json.load(open(p))
print("manifest:", p)
print("class_ids:", m.get("class_ids"))
print("num_fake:", m.get("num_fake"))
print("budget_per_class:", m.get("budget_per_class"))
print("paths:", len(m.get("paths", [])))
print("per_class_counts:", m.get("per_class_counts"))

expected = int(2000) * 2
if len(m.get("paths", [])) != expected:
    raise RuntimeError(f"Expected {expected} paths, got {len(m.get('paths', []))}")

pcc = m.get("per_class_counts") or {}
if int(pcc.get("4", -1)) != int(2000) or int(pcc.get("7", -1)) != int(2000):
    raise RuntimeError(f"Expected classes 4 and 7 to have budget 2000; got {pcc}")
PY

echo "[stage] Evaluate diffusion minority-heavy seed44 b2000"
python -m app.main eval --model diffusion --config "papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b2000_seed44.yaml" --artifacts "/home/bruno.fonkeng/gencys/artifacts_paper3"

echo "[stage] Latest matching summary"
python - <<'PY'
import json, glob

target = "paper3_diffusion_aug_minority_heavy_c4c7_b2000"
seed = int(44)
paths = sorted(glob.glob("/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/summaries/summary_*.json"))

hits = []
for p in paths:
    try:
        s = json.load(open(p))
    except Exception:
        continue
    cid = s.get("config_id") or (s.get("run_meta") or {}).get("config_id")
    sd = (s.get("run_meta") or {}).get("seed", s.get("seed"))
    try:
        sd = int(sd)
    except Exception:
        pass
    if cid == target and sd == seed:
        hits.append((p, s))

if not hits:
    raise RuntimeError(f"No matching summary found for {target} seed {seed}")

p, s = hits[-1]
d = s.get("deltas_RS_minus_R") or (s.get("utility") or {}).get("deltas_RS_minus_R")
print("latest_summary:", p)
print("config_id:", s.get("config_id"), (s.get("run_meta") or {}).get("config_id"))
print("seed:", (s.get("run_meta") or {}).get("seed", s.get("seed")))
print("deltas:", d)
print("manifest_path:", s.get("manifest_path") or (s.get("run_meta") or {}).get("manifest_path"))
PY

echo "[job] Done: $(date)"
