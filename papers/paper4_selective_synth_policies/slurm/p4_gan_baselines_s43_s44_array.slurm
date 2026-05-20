#!/bin/bash
#SBATCH --job-name=p4_gan_base
#SBATCH --output=papers/paper4_selective_synth_policies/slurm/logs/p4_gan_base_%A_%a.out
#SBATCH --error=papers/paper4_selective_synth_policies/slurm/logs/p4_gan_base_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5%1

set -euo pipefail

echo "[job] Paper4 GAN baseline train+synth+eval"
echo "[job] Host: $(hostname)"
echo "[job] Start: $(date)"
echo "[job] Array task: ${SLURM_ARRAY_TASK_ID}"

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate gcs_py310
fi

cd /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold

echo "[job] CWD: $(pwd)"
echo "[job] Git commit: $(git rev-parse --short HEAD)"
echo "[job] Python: $(which python)"
echo "[job] Python version: $(python --version)"

ARTIFACTS="/home/bruno.fonkeng/gencys/artifacts_paper4"
MODEL="gan"

CONFIGS=(
"papers/paper4_selective_synth_policies/configs/paper4_baseline_b25_s43.yaml"
"papers/paper4_selective_synth_policies/configs/paper4_baseline_b25_s44.yaml"
"papers/paper4_selective_synth_policies/configs/paper4_baseline_b500_s43.yaml"
"papers/paper4_selective_synth_policies/configs/paper4_baseline_b500_s44.yaml"
"papers/paper4_selective_synth_policies/configs/paper4_baseline_b2000_s43.yaml"
"papers/paper4_selective_synth_policies/configs/paper4_baseline_b2000_s44.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "[job] CONFIG=${CONFIG}"
echo "[stage] Config check"

python - <<PY
import yaml, json
from pathlib import Path

cfg_path = Path("${CONFIG}")
cfg = yaml.safe_load(cfg_path.read_text())

rm = cfg.get("run_meta", {})
config_id = rm.get("config_id")
seed = cfg.get("synth", {}).get("seed")
budget = rm.get("regime", {}).get("synth_budget_per_class")
expected_num_fake = int(budget) * 9

print("config:", cfg_path)
print("config_id:", config_id)
print("seed:", seed)
print("budget:", budget)
print("expected_num_fake:", expected_num_fake)

if config_id not in {"paper4_baseline_b25", "paper4_baseline_b500", "paper4_baseline_b2000"}:
    raise RuntimeError(f"Unexpected config_id: {config_id}")

if int(seed) not in {43, 44}:
    raise RuntimeError(f"Unexpected seed: {seed}")

if int(budget) not in {25, 500, 2000}:
    raise RuntimeError(f"Unexpected budget: {budget}")
PY

echo "[stage] Train"
python -m app.main train --model "${MODEL}" --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Synthesize"
python -m app.main synth --model "${MODEL}" --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Manifest check"
python - <<PY
import json, yaml
from pathlib import Path

cfg_path = Path("${CONFIG}")
cfg = yaml.safe_load(cfg_path.read_text())

rm = cfg.get("run_meta", {})
config_id = rm.get("config_id")
seed = int(cfg.get("synth", {}).get("seed"))
budget = int(rm.get("regime", {}).get("synth_budget_per_class"))
expected_num_fake = budget * 9

p = Path("${ARTIFACTS}") / "gan" / "synthetic" / config_id / f"seed{seed}" / "manifest.json"

if not p.exists():
    raise FileNotFoundError(p)

m = json.load(open(p))

print("manifest:", p)
print("num_fake:", m.get("num_fake"))
print("budget_per_class:", m.get("budget_per_class"))
print("per_class_counts:", m.get("per_class_counts"))
print("paths:", len(m.get("paths", [])))

if int(m.get("num_fake", -1)) != expected_num_fake:
    raise RuntimeError(f"Expected num_fake={expected_num_fake}, got {m.get('num_fake')}")

if len(m.get("paths", [])) != expected_num_fake:
    raise RuntimeError(f"Expected {expected_num_fake} paths, got {len(m.get('paths', []))}")
PY

echo "[stage] Evaluate"
python -m app.main eval --model "${MODEL}" --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Summary check"
python - <<PY
import json, glob, yaml
from pathlib import Path

cfg_path = Path("${CONFIG}")
cfg = yaml.safe_load(cfg_path.read_text())

rm_cfg = cfg.get("run_meta", {})
expected_cid = rm_cfg.get("config_id")
expected_seed = int(cfg.get("synth", {}).get("seed"))
expected_budget = int(rm_cfg.get("regime", {}).get("synth_budget_per_class"))

files = sorted(glob.glob("${ARTIFACTS}/gan/summaries/summary_*.json"))
hits = []

for p in files:
    try:
        s = json.load(open(p))
    except Exception:
        continue

    rm = s.get("run_meta") or {}
    cid = s.get("config_id") or rm.get("config_id")
    seed = rm.get("seed", s.get("seed"))
    budget = s.get("budget_per_class") or rm.get("budget_per_class")

    try:
        seed = int(seed)
    except Exception:
        continue

    try:
        budget = int(budget)
    except Exception:
        budget = None

    if cid == expected_cid and seed == expected_seed and budget == expected_budget:
        hits.append((p, s))

if not hits:
    raise RuntimeError(f"No matching summary found for config_id={expected_cid}, seed={expected_seed}, budget={expected_budget}")

p, s = hits[-1]
rm = s.get("run_meta") or {}

print("latest_summary:", p)
print("run_id:", s.get("run_id"))
print("config_id:", s.get("config_id"), rm.get("config_id"))
print("seed:", rm.get("seed", s.get("seed")))
print("budget:", s.get("budget_per_class"), rm.get("budget_per_class"))
print("num_fake:", s.get("counts", {}).get("num_fake") if isinstance(s.get("counts"), dict) else s.get("counts.num_fake"))
print("manifest_path:", s.get("manifest_path"))
print("deltas:", s.get("deltas_RS_minus_R") or (s.get("utility") or {}).get("deltas_RS_minus_R"))
PY

echo "[job] Done: $(date)"