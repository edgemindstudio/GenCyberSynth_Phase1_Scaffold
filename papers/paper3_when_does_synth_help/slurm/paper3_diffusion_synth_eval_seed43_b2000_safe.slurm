#!/bin/bash

#SBATCH --job-name=p3d_s43_b2000_safe

#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/p3d_s43_b2000_safe_%j.out

#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/p3d_s43_b2000_safe_%j.err

#SBATCH --time=12:00:00

#SBATCH --cpus-per-task=8

#SBATCH --mem=96G

#SBATCH --exclude=talon05



set -euo pipefail



echo "[job] Diffusion b2000 safe rerun seed43"

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

export PYTHONUNBUFFERED=1



CONFIG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_balanced_b2000_seed43.yaml"

ARTIFACTS="/home/bruno.fonkeng/gencys/artifacts_paper3"

MANIFEST="/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/synthetic/paper3_diffusion_aug_balanced_b2000/seed43/manifest.json"



echo "[stage] Synthesize diffusion seed43 b2000 with chunk_size=50"

python -u -m app.main synth --model diffusion --config "${CONFIG}" --artifacts "${ARTIFACTS}"



echo "[stage] Manifest check"

python - <<'PYCHK'

import json

from pathlib import Path



p = Path("/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/synthetic/paper3_diffusion_aug_balanced_b2000/seed43/manifest.json")

if not p.exists():

    raise FileNotFoundError(p)



m = json.load(open(p))

print("manifest:", p)

print("num_fake:", m.get("num_fake"))

print("budget_per_class:", m.get("budget_per_class"))

print("paths:", len(m.get("paths", [])))

print("per_class_counts:", m.get("per_class_counts"))



expected = 9 * 2000

if int(m.get("num_fake", -1)) != expected:

    raise RuntimeError(f"Expected num_fake={expected}, got {m.get('num_fake')}")

if len(m.get("paths", [])) != expected:

    raise RuntimeError(f"Expected {expected} paths, got {len(m.get('paths', []))}")



bad = {k: v for k, v in m.get("per_class_counts", {}).items() if int(v) != 2000}

if bad:

    raise RuntimeError(f"Bad class counts: {bad}")

PYCHK



echo "[stage] Evaluate diffusion seed43 b2000"

python -u -m app.main eval --model diffusion --config "${CONFIG}" --artifacts "${ARTIFACTS}"



echo "[stage] Latest matching summary"

python - <<'PYSUM'

import json, glob



files = sorted(glob.glob("/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/summaries/summary_*.json"))

hits = []



for p in files:

    s = json.load(open(p))

    rm = s.get("run_meta") or {}

    cid = s.get("config_id") or rm.get("config_id")

    sd = rm.get("seed", s.get("seed"))

    if cid == "paper3_diffusion_aug_balanced_b2000" and int(sd) == 43:

        hits.append((p, s))



if not hits:

    raise RuntimeError("No matching summary found")



p, s = hits[-1]

print("latest_summary:", p)

print("config_id:", s.get("config_id"), (s.get("run_meta") or {}).get("config_id"))

print("seed:", (s.get("run_meta") or {}).get("seed", s.get("seed")))

print("deltas:", s.get("deltas_RS_minus_R") or (s.get("utility") or {}).get("deltas_RS_minus_R"))

print("manifest_path:", s.get("manifest_path"))

PYSUM



echo "[job] Done: $(date)"

