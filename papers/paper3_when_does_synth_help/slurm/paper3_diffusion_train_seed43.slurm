#!/bin/bash

#SBATCH --job-name=p3_diff_s43
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/p3_diff_s43_%j.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/p3_diff_s43_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "[job] Paper 3 Diffusion training seed43"
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

CONFIG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_balanced_b25_seed43.yaml"
ARTIFACTS="/home/bruno.fonkeng/gencys/artifacts_paper3"
CKPT_DIR="/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/checkpoints/seed43"

echo "[stage] Compile touched files"
python -m py_compile app/main.py adapters/diffusion_adapter.py diffusion/train.py diffusion/sample.py diffusion/pipeline.py diffusion/models.py eval/runner.py

echo "[stage] Train diffusion seed43"
python -m app.main train --model diffusion --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Check diffusion checkpoints"
python - <<'PY'
from pathlib import Path

ckpt_dir = Path("/home/bruno.fonkeng/gencys/artifacts_paper3/diffusion/checkpoints/seed43")
print("checkpoint_dir:", ckpt_dir)

if not ckpt_dir.exists():
    raise FileNotFoundError(ckpt_dir)

files = sorted(p.name for p in ckpt_dir.glob("*"))
print("files:", files)

expected_any = [
    "DDPM_best.weights.h5",
    "DDPM_last.weights.h5",
    "DIFF_best.weights.h5",
    "DIFF_last.weights.h5",
    "DIF_best.weights.h5",
    "DIF_last.weights.h5",
]

found = [name for name in expected_any if (ckpt_dir / name).exists()]
print("recognized_checkpoints:", found)

if not found:
    raise RuntimeError("No recognized diffusion checkpoint was written.")
PY

echo "[job] Done: $(date)"