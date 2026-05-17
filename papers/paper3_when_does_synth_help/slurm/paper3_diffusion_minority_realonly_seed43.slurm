#!/bin/bash
#SBATCH --job-name=p3dm_real_s43
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/p3dm_real_s43_%j.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/p3dm_real_s43_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "[job] Paper 3 diffusion minority-heavy real-only seed43"
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
echo "[job] Config: papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_realonly_minority_heavy_c4c7_seed43.yaml"

echo "[stage] Evaluate real-only diffusion minority-heavy seed43"
python -m app.main eval --model diffusion --config "papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_realonly_minority_heavy_c4c7_seed43.yaml" --artifacts "/home/bruno.fonkeng/gencys/artifacts_paper3"

echo "[job] Done: $(date)"
