#!/bin/bash
#SBATCH --job-name=p3_resnet_arr
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/p3_resnet_arr_%A_%a.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/p3_resnet_arr_%A_%a.err
#SBATCH --array=0-17%2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL=2

echo "[job] Paper 3 ResNet sensitivity array"
echo "[job] Array task: ${SLURM_ARRAY_TASK_ID}"
echo "[job] Host: $(hostname)"
echo "[job] Start: $(date)"

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate gcs_py310
fi

cd /home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold
mkdir -p papers/paper3_when_does_synth_help/slurm/logs
mkdir -p papers/paper3_when_does_synth_help/results/raw/resnet_sensitivity_parts

echo "[job] CWD: $(pwd)"
echo "[job] Git commit: $(git rev-parse --short HEAD)"
echo "[job] Python: $(which python)"
echo "[job] Python version: $(python --version)"

python - <<'PY' > /tmp/paper3_resnet_tasks_${SLURM_JOB_ID}.tsv
families = ["gan", "vae", "diffusion"]
regimes = ["balanced", "minority_c4c7"]
seeds = [42, 43, 44]
budget = 2000
i = 0
for regime in regimes:
    for family in families:
        for seed in seeds:
            print(f"{i}\t{family}\t{regime}\t{budget}\t{seed}")
            i += 1
PY

TASK_FILE="/tmp/paper3_resnet_tasks_${SLURM_JOB_ID}.tsv"
TASK_LINE=$(awk -v id="${SLURM_ARRAY_TASK_ID}" '$1 == id {print}' "$TASK_FILE")

if [ -z "$TASK_LINE" ]; then
  echo "[error] No task line for array id ${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

FAMILY=$(echo "$TASK_LINE" | awk '{print $2}')
REGIME=$(echo "$TASK_LINE" | awk '{print $3}')
BUDGET=$(echo "$TASK_LINE" | awk '{print $4}')
SEED=$(echo "$TASK_LINE" | awk '{print $5}')

OUT_RAW="papers/paper3_when_does_synth_help/results/raw/resnet_sensitivity_parts/resnet_${REGIME}_${FAMILY}_b${BUDGET}_seed${SEED}.csv"

echo "[task] family=${FAMILY} regime=${REGIME} budget=${BUDGET} seed=${SEED}"
echo "[task] out=${OUT_RAW}"

python -u papers/paper3_when_does_synth_help/scripts/paper3_classifier_sensitivity_resnet.py --families "${FAMILY}" --regimes "${REGIME}" --budgets "${BUDGET}" --seeds "${SEED}" --epochs 5 --out-raw "${OUT_RAW}" --no-aggregate

echo "[stage] Output check"
python - <<PY
from pathlib import Path
import pandas as pd
p = Path("${OUT_RAW}")
print("exists:", p.exists(), "size:", p.stat().st_size if p.exists() else None)
if not p.exists() or p.stat().st_size == 0:
    raise RuntimeError(f"Missing or empty output: {p}")
df = pd.read_csv(p)
print(df[["family", "regime", "budget_per_class", "seed", "delta_macro_f1", "delta_balanced_accuracy", "delta_macro_auprc", "delta_class4_f1", "delta_class7_f1"]].to_string(index=False))
PY

echo "[job] Done: $(date)"