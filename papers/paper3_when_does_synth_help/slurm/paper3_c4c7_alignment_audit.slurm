#!/bin/bash
#SBATCH --job-name=p3_c47_audit
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/p3_c47_audit_%j.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/p3_c47_audit_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

echo "[job] Paper3 c4/c7 synthetic alignment audit"
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

mkdir -p papers/paper3_when_does_synth_help/slurm/logs

echo "[stage] Syntax check"
python -m py_compile papers/paper3_when_does_synth_help/scripts/paper3_audit_c4c7_synthetic_alignment.py

echo "[stage] Manifest count check"
python - <<'PY'
from pathlib import Path
root = Path("/home/bruno.fonkeng/gencys/artifacts_paper3")
manifests = sorted(root.glob("*/synthetic/*minority_heavy_c4c7*/seed*/manifest.json"))
print("manifest_count:", len(manifests))
if len(manifests) < 27:
    raise RuntimeError(f"Expected at least 27 manifests, found {len(manifests)}")
PY

echo "[stage] Full alignment audit"
python papers/paper3_when_does_synth_help/scripts/paper3_audit_c4c7_synthetic_alignment.py --max-per-class 2000

echo "[stage] Output check"
python - <<'PY'
from pathlib import Path
paths = [
    Path("papers/paper3_when_does_synth_help/results/raw/paper3_c4c7_synthetic_alignment_raw.csv"),
    Path("papers/paper3_when_does_synth_help/results/raw/paper3_c4c7_synthetic_alignment_pred_hist.csv"),
    Path("papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_aggregate_frozen_20260524.csv"),
    Path("papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_table.md"),
    Path("papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_table.tex"),
    Path("papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_summary_20260524.json"),
]
for p in paths:
    print(p, "exists=", p.exists(), "size=", p.stat().st_size if p.exists() else None)
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {p}")
PY

echo "[stage] Preview table"
cat papers/paper3_when_does_synth_help/results/frozen/paper3_c4c7_synthetic_alignment_table.md

echo "[job] Done: $(date)"