#!/bin/bash
set -euo pipefail

BASE="papers/paper3_when_does_synth_help/slurm"
LOGS="$BASE/logs"
mkdir -p "$LOGS"

REPO="/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"
ARTIFACTS="/home/bruno.fonkeng/gencys/artifacts_paper3"

for seed in 42 43 44; do
  for budget in 25 100 500 2000; do
    if [ "$seed" = "42" ]; then
      CFG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_balanced_b${budget}.yaml"
    else
      CFG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_balanced_b${budget}_seed${seed}.yaml"
    fi

    CID="paper3_diffusion_aug_balanced_b${budget}"
    JOB="p3d_s${seed}_b${budget}"
    OUT="$BASE/paper3_diffusion_synth_eval_seed${seed}_b${budget}.slurm"
    MANIFEST="${ARTIFACTS}/diffusion/synthetic/${CID}/seed${seed}/manifest.json"

    cat > "$OUT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB}
#SBATCH --output=${LOGS}/${JOB}_%j.out
#SBATCH --error=${LOGS}/${JOB}_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "[job] Diffusion synth+eval seed${seed} budget${budget}"
echo "[job] Host: \$(hostname)"
echo "[job] Start: \$(date)"

if [ -f "\$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate gcs_py310
fi

cd ${REPO}

echo "[job] CWD: \$(pwd)"
echo "[job] Git commit: \$(git rev-parse --short HEAD)"
echo "[job] Python: \$(which python)"
echo "[job] Python version: \$(python --version)"

CONFIG="${CFG}"
ARTIFACTS="${ARTIFACTS}"
MANIFEST="${MANIFEST}"
EXPECTED_CID="${CID}"
EXPECTED_SEED=${seed}
EXPECTED_BUDGET=${budget}
EXPECTED_NUM_FAKE=$((9 * ${budget}))

echo "[stage] Synthesize"
python -m app.main synth --model diffusion --config "\${CONFIG}" --artifacts "\${ARTIFACTS}"

echo "[stage] Manifest check"
python - <<PY
import json
from pathlib import Path

p = Path("${MANIFEST}")
if not p.exists():
    raise FileNotFoundError(p)

m = json.load(open(p))
print("manifest:", p)
print("num_fake:", m.get("num_fake"))
print("budget_per_class:", m.get("budget_per_class"))
print("per_class_counts:", m.get("per_class_counts"))
print("paths:", len(m.get("paths", [])))

expected_num_fake = ${budget} * 9
if int(m.get("num_fake", -1)) != expected_num_fake:
    raise RuntimeError(f"Expected num_fake={expected_num_fake}, got {m.get('num_fake')}")

if len(m.get("paths", [])) != expected_num_fake:
    raise RuntimeError(f"Expected {expected_num_fake} synthetic paths, got {len(m.get('paths', []))}")
PY

echo "[stage] Evaluate"
python -m app.main eval --model diffusion --config "\${CONFIG}" --artifacts "\${ARTIFACTS}"

echo "[stage] Latest diffusion summary check"
python - <<PY
import json, glob

files = sorted(glob.glob("${ARTIFACTS}/diffusion/summaries/summary_*.json"))
hits = []
for p in files:
    try:
        s = json.load(open(p))
    except Exception:
        continue
    rm = s.get("run_meta") or {}
    cid = s.get("config_id") or rm.get("config_id")
    seed = rm.get("seed", s.get("seed"))
    if cid == "${CID}" and int(seed) == ${seed}:
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

echo "[job] Done: \$(date)"
EOF

    chmod +x "$OUT"
    echo "[ok] wrote $OUT"
  done
done

for seed in 42 43 44; do
  if [ "$seed" = "42" ]; then
    CFG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_realonly_balanced.yaml"
  else
    CFG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_realonly_balanced_seed${seed}.yaml"
  fi

  CID="paper3_diffusion_realonly_balanced"
  JOB="p3d_real_s${seed}"
  OUT="$BASE/paper3_diffusion_realonly_eval_seed${seed}.slurm"

  cat > "$OUT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB}
#SBATCH --output=${LOGS}/${JOB}_%j.out
#SBATCH --error=${LOGS}/${JOB}_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "[job] Diffusion real-only eval seed${seed}"
echo "[job] Host: \$(hostname)"
echo "[job] Start: \$(date)"

if [ -f "\$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate gcs_py310
fi

cd ${REPO}

echo "[job] CWD: \$(pwd)"
echo "[job] Git commit: \$(git rev-parse --short HEAD)"
echo "[job] Python: \$(which python)"
echo "[job] Python version: \$(python --version)"

CONFIG="${CFG}"
ARTIFACTS="${ARTIFACTS}"

echo "[stage] Real-only evaluate"
python -m app.main eval --model diffusion --config "\${CONFIG}" --artifacts "\${ARTIFACTS}"

echo "[stage] Latest diffusion summary check"
python - <<PY
import json, glob

files = sorted(glob.glob("${ARTIFACTS}/diffusion/summaries/summary_*.json"))
hits = []
for p in files:
    try:
        s = json.load(open(p))
    except Exception:
        continue
    rm = s.get("run_meta") or {}
    cid = s.get("config_id") or rm.get("config_id")
    seed = rm.get("seed", s.get("seed"))
    if cid == "${CID}" and int(seed) == ${seed}:
        hits.append((p, s))

if not hits:
    raise RuntimeError("No matching diffusion real-only summary found")

p, s = hits[-1]
rm = s.get("run_meta") or {}
print("latest_summary:", p)
print("model:", s.get("model"))
print("config_id:", s.get("config_id"), rm.get("config_id"))
print("seed:", rm.get("seed", s.get("seed")))
print("deltas:", s.get("deltas_RS_minus_R") or (s.get("utility") or {}).get("deltas_RS_minus_R"))
print("manifest_path:", s.get("manifest_path"))
PY

echo "[job] Done: \$(date)"
EOF

  chmod +x "$OUT"
  echo "[ok] wrote $OUT"
done

echo "[done] All diffusion Paper 3 slurm scripts created."