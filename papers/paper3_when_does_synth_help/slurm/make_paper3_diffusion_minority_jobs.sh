#!/bin/bash
set -euo pipefail

mkdir -p papers/paper3_when_does_synth_help/slurm/logs

REPO="/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"
ARTS="/home/bruno.fonkeng/gencys/artifacts_paper3"
SLURM_DIR="papers/paper3_when_does_synth_help/slurm"

write_aug_job () {
  seed="$1"
  budget="$2"

  if [ "$seed" = "42" ]; then
    CONFIG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b${budget}.yaml"
  else
    CONFIG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b${budget}_seed${seed}.yaml"
  fi

  JOB="p3dm_s${seed}_b${budget}"
  OUT="${SLURM_DIR}/paper3_diffusion_minority_synth_eval_seed${seed}_b${budget}.slurm"
  MANIFEST="${ARTS}/diffusion/synthetic/paper3_diffusion_aug_minority_heavy_c4c7_b${budget}/seed${seed}/manifest.json"

  cat > "$OUT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB}
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/${JOB}_%j.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/${JOB}_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

echo "[job] Paper 3 diffusion minority-heavy synth+eval seed${seed} b${budget}"
echo "[job] Host: \$(hostname)"
echo "[job] Start: \$(date)"

if [ -f "\$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate gcs_py310
fi

cd "${REPO}"

echo "[job] CWD: \$(pwd)"
echo "[job] Git commit: \$(git rev-parse --short HEAD)"
echo "[job] Python: \$(which python)"
echo "[job] Python version: \$(python --version)"
echo "[job] Config: ${CONFIG}"

echo "[stage] Synthesize diffusion minority-heavy seed${seed} b${budget}"
python -m app.main synth --model diffusion --config "${CONFIG}" --artifacts "${ARTS}"

echo "[stage] Manifest check"
python - <<'PY'
import json
from pathlib import Path

p = Path("${MANIFEST}")
if not p.exists():
    raise FileNotFoundError(p)

m = json.load(open(p))
print("manifest:", p)
print("class_ids:", m.get("class_ids"))
print("num_fake:", m.get("num_fake"))
print("budget_per_class:", m.get("budget_per_class"))
print("paths:", len(m.get("paths", [])))
print("per_class_counts:", m.get("per_class_counts"))

expected = int(${budget}) * 2
if len(m.get("paths", [])) != expected:
    raise RuntimeError(f"Expected {expected} paths, got {len(m.get('paths', []))}")

pcc = m.get("per_class_counts") or {}
if int(pcc.get("4", -1)) != int(${budget}) or int(pcc.get("7", -1)) != int(${budget}):
    raise RuntimeError(f"Expected classes 4 and 7 to have budget ${budget}; got {pcc}")
PY

echo "[stage] Evaluate diffusion minority-heavy seed${seed} b${budget}"
python -m app.main eval --model diffusion --config "${CONFIG}" --artifacts "${ARTS}"

echo "[stage] Latest matching summary"
python - <<'PY'
import json, glob

target = "paper3_diffusion_aug_minority_heavy_c4c7_b${budget}"
seed = int(${seed})
paths = sorted(glob.glob("${ARTS}/diffusion/summaries/summary_*.json"))

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

echo "[job] Done: \$(date)"
EOF

  chmod +x "$OUT"
  echo "[ok] wrote $OUT"
}

write_real_job () {
  seed="$1"

  if [ "$seed" = "42" ]; then
    CONFIG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_realonly_minority_heavy_c4c7.yaml"
  else
    CONFIG="papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_realonly_minority_heavy_c4c7_seed${seed}.yaml"
  fi

  JOB="p3dm_real_s${seed}"
  OUT="${SLURM_DIR}/paper3_diffusion_minority_realonly_seed${seed}.slurm"

  cat > "$OUT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB}
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/${JOB}_%j.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/${JOB}_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "[job] Paper 3 diffusion minority-heavy real-only seed${seed}"
echo "[job] Host: \$(hostname)"
echo "[job] Start: \$(date)"

if [ -f "\$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "\$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate gcs_py310
fi

cd "${REPO}"

echo "[job] CWD: \$(pwd)"
echo "[job] Git commit: \$(git rev-parse --short HEAD)"
echo "[job] Python: \$(which python)"
echo "[job] Python version: \$(python --version)"
echo "[job] Config: ${CONFIG}"

echo "[stage] Evaluate real-only diffusion minority-heavy seed${seed}"
python -m app.main eval --model diffusion --config "${CONFIG}" --artifacts "${ARTS}"

echo "[job] Done: \$(date)"
EOF

  chmod +x "$OUT"
  echo "[ok] wrote $OUT"
}

# real-only jobs
for seed in 42 43 44; do
  write_real_job "$seed"
done

# augmentation jobs
write_aug_job 42 500
write_aug_job 42 2000

for seed in 43 44; do
  for budget in 100 500 2000; do
    write_aug_job "$seed" "$budget"
  done
done

echo "[done] Diffusion minority-heavy Slurm scripts created."