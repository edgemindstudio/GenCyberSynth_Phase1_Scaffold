#!/bin/bash
#SBATCH --job-name=p3_c47_eval
#SBATCH --output=papers/paper3_when_does_synth_help/slurm/logs/p3_c47_eval_%A_%a.out
#SBATCH --error=papers/paper3_when_does_synth_help/slurm/logs/p3_c47_eval_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-26%1

set -euo pipefail

echo "[job] Paper3 minority-heavy c4/c7 per-class eval"
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

ARTIFACTS="/home/bruno.fonkeng/gencys/artifacts_paper3"

MODELS=(
"gan"
"gan"
"gan"
"gan"
"gan"
"gan"
"gan"
"gan"
"gan"
"vae"
"vae"
"vae"
"vae"
"vae"
"vae"
"vae"
"vae"
"vae"
"diffusion"
"diffusion"
"diffusion"
"diffusion"
"diffusion"
"diffusion"
"diffusion"
"diffusion"
"diffusion"
)

CONFIGS=(
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_gan_aug_minority_heavy_c4c7_b100.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_gan_aug_minority_heavy_c4c7_b500.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_gan_aug_minority_heavy_c4c7_b2000.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_gan_aug_minority_heavy_c4c7_b100_seed43.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_gan_aug_minority_heavy_c4c7_b500_seed43.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_gan_aug_minority_heavy_c4c7_b2000_seed43.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_gan_aug_minority_heavy_c4c7_b100_seed44.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_gan_aug_minority_heavy_c4c7_b500_seed44.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_gan_aug_minority_heavy_c4c7_b2000_seed44.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_vae_aug_minority_heavy_c4c7_b100.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_vae_aug_minority_heavy_c4c7_b500.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_vae_aug_minority_heavy_c4c7_b2000.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_vae_aug_minority_heavy_c4c7_b100_seed43.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_vae_aug_minority_heavy_c4c7_b500_seed43.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_vae_aug_minority_heavy_c4c7_b2000_seed43.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_vae_aug_minority_heavy_c4c7_b100_seed44.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_vae_aug_minority_heavy_c4c7_b500_seed44.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_vae_aug_minority_heavy_c4c7_b2000_seed44.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b100.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b500.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b2000.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b100_seed43.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b500_seed43.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b2000_seed43.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b100_seed44.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b500_seed44.yaml"
"papers/paper3_when_does_synth_help/configs/regimes/paper3_regime_diffusion_aug_minority_heavy_c4c7_b2000_seed44.yaml"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "[job] MODEL=${MODEL}"
echo "[job] CONFIG=${CONFIG}"
echo "[job] ARTIFACTS=${ARTIFACTS}"

echo "[stage] Config check"
python - <<PY
import yaml
from pathlib import Path

cfg_path = Path("${CONFIG}")
cfg = yaml.safe_load(cfg_path.read_text()) or {}

rm = cfg.get("run_meta", {}) or {}
regime = rm.get("regime", {}) or {}
config_id = rm.get("config_id")
seed = cfg.get("synth", {}).get("seed") or cfg.get("seed") or rm.get("seed")
budget = regime.get("synth_budget_per_class") or cfg.get("synth", {}).get("n_per_class")
minority_classes = cfg.get("real_train_subsample", {}).get("minority_classes")
minority_fraction = cfg.get("real_train_subsample", {}).get("minority_fraction")
synth_classes = cfg.get("synth", {}).get("class_ids")

print("config:", cfg_path)
print("config_id:", config_id)
print("seed:", seed)
print("budget:", budget)
print("minority_classes:", minority_classes)
print("minority_fraction:", minority_fraction)
print("synth_classes:", synth_classes)

if "minority_heavy_c4c7" not in str(config_id):
    raise RuntimeError(f"Unexpected config_id: {config_id}")

if int(seed) not in {42, 43, 44}:
    raise RuntimeError(f"Unexpected seed: {seed}")

if int(budget) not in {100, 500, 2000}:
    raise RuntimeError(f"Unexpected budget: {budget}")

if [int(x) for x in minority_classes] != [4, 7]:
    raise RuntimeError(f"Unexpected minority_classes: {minority_classes}")

if float(minority_fraction) != 0.2:
    raise RuntimeError(f"Unexpected minority_fraction: {minority_fraction}")

if [int(x) for x in synth_classes] != [4, 7]:
    raise RuntimeError(f"Unexpected synth.class_ids: {synth_classes}")
PY

echo "[stage] Manifest check"
python - <<PY
import json, yaml
from pathlib import Path

cfg_path = Path("${CONFIG}")
cfg = yaml.safe_load(cfg_path.read_text()) or {}

rm = cfg.get("run_meta", {}) or {}
config_id = rm.get("config_id")
seed = int(cfg.get("synth", {}).get("seed") or cfg.get("seed") or rm.get("seed"))
budget = int((rm.get("regime", {}) or {}).get("synth_budget_per_class") or cfg.get("synth", {}).get("n_per_class"))
expected_num_fake = budget * 2

p = Path("${ARTIFACTS}") / "${MODEL}" / "synthetic" / config_id / f"seed{seed}" / "manifest.json"

if not p.exists():
    raise FileNotFoundError(p)

m = json.load(open(p))

print("manifest:", p)
print("class_ids:", m.get("class_ids"))
print("num_fake:", m.get("num_fake"))
print("budget_per_class:", m.get("budget_per_class"))
print("per_class_counts:", m.get("per_class_counts"))
print("paths:", len(m.get("paths", [])))

if [int(x) for x in m.get("class_ids", [])] != [4, 7]:
    raise RuntimeError(f"Expected class_ids=[4,7], got {m.get('class_ids')}")

if int(m.get("num_fake", -1)) != expected_num_fake:
    raise RuntimeError(f"Expected num_fake={expected_num_fake}, got {m.get('num_fake')}")

if len(m.get("paths", [])) != expected_num_fake:
    raise RuntimeError(f"Expected {expected_num_fake} paths, got {len(m.get('paths', []))}")

pcc = m.get("per_class_counts") or {}
if int(pcc.get("4", -1)) != budget or int(pcc.get("7", -1)) != budget:
    raise RuntimeError(f"Expected per_class_counts for 4 and 7 to equal {budget}, got {pcc}")
PY

echo "[stage] Evaluate"
python -m app.main eval --model "${MODEL}" --config "${CONFIG}" --artifacts "${ARTIFACTS}"

echo "[stage] Summary check"
python - <<PY
import json, glob, yaml
from pathlib import Path

cfg_path = Path("${CONFIG}")
cfg = yaml.safe_load(cfg_path.read_text()) or {}

rm_cfg = cfg.get("run_meta", {}) or {}
expected_cid = rm_cfg.get("config_id")
expected_seed = int(cfg.get("synth", {}).get("seed") or cfg.get("seed") or rm_cfg.get("seed"))
expected_budget = int((rm_cfg.get("regime", {}) or {}).get("synth_budget_per_class") or cfg.get("synth", {}).get("n_per_class"))

files = sorted(glob.glob("${ARTIFACTS}/${MODEL}/summaries/summary_*.json"), key=lambda p: Path(p).stat().st_mtime)
hits = []

for p in files:
    try:
        s = json.load(open(p))
    except Exception:
        continue

    rm = s.get("run_meta") or {}
    cid = s.get("config_id") or rm.get("config_id")
    seed = rm.get("seed", s.get("seed"))
    budget = (rm.get("regime") or {}).get("synth_budget_per_class") or s.get("budget_per_class") or rm.get("budget_per_class")

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
ro = s.get("utility_real_only") or {}
rs = s.get("utility_real_plus_synth") or {}
ro_pc = ro.get("per_class") or {}
rs_pc = rs.get("per_class") or {}

print("latest_summary:", p)
print("run_id:", s.get("run_id"))
print("config_id:", s.get("config_id"), (s.get("run_meta") or {}).get("config_id"))
print("seed:", (s.get("run_meta") or {}).get("seed", s.get("seed")))
print("budget:", s.get("budget_per_class"), (s.get("run_meta") or {}).get("budget_per_class"))
print("manifest_path:", s.get("manifest_path"))
print("has_real_only_per_class:", isinstance(ro_pc, dict))
print("has_real_plus_synth_per_class:", isinstance(rs_pc, dict))

if not isinstance(ro_pc, dict) or not isinstance(rs_pc, dict):
    raise RuntimeError("Missing per_class metrics in saved summary")

for c in [4, 7]:
    ro_f1 = ro_pc.get("f1", [None] * 9)[c]
    rs_f1 = rs_pc.get("f1", [None] * 9)[c]
    ro_rec = ro_pc.get("recall", [None] * 9)[c]
    rs_rec = rs_pc.get("recall", [None] * 9)[c]
    ro_prec = ro_pc.get("precision", [None] * 9)[c]
    rs_prec = rs_pc.get("precision", [None] * 9)[c]
    print(f"class_{c}_real_only_f1:", ro_f1)
    print(f"class_{c}_real_plus_synth_f1:", rs_f1)
    print(f"class_{c}_delta_f1:", None if ro_f1 is None or rs_f1 is None else rs_f1 - ro_f1)
    print(f"class_{c}_real_only_recall:", ro_rec)
    print(f"class_{c}_real_plus_synth_recall:", rs_rec)
    print(f"class_{c}_delta_recall:", None if ro_rec is None or rs_rec is None else rs_rec - ro_rec)
    print(f"class_{c}_real_only_precision:", ro_prec)
    print(f"class_{c}_real_plus_synth_precision:", rs_prec)
    print(f"class_{c}_delta_precision:", None if ro_prec is None or rs_prec is None else rs_prec - ro_prec)

print("deltas:", s.get("deltas_RS_minus_R") or (s.get("utility") or {}).get("deltas_RS_minus_R"))
PY

echo "[job] Done: $(date)"