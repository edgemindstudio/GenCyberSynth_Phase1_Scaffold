#!/usr/bin/env bash
#SBATCH -J gcs-tuning
#SBATCH -o slurm-tuning.%A_%a.out
#SBATCH -e slurm-tuning.%A_%a.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 27-00:00:00
#SBATCH --array=0-41%6
#SBATCH --requeue
#SBATCH --signal=B:USR1@1800

set -euo pipefail

# -----------------------
# Environment + repo root
# -----------------------
module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

trap 'echo "[signal] received USR1 (about to time out); consider saving checkpoints";' USR1

echo "[slurm] job_id=$SLURM_JOB_ID task_id=$SLURM_ARRAY_TASK_ID submit_dir=$SLURM_SUBMIT_DIR host=$(hostname)"
cd "$SLURM_SUBMIT_DIR" || { echo "[fatal] Cannot cd to submit dir"; exit 2; }

export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
export TF_XLA_FLAGS=--tf_xla_auto_jit=0


# -----------------------
# Helpers
# -----------------------
run_stage () {
  local name="$1"; shift
  echo "[stage] $name :: $*"
  set +e
  "$@"
  local rc=$?
  set -e
  echo "[rc] $name rc=$rc"
  return $rc
}

# -----------------------
# Toggle stages (MUST be defined before use)
# -----------------------
DO_TRAIN="${DO_TRAIN:-1}"
DO_SYNTH="${DO_SYNTH:-1}"
DO_EVAL="${DO_EVAL:-1}"

# -----------------------
# Experiment definition
# -----------------------
MODELS=(gan vae diffusion autoregressive restrictedboltzmann gaussianmixture maskedautoflow)
CONFIGS=(A B)
SEEDS=(42 43 44)

N_MODELS=${#MODELS[@]}
N_CONFIGS=${#CONFIGS[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL=$((N_MODELS * N_CONFIGS * N_SEEDS))

TASK=${SLURM_ARRAY_TASK_ID}
if (( TASK < 0 || TASK >= TOTAL )); then
  echo "[fatal] TASK out of range: $TASK (TOTAL=$TOTAL)"
  exit 3
fi

seed_idx=$(( TASK % N_SEEDS ))
tmp=$(( TASK / N_SEEDS ))
cfg_idx=$(( tmp % N_CONFIGS ))
model_idx=$(( tmp / N_CONFIGS ))

MODEL="${MODELS[$model_idx]}"
CFG_LETTER="${CONFIGS[$cfg_idx]}"
SEED="${SEEDS[$seed_idx]}"

BASE_CFG="${CFG:-configs/config.yaml}"
OVR="configs/tuning/${MODEL}_${CFG_LETTER}.yaml"

ARTS="${ARTS:-/home/bruno.fonkeng/gencys/artifacts}"

echo "[map] TASK=$TASK => MODEL=$MODEL CFG=${MODEL}_${CFG_LETTER} SEED=$SEED"
echo "[paths] BASE_CFG=$BASE_CFG"
echo "[paths] OVR=$OVR"
echo "[paths] ARTS=$ARTS"
echo "[toggles] DO_TRAIN=$DO_TRAIN DO_SYNTH=$DO_SYNTH DO_EVAL=$DO_EVAL"
python -V

test -f "$BASE_CFG" || { echo "[fatal] missing base config: $BASE_CFG"; exit 4; }
test -f "$OVR" || { echo "[fatal] missing override: $OVR"; exit 5; }

# -----------------------
# Build per-task merged config
# -----------------------
CFG_TASK="/tmp/gcs_cfg_${SLURM_JOB_ID}_${TASK}.yaml"

BASE_CFG_ABS="$(realpath "$BASE_CFG")"
OVR_ABS="$(realpath "$OVR")"

export CFG_TASK BASE_CFG_ABS OVR_ABS
export MODEL CFG_LETTER SEED
export SLURM_SUBMIT_DIR SLURM_JOB_ID SLURM_ARRAY_TASK_ID

python - <<'PY'
import os, yaml, hashlib, subprocess

def deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

base_path = os.environ["BASE_CFG_ABS"]
ovr_path  = os.environ["OVR_ABS"]
out_path  = os.environ["CFG_TASK"]

model = os.environ["MODEL"]
cfg_letter = os.environ["CFG_LETTER"]
seed = int(os.environ["SEED"])
job_id = os.environ.get("SLURM_JOB_ID")
task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
repo_root = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())

with open(base_path, "r") as f:
    base = yaml.safe_load(f) or {}
with open(ovr_path, "r") as f:
    ovr = yaml.safe_load(f) or {}

deep_update(base, ovr)
base["SEED"] = seed

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    git_commit = None

rm = base.get("run_meta") if isinstance(base.get("run_meta"), dict) else {}
rm.update({
    "job_id": job_id,
    "task_id": int(task_id) if task_id is not None else None,
    "model": model,
    "config_variant": cfg_letter,
    "config_path": os.path.abspath(out_path),
    "git_commit": git_commit,
    "base_cfg_abs": os.path.abspath(base_path),
    "override_cfg_abs": os.path.abspath(ovr_path),
})
base["run_meta"] = rm

yaml_str = yaml.safe_dump(base, sort_keys=False)
rm["config_sha1"] = hashlib.sha1(yaml_str.encode("utf-8")).hexdigest()
base["run_meta"] = rm

with open(out_path, "w") as f:
    f.write(yaml_str)

print("[cfg] wrote", out_path)
print("[cfg] sha1 ", rm["config_sha1"])
print("[cfg] git  ", git_commit)
PY

echo "[debug] --- merged cfg head ---"
sed -n '1,160p' "$CFG_TASK"

# -----------------------
# Paths for guards (SAFE now: MODEL/SEED/CFG_TASK exist)
# -----------------------
SYNTH_DIR="$ARTS/$MODEL/synthetic/${MODEL}_${CFG_LETTER}_seed${SEED}"
MANIFEST="$SYNTH_DIR/manifest.json"

SUMMARY_DIR="$ARTS/$MODEL/summaries"
SUMMARY_DONE_FLAG="$SUMMARY_DIR/done_${MODEL}_${CFG_LETTER}_seed${SEED}.txt"

# -----------------------
# Phase A — Train only (long)
# -----------------------
if [[ "$DO_TRAIN" == "1" ]]; then
  run_stage "train" python -u -m app.main train --model "$MODEL" --config "$CFG_TASK" --artifacts "$ARTS" || {
    echo "[warn] train($MODEL) failed; continuing."
  }
fi

# -----------------------
# Phase B — Synth only (usually shorter)
# -----------------------
if [[ "$DO_SYNTH" == "1" ]]; then
  if [[ -f "$MANIFEST" ]]; then
    echo "[skip] synth :: manifest exists at $MANIFEST"
  else
    mkdir -p "$SYNTH_DIR"
    run_stage "synth" python -u -m app.main synth --model "$MODEL" --config "$CFG_TASK" --artifacts "$ARTS" || {
      echo "[fatal] synth($MODEL) failed; stopping task."
      exit 1
    }
  fi
fi


# -----------------------
# Phase C — Eval only (short/medium)
# -----------------------
if [[ "$DO_EVAL" == "1" ]]; then
  if [[ -f "$SUMMARY_DONE_FLAG" ]]; then
    echo "[skip] eval :: done flag exists at $SUMMARY_DONE_FLAG"
  else
    run_stage "eval" python -u -m eval.runner --config "$CFG_TASK" --model "$MODEL" || {
      echo "[fatal] eval($MODEL) failed; stopping task."
      exit 1
    }
    mkdir -p "$SUMMARY_DIR"
    echo "done $(date -Iseconds) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID" > "$SUMMARY_DONE_FLAG"
  fi
fi

echo "[done] MODEL=$MODEL CFG=${MODEL}_${CFG_LETTER} SEED=$SEED"
