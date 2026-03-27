#!/usr/bin/env bash
#SBATCH -J gcs-paper1
#SBATCH -o slurm-paper1.%A_%a.out
#SBATCH -e slurm-paper1.%A_%a.err

#SBATCH -p talon-gpu32
#SBATCH --gres=gpu:1

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 27-00:00:00
#SBATCH --array=0-20%2

set -euo pipefail

# -------------------------
# Environment
# -------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- XLA / JIT safety: point TF to CUDA toolkit for ptxas + libdevice ---
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda12.8/toolkit/12.8.1}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# -------------------------
# Modules (GPU runtime)
# -------------------------
module purge
module --ignore_cache load shared
module --ignore_cache load slurm/slurm/25.05.2
module --ignore_cache load cuda12.8/toolkit/12.8.1
module --ignore_cache load cudnn9.7-cuda12.8/9.7.1.26
module list


cd "$SLURM_SUBMIT_DIR"
echo "[slurm] job_id=$SLURM_JOB_ID task_id=$SLURM_ARRAY_TASK_ID host=$(hostname) pwd=$(pwd)"
python -V

# -------------------------
# Task mapping
# -------------------------
MODELS=(gan vae diffusion autoregressive restrictedboltzmann gaussianmixture maskedautoflow)
SEEDS=(42 43 44)

N_MODELS=${#MODELS[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL=$((N_MODELS * N_SEEDS))

TASK=${SLURM_ARRAY_TASK_ID}
if (( TASK < 0 || TASK >= TOTAL )); then
  echo "[fatal] TASK out of range: $TASK (TOTAL=$TOTAL)"
  exit 2
fi

seed_idx=$(( TASK % N_SEEDS ))
model_idx=$(( TASK / N_SEEDS ))

MODEL="${MODELS[$model_idx]}"
SEED="${SEEDS[$seed_idx]}"

# -------------------------
# Config + artifacts
# -------------------------
CFG="${CFG:-configs/paper1_config.yaml}"
ARTS="${ARTS:-/home/bruno.fonkeng/gencys/artifacts_paper1_clean}"

export CFG
export ARTS

# guardrails
[[ -n "${MODEL:-}" ]] || { echo "[fatal] MODEL empty"; exit 9; }
[[ -n "${SEED:-}"  ]] || { echo "[fatal] SEED empty";  exit 9; }
[[ -f "$CFG"       ]] || { echo "[fatal] missing CFG: $CFG"; exit 9; }
[[ -n "${ARTS:-}"  ]] || { echo "[fatal] ARTS empty"; exit 9; }

# Verify cfg.paths.artifacts matches env ARTS (prevents accidental writes)
python - <<'PY'
import os, yaml, sys
cfg=yaml.safe_load(open(os.environ["CFG"])) or {}
want = (cfg.get("paths") or {}).get("artifacts", None)
have = os.environ["ARTS"]
print("[check] cfg.paths.artifacts =", want)
print("[check] ARTS env           =", have)
if want is None:
    print("[fatal] config missing paths.artifacts")
    sys.exit(9)
if want != have:
    print("[fatal] ARTS does not match cfg.paths.artifacts")
    sys.exit(9)
print("[OK] ARTS matches config")
PY

echo "[map] MODEL=$MODEL SEED=$SEED"
echo "[cfg] CFG=$CFG"
echo "[cfg] ARTS=$ARTS"

# -------------------------
# Per-task config path
# -------------------------
CFG_TASK="/tmp/paper1_${MODEL}_seed${SEED}_${SLURM_JOB_ID}_${TASK}.yaml"
export CFG_TASK
export MODEL
export PAPER1_SEED="$SEED"
export PAPER1_CFG="$CFG"
export PAPER1_CFG_TASK="$CFG_TASK"

# -------------------------
# Build per-task config:
# - set SEED
# - map synth.n_per_class -> SAMPLES_PER_CLASS (legacy)
# - optional diffusion default steps
# - optional SMOKE overrides
# - provenance stamp required by evaluator
# -------------------------
python - <<'PY'
import os, yaml, hashlib, subprocess

cfg = yaml.safe_load(open(os.environ["PAPER1_CFG"])) or {}
model = os.environ["MODEL"]
seed  = int(os.environ["PAPER1_SEED"])
out_path = os.environ["PAPER1_CFG_TASK"]

cfg["SEED"] = seed

# Map unified knob -> legacy knob used by multiple adapters/pipelines
if isinstance(cfg.get("synth"), dict) and "n_per_class" in cfg["synth"]:
    cfg["SAMPLES_PER_CLASS"] = int(cfg["synth"]["n_per_class"])

# Keep diffusion sampling practical on CPU unless explicitly set in YAML or env
# (Note: diffusion/sample.py reads DIFFUSION_STEPS as the reverse loop step count.)
if model == "diffusion" and "DIFFUSION_STEPS" not in cfg:
    cfg["DIFFUSION_STEPS"] = int(os.environ.get("DIFFUSION_STEPS", "50"))

# -------------------------
# SMOKE overrides (optional)
# -------------------------
SMOKE = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    smoke_epochs = int(os.environ.get("SMOKE_EPOCHS", "2"))
    smoke_bs     = int(os.environ.get("SMOKE_BS", "64"))
    smoke_synth  = int(os.environ.get("SMOKE_SYNTH_PER_CLASS", "10"))
    smoke_steps  = int(os.environ.get("SMOKE_DIFFUSION_STEPS", "20"))

    # For trainers that read train.*
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = smoke_epochs
    cfg["train"]["batch_size"] = smoke_bs

    # Some model trainers read legacy top-level keys
    # (safe for gan/vae/diffusion; others may ignore or have their own keys)
    if model in ("gan", "vae", "diffusion"):
        cfg["EPOCHS"] = smoke_epochs
        cfg["BATCH_SIZE"] = smoke_bs

    # Universal legacy synth knob used widely
    cfg["SAMPLES_PER_CLASS"] = smoke_synth

    # Diffusion sampling steps
    if model == "diffusion":
        cfg["DIFFUSION_STEPS"] = smoke_steps

    print(f"[smoke] epochs={smoke_epochs} bs={smoke_bs} "
          f"samples_per_class={smoke_synth} diffusion_steps={cfg.get('DIFFUSION_STEPS', None)}")

# -------------------------
# Provenance required by eval
# -------------------------
rm = cfg.get("run_meta") if isinstance(cfg.get("run_meta"), dict) else {}
rm.update({"model": model, "seed": seed, "config_path": out_path})

tmp = yaml.safe_dump(cfg, sort_keys=False)
rm["config_sha1"] = hashlib.sha1(tmp.encode("utf-8")).hexdigest()

try:
    rm["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    rm["git_commit"] = None

cfg["run_meta"] = rm

# Helpful debug line in every task output
print("[cfg_effective] "
      f"model={model} seed={seed} "
      f"SAMPLES_PER_CLASS={cfg.get('SAMPLES_PER_CLASS')} "
      f"DIFFUSION_STEPS={cfg.get('DIFFUSION_STEPS', None)} "
      f"train.epochs={(cfg.get('train') or {}).get('epochs', None)} "
      f"train.batch_size={(cfg.get('train') or {}).get('batch_size', None)}")

with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("[cfg_task] wrote", out_path)
print("[cfg_task] sha1 ", rm["config_sha1"])
print("[cfg_task] git  ", rm["git_commit"])
PY

python - <<'PY'
import tensorflow as tf
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

# -------------------------
# Stage toggles
# -------------------------
DO_TRAIN="${DO_TRAIN:-1}"
DO_SYNTH="${DO_SYNTH:-1}"
DO_EVAL="${DO_EVAL:-1}"
echo "[toggles] DO_TRAIN=$DO_TRAIN DO_SYNTH=$DO_SYNTH DO_EVAL=$DO_EVAL"


# -------------------------
# Prevent per-model synth/eval file clobbering
# -------------------------
LOCKDIR="$ARTS/.locks"
mkdir -p "$LOCKDIR"
exec 9>"$LOCKDIR/${MODEL}.lock"
flock 9
echo "[lock] acquired for MODEL=$MODEL"

# -------------------------
# Run stages
# -------------------------
if [[ "$DO_TRAIN" == "1" ]]; then
  echo "[stage] train"
  python -u -m app.main train --model "$MODEL" --config "$CFG_TASK" --artifacts "$ARTS"
fi

if [[ "$DO_SYNTH" == "1" ]]; then
  echo "[stage] synth"
  python -u -m app.main synth --model "$MODEL" --config "$CFG_TASK" --artifacts "$ARTS"
fi

if [[ "$DO_EVAL" == "1" ]]; then
  echo "[stage] eval"
  python -u -m app.main eval --model "$MODEL" --config "$CFG_TASK" --artifacts "$ARTS"
fi

echo "[lock] releasing for MODEL=$MODEL"
echo "[done] MODEL=$MODEL SEED=$SEED CFG_TASK=$CFG_TASK"