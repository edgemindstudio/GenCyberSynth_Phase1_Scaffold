#!/bin/bash
#SBATCH --job-name=paper3
#SBATCH --partition=talon
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=papers/paper3_when_does_synth_help/logs/%x.%A_%a.out
#SBATCH --error=papers/paper3_when_does_synth_help/logs/%x.%A_%a.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

: "${CFG:?CFG must be set}"
: "${ARTS:=/home/bruno.fonkeng/gencys/artifacts_paper3}"
: "${MODEL:=gan}"
: "${DO_TRAIN:=1}"
: "${DO_SYNTH:=1}"
: "${DO_EVAL:=1}"

echo "CFG=$CFG"
echo "ARTS=$ARTS"
echo "MODEL=$MODEL"
echo "DO_TRAIN=$DO_TRAIN DO_SYNTH=$DO_SYNTH DO_EVAL=$DO_EVAL"
echo "git=$(git rev-parse HEAD || true)"

if [ "${DO_TRAIN}" = "1" ]; then
  python -m app.main train --model "${MODEL}" --config "${CFG}" --artifacts "${ARTS}"
fi

if [ "${DO_SYNTH}" = "1" ]; then
  python -m app.main synth --model "${MODEL}" --config "${CFG}" --artifacts "${ARTS}"
fi

if [ "${DO_EVAL}" = "1" ]; then
  python -m app.main eval --model "${MODEL}" --config "${CFG}" --artifacts "${ARTS}"
fi

echo "done: $(date)"