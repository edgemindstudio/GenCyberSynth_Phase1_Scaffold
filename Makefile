# Makefile — Unified main Makefile for GenCyberSynth
# Includes:
#   - setup / smoke / onepass
#   - Phase-1 / Paper-1 freeze + gate + build
#   - JSONL / CSV / report / figure helpers
#   - Talon / Slurm helpers
#   - FID/KID parsing + plotting helpers
#
# Notes:
#   - This is the canonical Makefile.
#   - Makefile.bak and Makefile.fid can be kept as archive/reference,
#     but this file is the one you should use going forward.

SHELL := bash
.ONESHELL:
.SHELLFLAGS := -euo pipefail -c
.SILENT:

export PYTHONPATH := $(CURDIR)

.PHONY: help setup smoke smoke-all train synth eval onepass onepass-seeds onepass-all models-seeds \
        phase1_freeze phase1_scores phase1_check phase1_backfill phase1_gate \
        normalize-summaries paper1-jsonl table scores-csv grids report \
        figs-core figs-diversity figs-imbalance figs-qual figs-hparams figs-all \
        paper1_prepare paper1_build paper1 clean-summaries clean-synth demo \
        submit-array submit-array-gpu monitor lastlog tailf slurm-help all \
        fid-submit-cpu fid-parse fid-best fid-plot fid-grid-all fid-show \
        fid-best-3d fid-plot-3d fid-all fid-clean fid-top

# -----------------------------------------------------------------------------
# Globals (override on CLI)
# -----------------------------------------------------------------------------
PY                    ?= python
CFG                   ?= configs/config.yaml
SMOKE_CFG             ?= configs/config.smoke.yaml
MODELS                ?= gan diffusion vae autoregressive maskedautoflow restrictedboltzmann gaussianmixture
SMOKE_MODEL           ?= gan
SEEDS                 ?= 42 43 44
SYN_PER_CLASS         ?= 1000

# Paper-1 lock defaults
PHASE1_SUMMARY_NAME    ?= paper1.json
PHASE1_MANIFEST_NAME   ?= paper1_manifest.json
PHASE1_ALLOWED_BUDGETS ?= 2000

# Paths
SUMMARIES_DIR          ?= artifacts/summaries
OUT_JSONL              ?= $(SUMMARIES_DIR)/phase1_summaries.jsonl

# Optional schema
SCHEMA_PATH            ?= gcs-core/gcs_core/schemas/eval_summary.lite.schema.json

# FID/KID helper outputs
FID_COMBINED_CSV       ?= $(SUMMARIES_DIR)/fid_grid_combined.csv
FID_BEST_CSV           ?= $(SUMMARIES_DIR)/fid_grid_best_per_model.csv
FID_BEST_MD            ?= $(SUMMARIES_DIR)/fid_grid_best_per_model.md
FID_BEST_PNG           ?= $(SUMMARIES_DIR)/fid_best_bar.png

# Optional knobs for fid-submit-cpu
PER_CLASS_CAP          ?= 200
TOTAL_CAP              ?= 0
BATCH                  ?= 64
BACKBONE               ?= inception
IMG_SIZE               ?= 299

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	echo "GenCyberSynth unified Makefile"
	echo ""
	echo "Core:"
	echo "  make setup"
	echo "  make smoke SMOKE_MODEL=gan"
	echo "  make onepass MODEL=gan CFG=configs/config.yaml"
	echo "  make train|synth|eval CFG=configs/config.yaml"
	echo ""
	echo "Phase-1 / Paper-1:"
	echo "  make phase1_freeze"
	echo "  make phase1_scores"
	echo "  make phase1_check"
	echo "  make phase1_gate"
	echo "  make paper1_prepare"
	echo "  make paper1_build"
	echo "  make paper1"
	echo ""
	echo "Rollups:"
	echo "  make paper1-jsonl"
	echo "  make scores-csv"
	echo "  make table"
	echo "  make report"
	echo ""
	echo "Figures:"
	echo "  make figs-all"
	echo ""
	echo "FID/KID helper pipeline:"
	echo "  make fid-all"
	echo "  make fid-parse"
	echo "  make fid-best"
	echo "  make fid-best-3d"
	echo "  make fid-show"
	echo ""
	echo "Slurm/Talon:"
	echo "  make submit-array MODELS='gan diffusion vae' SEEDS='42 43 44'"
	echo "  make submit-array-gpu MODELS='gan diffusion vae' SEEDS='42 43 44'"
	echo "  make monitor"
	echo "  make lastlog"
	echo "  make tailf"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup:
	pip install -r requirements.txt
	mkdir -p "$(SUMMARIES_DIR)"

# -----------------------------------------------------------------------------
# Fast sanity
# -----------------------------------------------------------------------------
smoke:
	echo "== SMOKE $(SMOKE_MODEL) =="
	$(PY) -m app.main synth --model $(SMOKE_MODEL) --config $(SMOKE_CFG)
	$(PY) -m app.main eval  --model $(SMOKE_MODEL) --config $(SMOKE_CFG)

smoke-all:
	for m in $(MODELS); do \
	  echo "== SMOKE $$m =="; \
	  $(PY) -m app.main synth --model $$m --config $(CFG) || true; \
	  $(PY) -m app.main eval  --model $$m --config $(CFG) || true; \
	done

# -----------------------------------------------------------------------------
# Training / Synthesis / Evaluation loops
# -----------------------------------------------------------------------------
train:
	for m in $(MODELS); do \
	  echo "== TRAIN $$m =="; \
	  $(PY) -m app.main train --model $$m --config $(CFG) || true; \
	done

synth:
	for m in $(MODELS); do \
	  echo "== SYNTH $$m =="; \
	  $(PY) -m app.main synth --model $$m --config $(CFG) || true; \
	done

eval:
	for m in $(MODELS); do \
	  echo "== EVAL $$m =="; \
	  $(PY) -m app.main eval --model $$m --config $(CFG) || true; \
	done

# -----------------------------------------------------------------------------
# One-pass wrappers
# -----------------------------------------------------------------------------
# Usage: make onepass MODEL=gan
onepass:
	$(PY) -m app.main train --model $(MODEL) --config $(CFG) || true
	$(PY) -m app.main synth --model $(MODEL) --config $(CFG)
	$(PY) -m app.main eval  --model $(MODEL) --config $(CFG)

# Usage: make onepass-seeds MODEL=gan
onepass-seeds:
	@for s in $(SEEDS); do \
	  echo ">> MODEL=$(MODEL) SEED=$$s"; \
	  SEED=$$s $(MAKE) -s onepass MODEL=$(MODEL) CFG="$(CFG)"; \
	done

onepass-all:
	for m in $(MODELS); do \
	  $(MAKE) -s onepass MODEL=$$m CFG="$(CFG)"; \
	done

# Example: make models-seeds CMD='eval' CFG=configs/config.yaml
models-seeds:
	@if [ -z "$$CMD" ]; then echo "Set CMD, e.g. make models-seeds CMD='eval'"; exit 2; fi
	for m in $(MODELS); do \
	  echo "== $$m : $$CMD =="; \
	  $(PY) -m app.main $$CMD --model $$m --config $(CFG) || true; \
	done

# -----------------------------------------------------------------------------
# Phase-1 lock + sanity gate
# -----------------------------------------------------------------------------
phase1_freeze:
	PHASE1_ALLOWED_BUDGETS='$(PHASE1_ALLOWED_BUDGETS)' \
	PHASE1_SUMMARY_NAME='$(PHASE1_SUMMARY_NAME)' \
	PHASE1_MANIFEST_NAME='$(PHASE1_MANIFEST_NAME)' \
	$(PY) tools/freeze_phase1_snapshots.py

phase1_scores:
	PHASE1_SUMMARY_NAME="$(PHASE1_SUMMARY_NAME)" \
	$(PY) tools/build_phase1_scores.py

phase1_check:
	PHASE1_ALLOWED_BUDGETS="$(PHASE1_ALLOWED_BUDGETS)" \
	PHASE1_SUMMARY_NAME="$(PHASE1_SUMMARY_NAME)" \
	$(PY) tools/check_phase1_integrity.py

# Run on compute if needed
phase1_backfill:
	PHASE1_SUMMARY_NAME='$(PHASE1_SUMMARY_NAME)' \
	PHASE1_MANIFEST_NAME='$(PHASE1_MANIFEST_NAME)' \
	$(PY) scripts/backfill_kid_and_downstream.py

phase1_gate: phase1_freeze phase1_scores phase1_check
	@echo "[phase1_gate] OK"

# -----------------------------------------------------------------------------
# Normalize + JSONL
# -----------------------------------------------------------------------------
normalize-summaries:
	@echo "Normalizing summaries..."
	$(PY) scripts/normalize_summaries.py

paper1-jsonl: normalize-summaries
	@echo "Building consolidated JSONL from $(PHASE1_SUMMARY_NAME)..."
	PHASE1_SUMMARY_NAME='$(PHASE1_SUMMARY_NAME)' \
	OUT_JSONL='$(OUT_JSONL)' \
	$(PY) tools/build_paper1_jsonl.py

# -----------------------------------------------------------------------------
# Tables / CSV / report
# -----------------------------------------------------------------------------
table: paper1-jsonl
	@echo "Collecting phase-1 score table..."
	PHASE1_SUMMARY_NAME='$(PHASE1_SUMMARY_NAME)' $(PY) scripts/collect_scores.py

scores-csv: paper1-jsonl
	@echo "Exporting JSONL -> CSV..."
	$(PY) scripts/jsonl_to_csv.py

grids:
	$(PY) scripts/make_grids.py

report: table
	$(PY) scripts/phase1_report.py
	echo "Report: artifacts/phase1_report.md"

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
figs-core: paper1-jsonl
	$(PY) -m scripts.plots.core.pareto_downstream_vs_similarity
	$(PY) -m scripts.plots.core.per_class_delta_f1 || true
	$(PY) -m scripts.plots.core.per_class_delta_f1 --heatmap || true
	$(PY) -m scripts.plots.core.calibration_curves || true

figs-diversity: paper1-jsonl
	$(PY) -m scripts.plots.diversity.umap_projection || true
	$(PY) -m scripts.plots.diversity.ms_ssim_hist || true
	$(PY) -m scripts.plots.diversity.nn_distance_distrib || true

figs-imbalance: paper1-jsonl
	$(PY) -m scripts.plots.imbalance.class_counts_before_after || true
	$(PY) -m scripts.plots.imbalance.simple_stats_sanity --model=gan || true

figs-qual: paper1-jsonl
	$(PY) -m scripts.plots.qual.grids_panels || true

figs-hparams: paper1-jsonl
	$(PY) -m scripts.plots.hparams.parallel_coords \
	  --cols metrics.kid metrics.downstream.macro_f1 metrics.ms_ssim metrics.downstream.balanced_acc \
	  || true

figs-all: figs-core figs-diversity figs-imbalance figs-qual figs-hparams
	@echo "Figures -> artifacts/figures/**"

# -----------------------------------------------------------------------------
# Paper-1 / Phase-1 one-command flow
# -----------------------------------------------------------------------------
paper1_prepare: phase1_gate
	@echo "[paper1_prepare] OK"

paper1_build: paper1-jsonl scores-csv table figs-all report
	@echo "[paper1_build] OK"

paper1: paper1_prepare paper1_build
	@echo "[paper1] OK"

# -----------------------------------------------------------------------------
# FID/KID helper pipeline (merged from Makefile.fid)
# -----------------------------------------------------------------------------
fid-submit-cpu:
	@echo "Submitting CPU per-model FID/KID array..."
	@export CUDA_VISIBLE_DEVICES=""; \
	export TF_FORCE_CPU=1; \
	export TF_CPP_MIN_LOG_LEVEL=2; \
	export PER_CLASS_CAP="$(PER_CLASS_CAP)"; \
	export TOTAL_CAP="$(TOTAL_CAP)"; \
	export BATCH="$(BATCH)"; \
	export BACKBONE="$(BACKBONE)"; \
	export IMG_SIZE="$(IMG_SIZE)"; \
	sbatch -p talon-large slurm/fid_kid_per_model_cpu.sbatch

fid-parse:
	@mkdir -p "$(SUMMARIES_DIR)"
	$(PY) scripts/metrics/fid_parse.py "$(FID_COMBINED_CSV)"
	@column -t -s, "$(FID_COMBINED_CSV)" | head || true

fid-best: fid-parse
	$(PY) scripts/metrics/fid_best.py "$(FID_COMBINED_CSV)" "$(SUMMARIES_DIR)"

fid-plot: fid-best
	MPLBACKEND=Agg $(PY) scripts/metrics/fid_plot.py "$(FID_BEST_CSV)" "$(FID_BEST_PNG)"

fid-best-3d: fid-parse
	$(PY) scripts/metrics/fid_best_3d.py "$(FID_COMBINED_CSV)" "$(SUMMARIES_DIR)"

fid-plot-3d: fid-best-3d
	MPLBACKEND=Agg $(PY) scripts/metrics/fid_plot.py "$(SUMMARIES_DIR)/fid_grid_best_per_model_backbone_img.csv" "$(SUMMARIES_DIR)/fid_best_bar_backbone_img.png"

fid-grid-all: fid-plot fid-plot-3d
	@echo "Artifacts:"
	@echo "  $(FID_COMBINED_CSV)"
	@echo "  $(FID_BEST_CSV)"
	@echo "  $(FID_BEST_MD)"
	@echo "  $(FID_BEST_PNG)"
	@echo "  $(SUMMARIES_DIR)/fid_grid_best_per_model_backbone_img.csv"
	@echo "  $(SUMMARIES_DIR)/fid_grid_best_per_model_backbone_img.md"
	@echo "  $(SUMMARIES_DIR)/fid_best_bar_backbone_img.png"

fid-show: fid-parse
	@(head -n1 "$(FID_COMBINED_CSV)"; tail -n +2 "$(FID_COMBINED_CSV)" | sort -t, -k9,9g) | column -t -s, || true

fid-all: fid-parse fid-best fid-plot fid-best-3d fid-plot-3d fid-grid-all fid-show
	@echo "FID pipeline complete."

fid-clean:
	rm -f "$(SUMMARIES_DIR)/fid_grid_combined.csv" \
	      "$(SUMMARIES_DIR)/fid_grid_best_per_model.csv" \
	      "$(SUMMARIES_DIR)/fid_grid_best_per_model.md" \
	      "$(SUMMARIES_DIR)/fid_grid_best_per_model_backbone_img.csv" \
	      "$(SUMMARIES_DIR)/fid_grid_best_per_model_backbone_img.md" \
	      "$(SUMMARIES_DIR)/fid_best_bar.png" \
	      "$(SUMMARIES_DIR)/fid_best_bar_backbone_img.png" || true
	echo "Cleaned FID artifacts."

fid-top: fid-parse
	@(head -n1 "$(FID_COMBINED_CSV)"; tail -n +2 "$(FID_COMBINED_CSV)" | sort -t, -k9,9g | head -n 15) | column -t -s, || true

# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
clean-summaries:
	rm -f "$(OUT_JSONL)" || true
	rm -f artifacts/phase1_scores.csv || true
	rm -f artifacts/summaries/phase1_summaries.jsonl || true
	rm -f artifacts/*/summaries/paper1.json || true
	find artifacts -type f -path "artifacts/*/summaries/summary_*.json" -delete || true
	rm -f artifacts/*/summaries/latest.json || true
	echo "Cleaned summaries."

clean-synth:
	find artifacts -type f -path "artifacts/*/synthetic/**/*.png" -delete || true
	find artifacts -type f -path "artifacts/*/synthetic/*manifest*.json" -delete || true
	echo "Cleaned synthetic artifacts."

# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------
demo:
	$(PY) demo/app.py

# -----------------------------------------------------------------------------
# Talon / Slurm helpers
# -----------------------------------------------------------------------------
submit-array:
	@echo "Submitting CPU matrix job..."
	@sbatch --export=ALL,MODELS="$(MODELS)",SEEDS="$(SEEDS)",SYN_PER_CLASS="$(SYN_PER_CLASS)",REPO_DIR="$$(pwd)" \
		slurm/models_seeds_matrix_cpu.slurm

submit-array-gpu:
	@echo "Submitting GPU matrix job..."
	@sbatch --export=ALL,MODELS="$(MODELS)",SEEDS="$(SEEDS)",SYN_PER_CLASS="$(SYN_PER_CLASS)",REPO_DIR="$$(pwd)" \
		slurm/models_seeds_matrix_gpu.slurm

monitor:
	@watch -n 2 'squeue -u $$USER'

lastlog:
	@ls -t slurm/*.out 2>/dev/null | head -1 || ls -t *.out 2>/dev/null | head -1 || echo "no .out yet"

tailf:
	@f=$$(ls -t slurm/*.out 2>/dev/null | head -1 || ls -t *.out 2>/dev/null | head -1); \
	if [ -n "$$f" ]; then echo "Tailing $$f ..."; tail -n 200 -f "$$f"; else echo "no .out yet"; fi

slurm-help:
	echo "# Submit CPU matrix:"
	echo "make submit-array MODELS='gan diffusion vae' SEEDS='42 43 44' SYN_PER_CLASS=1000"
	echo "# Submit GPU matrix:"
	echo "make submit-array-gpu MODELS='gan diffusion vae' SEEDS='42 43 44' SYN_PER_CLASS=1000"
	echo ""
	echo "# Canonical Paper-1 / Phase-1 eval-only rerun:"
	echo "export CFG=configs/paper1_final_rerun.yaml"
	echo "export ARTS=/home/bruno.fonkeng/gencys/artifacts_paper1_final_rerun"
	echo "export DO_TRAIN=0"
	echo "export DO_SYNTH=0"
	echo "export DO_EVAL=1"
	echo "sbatch --array=0-20 slurm/run_paper1.slurm"

# -----------------------------------------------------------------------------
# Everything
# -----------------------------------------------------------------------------
all: setup synth eval grids table report
	echo "All done."