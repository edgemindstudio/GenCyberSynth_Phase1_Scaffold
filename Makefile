# Makefile — Phase 2 convenience targets (CI-safe)

SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.SILENT:

.PHONY: help setup smoke smoke-all train synth eval table grids report clean-summaries clean-synth all \
        onepass onepass-seeds onepass-all models-seeds slurm-help

# -------- Globals (override on CLI) ------------------------------------------
PY               ?= python
CFG              ?= configs/config.yaml
MODELS           ?= gan diffusion vae autoregressive maskedautoflow restrictedboltzmann gaussianmixture
SMOKE_MODEL      ?= gan
SEEDS            ?= 42 43 44
SYN_PER_CLASS    ?= 1000

# Paths (override as needed)
REAL_ROOT        ?= USTC-TFC2016_malware/real
SYN_BASE         ?= artifacts/synthetic
SUMMARIES_DIR    ?= artifacts/summaries
OUT_JSONL        ?= $(SUMMARIES_DIR)/phase1_summaries.jsonl
ENCODER          ?= artifacts/domain_encoder.pt

# -------- Help ---------------------------------------------------------------
help:
	echo "Targets:"
	echo "  setup             - pip install -r requirements.txt"
	echo "  smoke             - 1-model quick synth+eval (SMOKE_MODEL=$(SMOKE_MODEL))"
	echo "  smoke-all         - quick synth+eval on ALL MODELS"
	echo "  train             - iterate train over MODELS (uses SEEDS)"
	echo "  synth             - generate for MODELS (uses SYN_PER_CLASS, SEEDS)"
	echo "  eval              - evaluate for MODELS (uses SEEDS)"
	echo "  onepass           - train→synth→eval for one MODEL/SEED"
	echo "  onepass-seeds     - onepass over SEEDS for a single MODEL"
	echo "  onepass-all       - onepass-seeds across all MODELS"
	echo "  models-seeds      - run arbitrary app.main CMD over MODELS×SEEDS"
	echo "  grids             - build preview grids"
	echo "  table             - aggregate summaries → artifacts/phase1_scores.csv"
	echo "  report            - write artifacts/phase1_report.md"
	echo "  clean-summaries   - remove JSON summaries"
	echo "  clean-synth       - remove synthetic images & manifests"
	echo "  slurm-help        - print Slurm example"
	echo "  all               - setup → synth → eval → grids → table → report"

# -------- Setup --------------------------------------------------------------
setup:
	pip install -r requirements.txt
	mkdir -p "$(SUMMARIES_DIR)"

# -------- Fast CI sanity (single model) --------------------------------------
smoke:
	echo "== SMOKE $(SMOKE_MODEL) =="
	$(PY) -m app.main synth \
	  --model $(SMOKE_MODEL) --config $(CFG) \
	  --seed 42 --synth-per-class 64 \
	  --real-root $(REAL_ROOT) \
	  --synth-root $(SYN_BASE)/$(SMOKE_MODEL)/seed42
	$(PY) -m app.main eval \
	  --model $(SMOKE_MODEL) --config $(CFG) \
	  --seed 42 --real-root $(REAL_ROOT) \
	  --synth-root $(SYN_BASE)/$(SMOKE_MODEL)/seed42 \
	  --encoder $(ENCODER) --out $(OUT_JSONL)

# -------- Smoke for all models ----------------------------------------------
smoke-all:
	for m in $(MODELS); do \
	  echo "== SMOKE $$m =="; \
	  $(PY) -m app.main synth --model $$m --config $(CFG) --seed 42 \
	    --synth-per-class 64 --real-root $(REAL_ROOT) \
	    --synth-root $(SYN_BASE)/$$m/seed42 || true; \
	  $(PY) -m app.main eval --model $$m --config $(CFG) --seed 42 \
	    --real-root $(REAL_ROOT) \
	    --synth-root $(SYN_BASE)/$$m/seed42 \
	    --encoder $(ENCODER) --out $(OUT_JSONL) || true; \
	done

# -------- Training / Synthesis / Evaluation loops ---------------------------
train:
	for m in $(MODELS); do \
	  for s in $(SEEDS); do \
	    echo "== TRAIN $$m seed=$$s =="; \
	    $(PY) -m app.main train --model $$m --config $(CFG) --seed $$s || true; \
	  done; \
	done

synth:
	for m in $(MODELS); do \
	  for s in $(SEEDS); do \
	    echo "== SYNTH $$m seed=$$s =="; \
	    $(PY) -m app.main synth --model $$m --config $(CFG) \
	      --seed $$s --synth-per-class $(SYN_PER_CLASS) \
	      --real-root $(REAL_ROOT) \
	      --synth-root $(SYN_BASE)/$$m/seed$$s || true; \
	  done; \
	done

eval:
	for m in $(MODELS); do \
	  for s in $(SEEDS); do \
	    echo "== EVAL $$m seed=$$s =="; \
	    $(PY) -m app.main eval --model $$m --config $(CFG) \
	      --seed $$s --real-root $(REAL_ROOT) \
	      --synth-root $(SYN_BASE)/$$m/seed$$s \
	      --encoder $(ENCODER) --out $(OUT_JSONL) || true; \
	  done; \
	done

# -------- Convenient one-pass wrappers --------------------------------------
# Usage:
#   make onepass MODEL=gan SEED=42 SYN_PER_CLASS=1000
onepass:
	$(PY) -m app.main train --model $(MODEL) --config $(CFG) --seed $(SEED) || true
	$(PY) -m app.main synth --model $(MODEL) --config $(CFG) \
	  --seed $(SEED) --synth-per-class $(SYN_PER_CLASS) \
	  --real-root $(REAL_ROOT) \
	  --synth-root $(SYN_BASE)/$(MODEL)/seed$(SEED)
	$(PY) -m app.main eval --model $(MODEL) --config $(CFG) \
	  --seed $(SEED) --real-root $(REAL_ROOT) \
	  --synth-root $(SYN_BASE)/$(MODEL)/seed$(SEED) \
	  --encoder $(ENCODER) --out $(OUT_JSONL)

# Run onepass for a single model across SEEDS
# Usage: make onepass-seeds MODEL=gan
onepass-seeds:
	for s in $(SEEDS); do \
	  $(MAKE) onepass MODEL=$(MODEL) SEED=$$s; \
	done

# Run onepass-seeds across all MODELS
onepass-all:
	for m in $(MODELS); do \
	  $(MAKE) onepass-seeds MODEL=$$m; \
	done

# Low-level helper to run an arbitrary app.main subcommand over MODELS×SEEDS
# Usage: make models-seeds CMD="eval --out $(OUT_JSONL)"
models-seeds:
	@if [ -z "$$CMD" ]; then echo "Set CMD, e.g., make models-seeds CMD='eval --out $(OUT_JSONL)'"; exit 2; fi
	for m in $(MODELS); do \
	  for s in $(SEEDS); do \
	    echo "== $$m seed=$$s : $$CMD =="; \
	    $(PY) -m app.main $$CMD --model $$m --config $(CFG) --seed $$s || true; \
	  done; \
	done

# -------- Tables / Grids / Report -------------------------------------------
table:
	$(PY) scripts/collect_scores.py

grids:
	$(PY) scripts/make_grids.py

report:
	$(PY) scripts/phase1_report.py
	echo "Report: artifacts/phase1_report.md"

# -------- Cleaning -----------------------------------------------------------
clean-summaries:
	rm -f $(OUT_JSONL) || true
	find $(SUMMARIES_DIR) -type f -name 'summary_*.json' -delete || true
	rm -f artifacts/*/summaries/latest.json || true
	echo "Cleaned summaries."

clean-synth:
	find $(SYN_BASE) -type f -path "$(SYN_BASE)/*/seed*/*" -delete || true
	echo "Cleaned synthetic artifacts."

# -------- Slurm example (print-only) ----------------------------------------
slurm-help:
	echo "# Example array: seeds 42..44 on one model"
	echo "sbatch --array=42-44 model-template/scripts/slurm_array_example.sh \\"
	echo "  -- make onepass MODEL=gan SEED=\$$SLURM_ARRAY_TASK_ID SYN_PER_CLASS=$(SYN_PER_CLASS)"
	echo "# Or all models (per-seed)"
	echo "sbatch --array=42-44 --wrap 'make onepass-all SYN_PER_CLASS=$(SYN_PER_CLASS)'"

# -------- Everything ---------------------------------------------------------
all: setup synth eval grids table report
