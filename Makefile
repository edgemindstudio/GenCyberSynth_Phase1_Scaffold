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
SEEDS            ?= 42 43 44            # kept for future CLI expansion
SYN_PER_CLASS    ?= 1000                # kept for future CLI expansion

# Paths (override as needed)
REAL_ROOT        ?= USTC-TFC2016_malware/real   # kept for future CLI expansion
SYN_BASE         ?= artifacts/synthetic         # kept for future CLI expansion
SUMMARIES_DIR    ?= artifacts/summaries
OUT_JSONL        ?= $(SUMMARIES_DIR)/phase1_summaries.jsonl
ENCODER          ?= artifacts/domain_encoder.pt # kept for future CLI expansion

# -------- Help ---------------------------------------------------------------
help:
	echo "Targets:"
	echo "  setup             - pip install -r requirements.txt"
	echo "  smoke             - 1-model quick synth+eval (SMOKE_MODEL=$(SMOKE_MODEL))"
	echo "  smoke-all         - quick synth+eval on ALL MODELS"
	echo "  train             - iterate train over MODELS"
	echo "  synth             - generate for MODELS"
	echo "  eval              - evaluate for MODELS"
	echo "  onepass           - train→synth→eval for one MODEL (use: make onepass MODEL=gan)"
	echo "  onepass-seeds     - alias of onepass (seeds reserved for future CLI flags)"
	echo "  onepass-all       - onepass across all MODELS"
	echo "  models-seeds      - run custom subcommand over MODELS (reserved for future flags)"
	echo "  grids             - build preview grids"
	echo "  table             - aggregate summaries → artifacts/phase1_scores.csv"
	echo "  report            - write artifacts/phase1_report.md"
	echo "  clean-summaries   - remove JSON summaries"
	echo "  clean-synth       - remove synthetic images & manifests"
	echo "  slurm-help        - print Slurm example (uses config-only commands)"
	echo "  all               - setup → synth → eval → grids → table → report"

# -------- Setup --------------------------------------------------------------
setup:
	pip install -r requirements.txt
	mkdir -p "$(SUMMARIES_DIR)"

# -------- Fast CI sanity (single model) --------------------------------------
smoke:
	echo "== SMOKE $(SMOKE_MODEL) =="
	$(PY) -m app.main synth --model $(SMOKE_MODEL) --config $(CFG)
	$(PY) -m app.main eval  --model $(SMOKE_MODEL) --config $(CFG)

# -------- Smoke for all models ----------------------------------------------
smoke-all:
	for m in $(MODELS); do \
	  echo "== SMOKE $$m =="; \
	  $(PY) -m app.main synth --model $$m --config $(CFG) || true; \
	  $(PY) -m app.main eval  --model $$m --config $(CFG) || true; \
	done

# -------- Training / Synthesis / Evaluation loops ---------------------------
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

# -------- Convenient one-pass wrappers --------------------------------------
# Usage:
#   make onepass MODEL=gan
onepass:
	$(PY) -m app.main train --model $(MODEL) --config $(CFG) || true
	$(PY) -m app.main synth --model $(MODEL) --config $(CFG)
	$(PY) -m app.main eval  --model $(MODEL) --config $(CFG)

# Seeds kept for later; currently same as onepass
onepass-seeds:
	$(MAKE) onepass MODEL=$(MODEL)

onepass-all:
	for m in $(MODELS); do \
	  $(MAKE) onepass MODEL=$$m; \
	done

# Low-level helper: run a custom subcommand (without extra flags for now)
# Usage example (current CLI): make models-seeds CMD="eval"
models-seeds:
	@if [ -z "$$CMD" ]; then echo "Set CMD, e.g., make models-seeds CMD='eval'"; exit 2; fi
	for m in $(MODELS); do \
	  echo "== $$m : $$CMD =="; \
	  $(PY) -m app.main $$CMD --model $$m --config $(CFG) || true; \
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
	echo "# Example array (config-only commands):"
	echo "sbatch --array=1-3 model-template/scripts/slurm_array_example.sh \\"
	echo "  -- make onepass MODEL=gan"

# -------- Everything ---------------------------------------------------------
all: setup synth eval grids table report
	echo "All done."