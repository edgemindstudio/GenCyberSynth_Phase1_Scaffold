# Makefile — Phase 2 convenience targets (CI-safe, Talon-ready)

SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.SILENT:

.PHONY: help setup smoke smoke-all train synth eval table grids report clean-summaries clean-synth all \
        onepass onepass-seeds onepass-all models-seeds slurm-help summaries-jsonl scores-csv \
        demo submit-array submit-array-gpu monitor lastlog tailf aggregate container sif

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

# Optional schema (used only if present)
SCHEMA_PATH      ?= gcs-core/gcs_core/schemas/eval_summary.lite.schema.json

# -------- Help ---------------------------------------------------------------
help:
	echo "Targets:"
	echo "  setup             - pip install -r requirements.txt"
	echo "  smoke             - 1-model quick synth+eval (SMOKE_MODEL=$(SMOKE_MODEL))"
	echo "  smoke-all         - quick synth+eval on ALL MODELS"
	echo "  train/synth/eval  - loops over MODELS"
	echo "  onepass           - train→synth→eval for one MODEL (use: make onepass MODEL=gan)"
	echo "  onepass-seeds     - onepass over SEEDS for one MODEL (use: make onepass-seeds MODEL=gan)"
	echo "  onepass-all       - onepass across all MODELS"
	echo "  models-seeds      - run custom CMD over MODELS (e.g., make models-seeds CMD='eval')"
	echo "  grids             - build preview grids"
	echo "  table             - legacy aggregate (collect_scores.py)"
	echo "  scores-csv        - JSONL → tiny CSV (artifacts/phase1_scores.csv)"
	echo "  summaries-jsonl   - consolidate per-model JSON → JSONL (schema optional)"
	echo "  clean-summaries   - remove JSON summaries and JSONL"
	echo "  clean-synth       - remove synthetic artifacts"
	echo "  demo              - run local Gradio viewer (demo/app.py)"
	echo "  submit-array      - submit CPU matrix to Talon (slurm/models_seeds_matrix_cpu.slurm)"
	echo "  submit-array-gpu  - submit GPU matrix to Talon (slurm/models_seeds_matrix_gpu.slurm)"
	echo "  monitor|lastlog|tailf - Talon helpers"
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
# Usage: make onepass MODEL=gan
onepass:
	$(PY) -m app.main train --model $(MODEL) --config $(CFG) || true
	$(PY) -m app.main synth --model $(MODEL) --config $(CFG)
	$(PY) -m app.main eval  --model $(MODEL) --config $(CFG)

# Usage: make onepass-seeds MODEL=gan
onepass-seeds:
	@for s in $(SEEDS); do \
	  echo ">> MODEL=$(MODEL) SEED=$$s (note: seed is currently handled inside app/main.py if supported)"; \
	  $(MAKE) -s onepass MODEL=$(MODEL); \
	done

onepass-all:
	for m in $(MODELS); do \
	  $(MAKE) -s onepass MODEL=$$m; \
	done

# Run a custom subcommand over MODELS (current CLI only; flags reserved)
# Example: make models-seeds CMD="eval"
models-seeds:
	@if [ -z "$$CMD" ]; then echo "Set CMD, e.g., make models-seeds CMD='eval'"; exit 2; fi
	for m in $(MODELS); do \
	  echo "== $$m : $$CMD =="; \
	  $(PY) -m app.main $$CMD --model $$m --config $(CFG) || true; \
	done

# -------- Tables / Grids / Report -------------------------------------------
table:
	$(PY) scripts/collect_scores.py

scores-csv:
	$(PY) scripts/jsonl_to_csv.py

grids:
	$(PY) scripts/make_grids.py

report:
	$(PY) scripts/phase1_report.py
	echo "Report: artifacts/phase1_report.md"

# -------- Cleaning -----------------------------------------------------------
clean-summaries:
	rm -f "$(OUT_JSONL)" || true
	find "$(SUMMARIES_DIR)" -type f -name 'summary_*.json' -delete || true
	rm -f artifacts/*/summaries/latest.json || true
	echo "Cleaned summaries."

clean-synth:
	find "$(SYN_BASE)" -type f -path "$(SYN_BASE)/*/seed*/*" -delete || true
	echo "Cleaned synthetic artifacts."

# -------- Consolidate per-model JSON summaries → one JSONL -------------------
summaries-jsonl:
	@echo "Building consolidated JSONL…"
	@if compgen -G "artifacts/*/summaries/summary_*.json" > /dev/null; then \
	  SCHEMA_ARG=""; \
	  if [ -f "$(SCHEMA_PATH)" ]; then \
	    echo "Using schema: $(SCHEMA_PATH)"; \
	    SCHEMA_ARG="--schema $(SCHEMA_PATH)"; \
	  else \
	    echo "Schema not found → skipping JSON Schema validation (fast path)"; \
	  fi; \
	  $(PY) scripts/summaries_to_jsonl.py \
	    --glob "artifacts/*/summaries/summary_*.json" \
	    --out "$(OUT_JSONL)" $$SCHEMA_ARG --reset; \
	  echo "Built $(OUT_JSONL)"; \
	else \
	  echo "No per-model summaries found at artifacts/*/summaries/summary_*.json"; \
	  exit 2; \
	fi

# ---- Demo (local only) ------------------------------------------------------
demo:
	$(PY) demo/app.py

# ---- Talon helpers (Phase 3) -----------------------------------------------
# Submit CPU matrix with env overrides (uses slurm/models_seeds_matrix_cpu.slurm)
submit-array:
	@echo "Submitting CPU matrix job…"
	@sbatch --export=ALL,MODELS="$(MODELS)",SEEDS="$(SEEDS)",SYN_PER_CLASS="$(SYN_PER_CLASS)",REPO_DIR="$$(pwd)" \
		slurm/models_seeds_matrix_cpu.slurm

# Submit GPU matrix
submit-array-gpu:
	@echo "Submitting GPU matrix job…"
	@sbatch --export=ALL,MODELS="$(MODELS)",SEEDS="$(SEEDS)",SYN_PER_CLASS="$(SYN_PER_CLASS)",REPO_DIR="$$(pwd)" \
		slurm/models_seeds_matrix_gpu.slurm

# Watch your queue (press q to exit)
monitor:
	@watch -n 2 'squeue -u $$USER'

# Print the most recent .out file path
lastlog:
	@ls -t slurm/*.out 2>/dev/null | head -1 || ls -t *.out 2>/dev/null | head -1 || echo "no .out yet"

# Tail the most recent .out (Ctrl-C to exit)
tailf:
	@f=$$(ls -t slurm/*.out 2>/dev/null | head -1 || ls -t *.out 2>/dev/null | head -1); \
	if [ -n "$$f" ]; then echo "Tailing $$f …"; tail -n 200 -f "$$f"; else echo "no .out yet"; fi

# -------- Slurm example (print-only) ----------------------------------------
slurm-help:
	echo "# Submit CPU matrix (override on CLI as needed):"
	echo "make submit-array MODELS='diffusion cdcgan cvae' SEEDS='42 43 44' SYN_PER_CLASS=1000"
	echo "# Submit GPU matrix:"
	echo "make submit-array-gpu MODELS='gan diffusion vae' SEEDS='42 43 44'"

# -------- Everything ---------------------------------------------------------
all: setup synth eval grids table report
	echo "All done."
