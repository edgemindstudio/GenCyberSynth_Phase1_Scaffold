# Makefile — CI-safe & Talon-ready (robust JSONL consolidation)

SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.SILENT:

.PHONY: help setup smoke smoke-all train synth eval onepass onepass-seeds onepass-all models-seeds \
        grids table scores-csv report clean-summaries clean-synth summaries-jsonl demo \
        submit-array submit-array-gpu monitor lastlog tailf slurm-help all

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

# Optional JSON Schema (used only if present)
SCHEMA_PATH      ?= gcs-core/gcs_core/schemas/eval_summary.lite.schema.json

# Where CI may download the artifacts bundle (actions/download-artifact)
DOWNLOADED_PREFIX ?= phase1-artifacts-raw

# -------- Help ---------------------------------------------------------------
help:
	echo "Targets:"
	echo "  setup             - pip install -r requirements.txt & ensure dirs"
	echo "  smoke             - quick synth+eval for one model (SMOKE_MODEL=$(SMOKE_MODEL))"
	echo "  smoke-all         - quick synth+eval for ALL MODELS"
	echo "  train/synth/eval  - loops over MODELS"
	echo "  onepass           - train→synth→eval for one MODEL (make onepass MODEL=gan)"
	echo "  onepass-seeds     - onepass over SEEDS for one MODEL"
	echo "  onepass-all       - onepass across all MODELS"
	echo "  models-seeds      - run custom CMD over MODELS (make models-seeds CMD='eval')"
	echo "  grids             - build preview grids"
	echo "  table             - legacy aggregate (collect_scores.py)"
	echo "  scores-csv        - JSONL → tiny CSV (artifacts/phase1_scores.csv)"
	echo "  summaries-jsonl   - consolidate per-model JSON → JSONL (handles CI path)"
	echo "  clean-summaries   - remove JSON summaries and JSONL"
	echo "  clean-synth       - remove synthetic artifacts"
	echo "  demo              - run local Gradio viewer (demo/app.py)"
	echo "  submit-array(*gpu)- Talon Slurm matrix jobs"
	echo "  monitor/lastlog/tailf - Talon helpers"
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
	  echo ">> MODEL=$(MODEL) SEED=$$s"; \
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
# Works both locally and in CI (where files live under phase1-artifacts-raw/artifacts/)
summaries-jsonl:
	@echo "Building consolidated JSONL…"
	@{ \
	  # init locals (required because of -u / nounset)
	  found_any=0; \
	  schema_arg=""; \
	  # optional schema
	  if [ -f "$(SCHEMA_PATH)" ]; then \
	    echo "Using schema: $(SCHEMA_PATH)"; \
	    schema_arg="--schema $(SCHEMA_PATH)"; \
	  else \
	    echo "Schema not found → skipping JSON Schema validation (fast path)"; \
	  fi; \
	  # Pass 1: local artifacts
	  if compgen -G "artifacts/*/summaries/summary_*.json" > /dev/null; then \
	    echo "[pass1] artifacts/*/summaries/summary_*.json"; \
	    found_any=1; \
	    $(PY) scripts/summaries_to_jsonl.py \
	      --glob "artifacts/*/summaries/summary_*.json" \
	      --out "$(OUT_JSONL)" $$schema_arg --reset; \
	  fi; \
	  # Pass 2: downloaded CI bundle (actions/download-artifact)
	  if compgen -G "phase1-artifacts-raw/artifacts/*/summaries/summary_*.json" > /dev/null; then \
	    echo "[pass2] phase1-artifacts-raw/artifacts/*/summaries/summary_*.json"; \
	    found_any=1; \
	    $(PY) scripts/summaries_to_jsonl.py \
	      --glob "phase1-artifacts-raw/artifacts/*/summaries/summary_*.json" \
	      --out "$(OUT_JSONL)" $$schema_arg --reset; \
	  fi; \
	  # Final check
	  if [ $$found_any -eq 0 ]; then \
	    echo "No per-model summaries found under either path."; \
	    exit 2; \
	  else \
	    echo "Built $(OUT_JSONL)"; \
	  fi; \
	}

# ---- Demo (local only) ------------------------------------------------------
demo:
	$(PY) demo/app.py

# ---- Talon helpers (Phase 3) -----------------------------------------------
submit-array:
	@echo "Submitting CPU matrix job…"
	@sbatch --export=ALL,MODELS="$(MODELS)",SEEDS="$(SEEDS)",SYN_PER_CLASS="$(SYN_PER_CLASS)",REPO_DIR="$$(pwd)" \
		slurm/models_seeds_matrix_cpu.slurm

submit-array-gpu:
	@echo "Submitting GPU matrix job…"
	@sbatch --export=ALL,MODELS="$(MODELS)",SEEDS="$(SEEDS)",SYN_PER_CLASS="$(SYN_PER_CLASS)",REPO_DIR="$$(pwd)" \
		slurm/models_seeds_matrix_gpu.slurm

monitor:
	@watch -n 2 'squeue -u $$USER'

lastlog:
	@ls -t slurm/*.out 2>/dev/null | head -1 || ls -t *.out 2>/dev/null | head -1 || echo "no .out yet"

tailf:
	@f=$$(ls -t slurm/*.out 2>/dev/null | head -1 || ls -t *.out 2>/dev/null | head -1); \
	if [ -n "$$f" ]; then echo "Tailing $$f …"; tail -n 200 -f "$$f"; else echo "no .out yet"; fi

slurm-help:
	echo "# Submit CPU matrix (override on CLI as needed):"
	echo "make submit-array MODELS='diffusion cdcgan cvae' SEEDS='42 43 44' SYN_PER_CLASS=1000"
	echo "# Submit GPU matrix:"
	echo "make submit-array-gpu MODELS='gan diffusion vae' SEEDS='42 43 44'"

# -------- Everything ---------------------------------------------------------
all: setup synth eval grids table report
	echo "All done."
