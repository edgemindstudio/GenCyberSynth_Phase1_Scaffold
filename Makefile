# Makefile — CI-safe & Talon-ready (Phase-1 lock + gate; backfill runs separately on compute)

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
        submit-array submit-array-gpu monitor lastlog tailf slurm-help all

# -------- Globals (override on CLI) ------------------------------------------
PY               ?= python
CFG              ?= configs/config.yaml
SMOKE_CFG        ?= configs/config.smoke.yaml
MODELS           ?= gan diffusion vae autoregressive maskedautoflow restrictedboltzmann gaussianmixture
SMOKE_MODEL      ?= gan
SEEDS            ?= 42 43 44
SYN_PER_CLASS    ?= 1000

# Paper-1 lock defaults
PHASE1_SUMMARY_NAME    ?= paper1.json
PHASE1_MANIFEST_NAME   ?= paper1_manifest.json
PHASE1_ALLOWED_BUDGETS ?= 2000

# Paths
SUMMARIES_DIR    ?= artifacts/summaries
OUT_JSONL        ?= $(SUMMARIES_DIR)/phase1_summaries.jsonl

# -------- Help ---------------------------------------------------------------
help:
	echo "Targets:"
	echo "  phase1_freeze     - freeze snapshots + paper-lock manifests"
	echo "  phase1_gate       - freeze → scores → check (fails fast)"
	echo "  phase1_backfill   - backfill KID + downstream into paper1.json (RUN ON COMPUTE)"
	echo "  paper1_prepare    - gate only (lightweight)"
	echo "  paper1_build      - build JSONL/CSV/table/figs/report (lightweight-ish)"
	echo "  paper1            - prepare + build (NO backfill)"
	echo ""
	echo "Common overrides:"
	echo "  make paper1_prepare PHASE1_ALLOWED_BUDGETS=2000 PHASE1_SUMMARY_NAME=paper1.json PHASE1_MANIFEST_NAME=paper1_manifest.json"

# -------- Setup --------------------------------------------------------------
setup:
	pip install -r requirements.txt
	mkdir -p "$(SUMMARIES_DIR)"

# -------- Fast CI sanity -----------------------------------------------------
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
onepass:
	$(PY) -m app.main train --model $(MODEL) --config $(CFG) || true
	$(PY) -m app.main synth --model $(MODEL) --config $(CFG)
	$(PY) -m app.main eval  --model $(MODEL) --config $(CFG)

onepass-seeds:
	@for s in $(SEEDS); do \
	  echo ">> MODEL=$(MODEL) SEED=$$s"; \
	  SEED=$$s $(MAKE) -s onepass MODEL=$(MODEL); \
	done

onepass-all:
	for m in $(MODELS); do \
	  $(MAKE) -s onepass MODEL=$$m; \
	done

models-seeds:
	@if [ -z "$$CMD" ]; then echo "Set CMD, e.g., make models-seeds CMD='eval'"; exit 2; fi
	for m in $(MODELS); do \
	  echo "== $$m : $$CMD =="; \
	  $(PY) -m app.main $$CMD --model $$m --config $(CFG) || true; \
	done

# -------- Phase-1 lock + sanity gate ----------------------------------------
# 1) Freeze paper snapshot(s) so later smoke runs can't change the paper build
phase1_freeze:
	PHASE1_ALLOWED_BUDGETS='$(PHASE1_ALLOWED_BUDGETS)' \
	PHASE1_SUMMARY_NAME='$(PHASE1_SUMMARY_NAME)' \
	PHASE1_MANIFEST_NAME='$(PHASE1_MANIFEST_NAME)' \
	$(PY) tools/freeze_phase1_snapshots.py

# 2) Build paper CSV from frozen snapshots (quick)
phase1_scores:
	PHASE1_SUMMARY_NAME="$(PHASE1_SUMMARY_NAME)" \
	$(PY) tools/build_phase1_scores.py

# 3) Enforce integrity (budget, num_fake, presence of required keys)
phase1_check:
	PHASE1_ALLOWED_BUDGETS="$(PHASE1_ALLOWED_BUDGETS)" \
	PHASE1_SUMMARY_NAME="$(PHASE1_SUMMARY_NAME)" \
	$(PY) tools/check_phase1_integrity.py

# Backfill is intentionally separate — run on compute node (GPU ok)
phase1_backfill:
	PHASE1_SUMMARY_NAME='$(PHASE1_SUMMARY_NAME)' \
	PHASE1_MANIFEST_NAME='$(PHASE1_MANIFEST_NAME)' \
	$(PY) scripts/backfill_kid_and_downstream.py

phase1_gate: phase1_freeze phase1_scores phase1_check
	@echo "[phase1_gate] OK"

# -------- Normalize + build JSONL from paper snapshots -----------------------
normalize-summaries:
	@echo "Normalizing summaries (schema harmonization)…"
	$(PY) scripts/normalize_summaries.py

paper1-jsonl: normalize-summaries
	@echo "Building consolidated JSONL from $(PHASE1_SUMMARY_NAME)…"
	PHASE1_SUMMARY_NAME='$(PHASE1_SUMMARY_NAME)' \
	OUT_JSONL='$(OUT_JSONL)' \
	$(PY) tools/build_paper1_jsonl.py

# -------- Tables / CSV / Report ---------------------------------------------
table: paper1-jsonl
	@echo "Collecting phase-1 score table (collect_scores.py)…"
	PHASE1_SUMMARY_NAME='$(PHASE1_SUMMARY_NAME)' $(PY) scripts/collect_scores.py

scores-csv: paper1-jsonl
	@echo "Exporting consolidated JSONL → CSV (jsonl_to_csv.py)…"
	$(PY) scripts/jsonl_to_csv.py

grids:
	$(PY) scripts/make_grids.py

report: table
	$(PY) scripts/phase1_report.py
	echo "Report: artifacts/phase1_report.md"

# -------- Figures ------------------------------------------------------------
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

# parallel_coords needs --cols; give it a sane default set
figs-hparams: paper1-jsonl
	$(PY) -m scripts.plots.hparams.parallel_coords \
	  --cols metrics.kid metrics.downstream.macro_f1 metrics.ms_ssim metrics.downstream.balanced_acc \
	  || true

figs-all: figs-core figs-diversity figs-imbalance figs-qual figs-hparams
	@echo "Figures → artifacts/figures/**"

# -------- One-command Paper-1 build (NO backfill) -----------------------------
paper1_prepare: phase1_gate
	@echo "[paper1_prepare] OK"

paper1_build: paper1-jsonl scores-csv table figs-all report
	@echo "[paper1_build] OK"

paper1: paper1_prepare paper1_build
	@echo "[paper1] OK"

# -------- Cleaning -----------------------------------------------------------
clean-summaries:
	rm -f "$(OUT_JSONL)" || true
	rm -f artifacts/phase1_scores.csv || true
	rm -f artifacts/summaries/phase1_summaries.jsonl || true
	rm -f artifacts/*/summaries/paper1.json || true
	echo "Cleaned summaries."

# Your synthetic lives at artifacts/<model>/synthetic/** (not artifacts/synthetic/**)
clean-synth:
	find artifacts -type f -path "artifacts/*/synthetic/**/*.png" -delete || true
	find artifacts -type f -path "artifacts/*/synthetic/*manifest*.json" -delete || true
	echo "Cleaned synthetic artifacts."

demo:
	$(PY) demo/app.py

# -------- Talon helpers ------------------------------------------------------
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
	echo "# Submit CPU matrix:"
	echo "make submit-array MODELS='gan diffusion' SEEDS='42 43 44' SYN_PER_CLASS=1000"
	echo "# Submit GPU matrix:"
	echo "make submit-array-gpu MODELS='gan diffusion vae' SEEDS='42 43 44'"

all: setup synth eval paper1
	echo "All done."
