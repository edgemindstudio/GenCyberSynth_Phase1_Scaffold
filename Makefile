# Makefile

# Use bash for all recipes (fixes: "/bin/sh: 0: Illegal option -o pipefail")
SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

.SILENT:
# .ONESHELL is optional; you can keep it or drop it. With bash it’s fine.
# .ONESHELL:

.SILENT:
.SHELLFLAGS = -eo pipefail -c
.ONESHELL:

.PHONY: help setup smoke smoke-all train synth eval table grids report clean-summaries clean-synth all

PY ?= python
CFG ?= configs/config.yaml

# All model families in your scaffold (override on the CLI if needed)
MODELS ?= gan diffusion vae autoregressive maskedautoflow restrictedboltzmann gaussianmixture

# Fast CI target uses a single model by default; override: `make smoke SMOKE_MODEL=diffusion`
SMOKE_MODEL ?= gan

help:
	echo "Targets:"
	echo "  setup            - pip install -r requirements.txt"
	echo "  smoke            - quick synth+eval on a single model (default: $(SMOKE_MODEL))"
	echo "  smoke-all        - quick synth+eval on ALL MODELS"
	echo "  train            - (stub) iterate train over MODELS (best-effort)"
	echo "  synth            - generate samples for MODELS"
	echo "  eval             - evaluate samples for MODELS"
	echo "  grids            - build preview grids into artifacts/preview_grids/"
	echo "  report           - write artifacts/phase1_report.md"
	echo "  table            - aggregate summaries → artifacts/phase1_scores.csv"
	echo "  clean-summaries  - remove summary JSONs"
	echo "  clean-synth      - remove synthetic images & manifests"
	echo "  all              - setup → synth → eval → grids → table → report"

setup:
	pip install -r requirements.txt

# ---- Fast CI sanity check (one model) ---------------------------------------
smoke:
	echo "== SMOKE $(SMOKE_MODEL) =="
	$(PY) -m app.main synth --model $(SMOKE_MODEL) --config $(CFG)
	$(PY) -m app.main eval  --model $(SMOKE_MODEL) --config $(CFG)

# ---- Smoke for all models (slower) ------------------------------------------
smoke-all:
	for m in $(MODELS); do \
	  echo "== SMOKE $$m =="; \
	  $(PY) -m app.main synth --model $$m --config $(CFG) || true; \
	  $(PY) -m app.main eval  --model $$m --config $(CFG) || true; \
	done

# ---- Training loop (best-effort; some adapters may not implement) -----------
train:
	for m in $(MODELS); do \
	  echo "== TRAIN $$m =="; \
	  $(PY) -m app.main train --model $$m --config $(CFG) || true; \
	done

# ---- Synthesis --------------------------------------------------------------
synth:
	for m in $(MODELS); do \
	  echo "== SYNTH $$m =="; \
	  $(PY) -m app.main synth --model $$m --config $(CFG) || true; \
	done

# ---- Evaluation -------------------------------------------------------------
eval:
	for m in $(MODELS); do \
	  echo "== EVAL $$m =="; \
	  $(PY) -m app.main eval --model $$m --config $(CFG) || true; \
	done

# ---- Tables / Grids / Report -----------------------------------------------
table:
	$(PY) scripts/collect_scores.py

grids:
	$(PY) scripts/make_grids.py

report:
	$(PY) scripts/phase1_report.py
	echo "Report: artifacts/phase1_report.md"

# ---- Cleaning helpers -------------------------------------------------------
clean-summaries:
	find artifacts -type f -path "*/summaries/summary_*.json" -delete || true
	rm -f artifacts/*/summaries/latest.json || true
	echo "Cleaned summaries."

clean-synth:
	find artifacts -type f -path "*/synthetic/*" -delete || true
	echo "Cleaned synthetic artifacts."

# ---- Everything -------------------------------------------------------------
all: setup synth eval grids table report
