
.PHONY: setup smoke synth eval table

PYTHON ?= python

setup:
	pip install -U pip
	pip install -e .
	pip install "gcs-core>=0.1.0"

smoke:
	$(PYTHON) -m pytest -q tests/test_smoke.py

synth:
	$(PYTHON) app/main.py synth --simulate --model-name TEMPLATE --per-class 1000 --out artifacts/synthetic/

eval:
	$(PYTHON) app/main.py eval --simulate --model-name TEMPLATE --fid-cap 200 --json runs/summary.jsonl

table:
	$(PYTHON) tools/aggregate_results.py runs/table.csv runs/summary.jsonl
