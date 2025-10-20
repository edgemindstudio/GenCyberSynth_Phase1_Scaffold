# GenCyberSynth

![Python](https://img.shields.io/badge/Python-3.11-blue)

Unified training → synthesis → evaluation scaffold for cybersecurity image generative models.

Badges: Python 3.11 • CI • License

## Overview

GenCyberSynth coordinates per-model pipelines that optionally train, then synthesize, and finally evaluate cybersecurity image generators through a shared CLI. Every evaluation run appends exactly one compact JSON line to `artifacts/<model>/summaries/summary.jsonl`, guaranteeing downstream compatibility. Model checkpoints, manifests, evaluation logs, and preview grids are consistently written under `artifacts/<model>/...` for reproducible automation.

## Features

- **Model adapters for existing families**: `gan`, `diffusion`, `vae`, `autoregressive`, `maskedautoflow`, `restrictedboltzmann`, `gaussianmixture`.
- **Shared evaluator core** from the `gcs-core` submodule (metrics, schemas, reusable components).
- **Unified CLI entrypoints** in `app/main.py` exposing `train`, `synth`, `eval`, and `list`.

## Repo Layout 

```bash
    configs/                 # YAML defaults for orchestrating models and evaluation
    app/                     # CLI entrypoints and orchestration glue
    adapters/                # (If present) common adapter interfaces
    gan/ ...                 # GAN training/sampling implementation
    diffusion/ ...           # Diffusion pipeline modules
    vae/ ...                 # VAE training/sampling utilities
    autoregressive/ ...      # PixelCNN-style sampler utilities
    maskedautoflow/ ...      # Masked auto-flow training and sampling
    restrictedboltzmann/ ... # RBM synthesis utilities
    gaussianmixture/ ...     # Gaussian mixture synthesis pipeline
    scripts/                 # Automation helpers for grids, summaries, tables, reports
    artifacts/               # Output root for checkpoints, synth data, summaries, grids
    gcs-core/                # Submodule delivering evaluators, metrics, schemas
    .github/                 # Workflows and contributor instructions
    demo/                    # Optional Gradio viewer for preview grids
    tests/                   # Minimal smoke tests for CI sanity

```

---
## Quickstart (Local CPU)

Use Python 3.11. Local installs rely on `requirements.txt` (CI pins against `requirements.ci.txt`).

```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
```
**Make targets** (from the Makefile):
- `make setup`: Install dependencies and create summary directories.
- `make smoke`: Single-model synth + eval (SMOKE_MODEL=gan by default).
- `make smoke-all`: Iterate smoke tests for all configured models
- `make train`: Train for each model/seed (light wrapper over the CLI).
- `make synth`: Generate for each model/seed (uses SYN_PER_CLASS).
- `make eval`: Evaluate for each model/seed; appends one summary JSON.
- `make grids`: Build preview grids in `artifacts/preview_grids`.
- `make summaries-jsonl`: Consolidate per-run summaries into JSONL.
- `make table`: Produce a compact CSV table from the JSONL summaries.
- `make report`: Generate a markdown snapshot at artifacts/phase1_report.md.
- `make demo`: Launch the Gradio preview after grids are generated.
- `make slurm-help`: Print example Slurm array invocations.
- `make all: setup → synth → eval → grids → table → report`.

**Default smoke**:
```bash
    make smoke
```

**Override models**
```bash
    make smoke SMOKE_MODEL=vae
    make synth MODELS="gan diffusion"
```
---

## Configuration

Central configuration resides in `configs/config.yaml`. Seeds, artifact paths, per-class caps, and evaluator options are read by adapters and `eval` runners.

The `gcs-core` submodule (under `gcs-core/`) provides shared evaluators and schemas. If you run outside the repo root, add it to your `PYTHONPATH`:

```bash
    export PYTHONPATH=$(pwd)/gcs-core:$PYTHONPATH
```
---

## Artifacts & Metrics
- **Synthetic dumps**: `artifacts/<model>/synthetic/`
- **Evaluation summaries**: `artifacts/<model>/summaries/summary_*.json`
- **Consolidated JSONL**: `artifacts/summaries/phase1_summaries.jsonl`
- **Preview grids**: `artifacts/preview_grids/<model>_grid.png`
- **CSV table**: `artifacts/phase1_scores.csv`

**Schema reference**: `gcs-core/gcs_core/schemas/eval_summary.lite.schema.json`

### Minimal JSON line (example)
```json
{
  "timestamp": "2024-03-08T12:34:56Z",
  "run_id": "gan_seed42",
  "model": "gan",
  "seed": 42,
  "metrics": {
    "fid_macro": 18.42,
    "cfid_macro": 22.73,
    "js": 0.031,
    "kl": 0.144,
    "diversity": 0.912,
    "real_only": { "accuracy": 0.91, "macro_f1": 0.89 },
    "real_plus_synth": { "accuracy": 0.93, "macro_f1": 0.91 }
  },
  "deltas": { "accuracy": 0.02, "macro_f1": 0.02 },
  "counts": { "num_fake": 9000 }
}
```

### Aggregation workflow

Consolidate per-run summaries:
```bash
    python scripts/summaries_to_jsonl.py \
    --glob "artifacts/*/summaries/summary_*.json" \
    --out artifacts/summaries/phase1_summaries.jsonl \
    --schema gcs-core/gcs_core/schemas/eval_summary.lite.schema.json
```
Generate a CSV table:
```bash
    python scripts/jsonl_to_csv.py
    # writes artifacts/phase1_scores.csv
```
---

## CI

The workflow in `.github/workflows/ci.yml`:
- Installs `requirements.ci.txt`.
- Runs `make smoke-all`.
- Aggregates JSON summaries (`make summaries-jsonl`).
- Builds preview grids (`make grids`).
- Publishes artifacts:
- - `phase1-artifacts-raw`
- - `preview-grids`
- - `phase1-summaries`

All workflow links are relative to this repository’s files.
---

## Optional Demo

A tiny Gradio viewer in `demo/app.py` serves preview grids. After building grids:
```bash
    make grids
    python app.py
```

Then open the local URL shown in the console.
---

## Talon / HPC Usage

Batch execution is supported via the printed examples from:
```bash
    make slurm-help
```

This echoes Slurm array patterns suitable for Apptainer/Singularity images, seed sweeps, and per-model runs. Pipe the printed command into your cluster environment and adjust image/data paths as needed.

---

## 📘 Author

- **Bruno Fonkeng**
- [Github](https://github.com/edgemindstudio) | [LinkedIn](https://www.linkedin.com/in/edgemindstudio/)

---

## License

This project is licensed under the MIT License.