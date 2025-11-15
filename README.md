# GenCyberSynth

![Python](https://img.shields.io/badge/Python-3.11-blue)

Unified training → synthesis → evaluation scaffold for cybersecurity image generative models.

Badges: Python 3.11 • CI • License

## Overview

GenCyberSynth coordinates per-model pipelines that optionally train, then synthesize, and finally evaluate cybersecurity image generators through a shared CLI. Every evaluation run appends exactly one compact JSON line to `artifacts/<model>/summaries/summary.json`, and consolidated JSONL builds live in `artifacts/summaries/`, enabling reproducible analysis and plotting.

## Highlights

- **Model families**: `gan`, `diffusion`, `vae`, `autoregressive`, `maskedautoflow`, `restrictedboltzmann`, `gaussianmixture`.
- Shared evaluator logic via the **gcs-core** submodule (metrics, schemas, reusable components).
- Clean **FID/KID** helpers with a standalone Makefile.fid (no TAB pitfalls), including a one-shot fid-all pipeline and a 3D “best per (model, backbone, img_size)” view.

## Repository Layout 

```bash
    gen-cyber-synth/configs/                 # YAML defaults for orchestrating models and evaluation
    app/                     # CLI entry points and orchestration glue
    adapters/                # Adapter interfaces (if present)
    gan/                     # GAN training/sampling implementation
    diffusion/               # Diffusion pipeline modules
    vae/                     # VAE training/sampling utilities
    autoregressive/          # PixelCNN-style sampler utilities
    maskedautoflow/          # Masked auto-flow training and sampling
    restrictedboltzmann/     # RBM synthesis utilities
    gaussianmixture/         # Gaussian mixture synthesis pipeline
    scripts/                 # Automation helpers (grids, summaries, tables, reports, metrics)
      └── metrics/
          fid_parse.py
          fid_best.py
          fid_plot.py
          fid_best_3d.py     # NEW: best per (model, backbone, img_size)
    artifacts/               # Output root (checkpoints, synthetic dumps, summaries, figures)
    gcs-core/                # Submodule with evaluators, metrics, schemas
    .github/                 # CI workflows
    demo/                    # Optional Gradio viewer for preview grids
    tests/                   # Minimal smoke tests for CI sanity
    Makefile                 # Main pipeline targets (train/synth/eval etc.)
    Makefile.fid             # NEW: FID/KID helpers (standalone; tabless recipes)
    requirements.txt        # Local development dependencies
```

---
## Quickstart (Local CPU)

Use Python 3.10 or 3.11.

```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    make setup
    make smoke

```

**Common targets (from the main `Makefile`):** :\
- `make setup` – install deps and ensure summary dirs exist.
- `make smoke` – synth + eval for a single model (default `SMOKE_MODEL=gan`).
- `make smoke-all` – quick synth + eval across all configured models.
- `make train|synth|eval` – loop over `MODELS` for that stage.
- `make onepass MODEL=gan` – train→synth→eval for a single model.
- `make onepass-seeds MODEL=gan` – run `onepass` over the configured `SEEDS`.
- `make onepass-all` – `onepass` for all `MODELS`.
- `make grids` – build preview grids.
- `make summaries-jsonl` – consolidate per-run summaries to JSONL.
- `make table` – CSV table from the JSONL summaries.
- `make report` – snapshot markdown at `artifacts/phase1_report.md`.
- `make demo` – launch the Gradio preview after grids are built.
- `make slurm-help` – print sample Slurm array invocations.
- `make all – setup → synth → eval → grids → table → report`.

```bash
    make smoke
    make smoke SMOKE_MODEL=vae
    make synth MODELS="gan diffusion"
```
---

## Configuration

Global settings live in `configs/config.yaml` (seeds, caps, artifact paths, evaluator options).
Shared evaluators/schemas are in the `gcs-core` submodule.

If running outside the repo root:

```bash
    export PYTHONPATH=$(pwd)/gcs-core:$PYTHONPATH
```

---

## Artifacts & Metrics

- **Synthetic**: `artifacts/<model>/synthetic/`
- **Per-run JSON summaries**: `artifacts/<model>/summaries/summary_*.json`
- **Consolidated JSONL**: `artifacts/summaries/phase1_summaries.jsonl`
- **Preview grids**: `artifacts/preview_grids/<model>_grid.png`
- **Compact CSV**: `artifacts/phase1_scores.csv`
**Schema**: `gcs-core/gcs_core/schemas/eval_summary.lite.schema.json`


---


### Minimal JSON (example)
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

### Aggregation

```bash
    python scripts/summaries_to_jsonl.py \
    --glob "artifacts/*/summaries/summary_*.json" \
    --out artifacts/summaries/phase1_summaries.jsonl \
    --schema gcs-core/gcs_core/schemas/eval_summary.lite.schema.json

    python scripts/jsonl_to_csv.py
    # writes artifacts/phase1_scores.csv
```
---

## FID/KID Workflow (Clean, Standalone)

We maintain a separate `Makefile.fid` that contains tabless recipes (uses `.RECIPEPREFIX := >`) to avoid “missing separator” errors. This file is CI/Talon-safe and headless plotting friendly.

**Key Targets (in `Makefile.fid`)**
- `fid-parse` – parse Slurm logs → combined CSV 
   → `artifacts/summaries/fid_grid_combined.csv`
- `fid-best` – best FID per model
   → `fid_grid_best_per_model.csv/.md, fid_best_bar.png`
- `fid-best-3d` – best FID per **(model, backbone, img_size)**
   → `fid_grid_best_per_model_backbone_img.csv/.md`
- `fid-plot / fid-plot-3d` – bar charts (headless)
- `fid-show` – pretty, fid-sorted table view in terminal
- `fid-grid-all` – echo artifacts produced
- `fid-all` – **one-shot** end-to-end FID pipeline

We also added `scripts/metrics/fid_best_3d.py` which does the group-by across `(model, backbone, img_size)` and writes markdown + CSV.

**One-shot run**:

```bash
   # Headless-friendly (e.g., Talon, CI)
   MPLBACKEND=Agg make -f Makefile.fid fid-all

```

**What you’ll see produced**
- `artifacts/summaries/fid_grid_combined.csv`
- `artifacts/summaries/fid_grid_best_per_model.csv`
- `artifacts/summaries/fid_grid_best_per_model.md`
- `artifacts/summaries/fid_best_bar.png`
- `artifacts/summaries/fid_grid_best_per_model_backbone_img.csv`
- `artifacts/summaries/fid_grid_best_per_model_backbone_img.md`
- `artifacts/summaries/fid_best_bar_backbone_img.png`

**Quick looks**

```bash
    # Pretty table sorted by FID (ascending)
    make -f Makefile.fid fid-show

    # Rebuild only plots
    MPLBACKEND=Agg make -f Makefile.fid fid-plot fid-plot-3d

```
**Optional handy targets (can be added to `Makefile.fid`):**

```make
    .PHONY: fid-clean
    fid-clean:
    > rm -f "$(SUMMARIES_DIR)/fid_grid_combined.csv" \
    >       "$(SUMMARIES_DIR)/fid_grid_best_per_model.csv" \
    >       "$(SUMMARIES_DIR)/fid_grid_best_per_model.md" \
    >       "$(SUMMARIES_DIR)/fid_grid_best_per_model_backbone_img.csv" \
    >       "$(SUMMARIES_DIR)/fid_grid_best_per_model_backbone_img.md" \
    >       "$(SUMMARIES_DIR)/fid_best_bar.png" \
    >       "$(SUMMARIES_DIR)/fid_best_bar_backbone_img.png" || true
    > echo "Cleaned FID artifacts."
    
    .PHONY: fid-top
    fid-top: fid-parse
    > @(head -n1 "$(FID_COMBINED_CSV)"; \
    >   tail -n +2 "$(FID_COMBINED_CSV)" | sort -t, -k9,9g | head -n 15) | column -t -s,
```

**Calling from the main `Makefile` (optional)**

To trigger the FID pipeline without `-f Makefile.fid`, include this near the top of your **main** `Makefile`:

Then you can run `make fid-all` from the main `Makefile`.

```make
    -include Makefile.fid

    .PHONY: fid
    fid: fid-all

```

Then simply run:
```bash
    MPLBACKEND=Agg make fid
```
---

## CI

The workflow in `.github/workflows/ci.yml` should:
- Installs `requirements.ci.txt`.
- Runs `make smoke-all`.
- Consolidate summaries (`make summaries-jsonl`).
- Builds preview grids (`make grids`).
- Optionally run the FID pipeline:
```bash
    MPLBACKEND=Agg make -f Makefile.fid fid-all
```
- Publish artifacts (`phase1-artifacts-raw, preview-grids, phase1-summaries`, FID figures).
- 
All workflow links are relative to this repository’s files.
---

## Talon / HPC Usage

**Submit matrix jobs:**

```bash
    make slurm-help
    # Prints examples like:
    # make submit-array MODELS='diffusion cdcgan cvae' SEEDS='42 43 44' SYN_PER_CLASS=1000
    # make submit-array-gpu MODELS='gan diffusion vae' SEEDS='42 43 44'
```

**Monitor**
```bash
    watch -n 2 'squeue -u $USER'
```
**Tail the latest**
```bash
    make tailf
```  
**After runs complete (on Talon or locally), build FID figures headlessly:**
```bash
    MPLBACKEND=Agg make -f Makefile.fid fid-all
```
---

## Troubleshooting

- **“missing separator” in make**
   Caused by TAB vs spaces or CRLF endings.
   -- Our `Makefile.fid` uses `.RECIPEPREFIX := >` to prevent TAB issues.
   -- Normalize endings: `sed -i 's/\r$//' Makefile Makefile.fid scripts/metrics/*.py`
- **Headless plotting (servers/CI)**
    Always force: `MPLBACKEND=Agg`.
- **Permissions**
    If any helper isn’t executable:
    **chmod +x scripts/metrics/*.py** (harmless if already executable).
- **Duplicate outputs**
   `fid_best.py` and `fid_best_3d.py` both write “best” CSV/MD (different groupings). That’s intended—filenames are distinct:
   -- per-model: `fid_grid_best_per_model.*`
   -- per-(model,backbone,img): `fid_grid_best_per_model_backbone_img.*`
- **Paths**
   The FID helpers expect logs/rows aggregated by **fid_parse.py**. If **fid_grid_combined.csv** is empty, re-run **fid-parse** or confirm Slurm outputs exist.

---

## Author

- **Bruno Fonkeng**
- [Github](https://github.com/edgemindstudio) | [LinkedIn](https://www.linkedin.com/in/edgemindstudio/)

---

## License

This project is licensed under the MIT License.