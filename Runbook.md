# Runbook

One-page guide to train → synthesize → evaluate, plus FID/KID, artifacts, Slurm, and quick troubleshooting.
Badges: Python 3.10 or 3.11 • CI • License

## TL;DR

```bash
    # first time on a machine
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export PYTHONPATH="$(pwd)/gcs-core:$PYTHONPATH"
    make setup
    
    # quick health check (CPU)
    make smoke SMOKE_MODEL=gan CFG=configs/config.smoke.yaml
    
    # one model end-to-end
    make onepass MODEL=gan CFG=configs/config.yaml
    
    # FID/KID bar plots from logs
    MPLBACKEND=Agg make -f Makefile.fid fid-all

```

---

### Models & Configs

Supported models: `gan diffusion vae autoregressive maskedautoflow restrictedboltzmann gaussianmixture`
Default config: `configs/config.yaml`
Smoke config: `configs/config.smoke.yaml`

Override on any command, e.g.:

```bash
    make synth CFG=configs/config.yaml MODELS="gan diffusion" SYN_PER_CLASS=1000

```

### Local Setup (CPU-only by default)

```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export PYTHONPATH="$(pwd)/gcs-core:$PYTHONPATH"
    
    # Helpful env for stable CPU numerics
    export CUDA_VISIBLE_DEVICES=""
    export TF_CPP_MIN_LOG_LEVEL=2
    export TF_ENABLE_ONEDNN_OPTS=0
    
    make setup

```
---

### Core Make Targets

```bash
    # fast smoke (default gan; change via SMOKE_MODEL)
    make smoke SMOKE_MODEL=vae CFG=configs/config.smoke.yaml
    
    # train/synth/eval loops over MODELS
    make train  CFG=configs/config.yaml
    make synth  CFG=configs/config.yaml SYN_PER_CLASS=1000
    make eval   CFG=configs/config.yaml
    
    # single model end-to-end
    make onepass MODEL=gan CFG=configs/config.yaml
    
    # sweep a custom subcommand across MODELS (e.g., eval)
    make models-seeds CMD="eval" CFG=configs/config.yaml

```
Artifacts land here:

- Synthetic: `artifacts/<model>/synthetic/`
- Per-run eval JSON: `artifacts/<model>/summaries/summary_*.json`
- Consolidated JSONL: `artifacts/summaries/phase1_summaries.jsonl`
- Preview grids: `artifacts/preview_grids/`
- CSV scores: `artifacts/phase1_scores.csv`

---

### Post-Processing & Figures

```bash
    # grids & basic reports
    make grids
    make scores-csv
    make report   # writes artifacts/phase1_report.md

```

---

### FID/KID Pipeline (Standalone)

`Makefile.fid` provides a clean, tabless pipeline from Slurm logs → CSVs → plots.

```bash
    # everything: parse → best per model → plot → best per (model, backbone, img) → plot → pretty print
    MPLBACKEND=Agg make -f Makefile.fid fid-all
    
    # parts
    make -f Makefile.fid fid-parse     # builds artifacts/summaries/fid_grid_combined.csv
    make -f Makefile.fid fid-best      # best per model → CSV+MD
    MPLBACKEND=Agg make -f Makefile.fid fid-plot
    make -f Makefile.fid fid-best-3d   # best per (model, backbone, img_size) → CSV+MD
    MPLBACKEND=Agg make -f Makefile.fid fid-plot-3d
    make -f Makefile.fid fid-show      # sorted pretty table

```
Outputs to expect:
- `artifacts/summaries/fid_grid_combined.csv`
- `artifacts/summaries/fid_grid_best_per_model.csv & .md`
- `artifacts/summaries/fid_best_bar.png`
- `artifacts/summaries/fid_grid_best_per_model_backbone_img.csv & .md`
- `artifacts/summaries/fid_best_bar_backbone_img.png`

### Talon / Slurm

Submit matrix jobs (edit partition/image as needed in slurm scripts):

```bash
    # CPU matrix
    make submit-array MODELS="gan diffusion vae" SEEDS="42 43 44" SYN_PER_CLASS=1000
    
    # GPU matrix
    make submit-array-gpu MODELS="gan diffusion vae" SEEDS="42 43 44" SYN_PER_CLASS=1000

```
Monitor & logs:
```bash
    watch -n 2 'squeue -u $USER'
    make lastlog
    make tailf

```
After jobs finish:
```bash
    MPLBACKEND=Agg make -f Makefile.fid fid-all
    
```
---

### Common Variants

```bash
   # change sweep
    MODELS="gan diffusion"; SEEDS="41 42 43"; SYN_PER_CLASS=1500
    
    # example
    make synth CFG=configs/config.yaml MODELS="$MODELS" SYN_PER_CLASS=$SYN_PER_CLASS
    make eval  CFG=configs/config.yaml MODELS="$MODELS"

```
---

### Cleaning

```make
    make clean-summaries   # remove per-run JSON + JSONL
    make clean-synth       # clear synthetic dumps

```

---

### Troubleshooting

- **“missing separator” in Makefile**: our main `Makefile` uses tabs; `Makefile.fid` is tabless via `.RECIPEPREFIX`. Don’t mix blocks between them.
- **CUDA / cuInit errors on CPU**: they’re harmless if `CUDA_VISIBLE_DEVICES=""` is set—TensorFlow will run on CPU.
- **Empty FID CSVs**: ensure Slurm FID/KID jobs ran and logs exist under `slurm/logs/`. Then run `make -f Makefile.fid fid-parse`.
- **Schema errors**: verify `gcs-core` is on `PYTHONPATH`.
- **Plots on headless nodes**: set `MPLBACKEND=Agg`.

---

**Bruno Fonkeng** . GenCyberSynth . MIT License
[Github](https://github.com/edgemindstudio) | [LinkedIn](https://www.linkedin.com/in/edgemindstudio/)