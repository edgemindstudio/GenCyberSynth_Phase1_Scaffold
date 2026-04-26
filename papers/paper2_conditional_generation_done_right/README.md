# Paper 2: Conditional Generation Done Right

## Title

Conditional Generation Done Right: Class-Conditional Evaluation, Conditioning Audits, and Minority-Class Failure Analysis for Synthetic Cybersecurity Images

## Repository Role

This is Paper 2 of the GenCyberSynth dissertation project. It is implemented as a paper module inside the existing `GenCyberSynth_Phase1_Scaffold` repository.

This is not a separate repository and not a clean-slate rewrite. It must reuse shared dataset handling, shared model and evaluation infrastructure, and existing artifact conventions wherever possible.

## Scope

This paper studies whether class-conditional synthetic malware image generators actually respect requested class labels and whether global generative metrics hide class-level or minority-class failures.

The paper focuses on class-conditional generation quality, label consistency, per-class behavior, and downstream detection utility, especially for minority malware classes.

## Core Research Questions

1. Do class-conditional generators produce samples that match the requested class condition?
2. Which malware classes are easiest or hardest to generate conditionally?
3. Do minority classes show worse fidelity, diversity, or downstream utility?
4. Can global generative metrics hide class-level conditional generation failure?
5. Does class-conditional augmentation improve minority-class detection compared with real-only baselines?

## Main Evaluation Areas

- Conditioning audits
- Per-class generative metrics
- Per-class downstream utility
- Minority-class failure analysis
- Label-consistency analysis
- Class-leakage analysis
- Global-versus-class-level metric comparison

## Expected Subdirectories

- `configs/`: paper-specific configs and overrides
- `slurm/`: paper-specific Slurm job scripts
- `scripts/`: paper-specific orchestration and analysis scripts
- `logs/`: Slurm logs separated by phase
- `results/`: raw outputs, tables, and figures
- `manifests/`: paper-level manifests and run indexes
- `notes/`: planning notes, decisions, and experiment logs
- `paper/`: manuscript-related materials

## Workflow Rules

1. All Paper 2 work must live under `papers/paper2_conditional_generation_done_right/`.
2. Paper 2 must reuse shared repository infrastructure instead of duplicating full pipelines.
3. Slurm logs must not be written to the repository root.
4. Paper-specific Slurm scripts must write logs into the `logs/` subdirectories.
5. Experiments must be traceable through configs, manifests, summaries, tables, and figures.
6. Raw logs are execution evidence, but the primary reproducibility evidence should be frozen configs, manifests, checksums, structured summaries, and paper-level result artifacts.

## Initial Scientific Direction

Paper 1 established a standardized cross-family evaluation framework for synthetic cybersecurity image generators.

Paper 2 extends that foundation by focusing specifically on conditional generation. The central claim is that class-conditional generation should not be judged only by global image-quality metrics. A generator may achieve acceptable global scores while still failing to respect class labels, leaking samples across classes, or underperforming on minority classes.

This paper therefore evaluates conditional generation using both global and class-level evidence.

## Planned Outputs

- Per-class conditioning audit tables
- Requested-label versus predicted-label confusion matrices
- Per-class KID or equivalent class-level fidelity metrics
- Per-class diversity summaries
- Minority-class failure reports
- Real-only versus real-plus-synthetic downstream utility tables
- Global-versus-class-level metric comparison figures
- Paper-level reproducibility manifest