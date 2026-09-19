# Paper 01 Benchmark

## TrustForge study identity

`paper01_benchmark`

## Historical study identity

GenCyberSynth Phase-1

## Purpose

This directory is the TrustForge representation of the frozen Paper 1
benchmark study.

It does not relocate, rewrite, normalize, or replace the historical Paper 1
execution materials.

Historical configurations, Slurm logs, generated artifacts, result tables,
and Git history remain authoritative evidence in their original locations.

The files in this directory provide a stable map from TrustForge scientific
identity to that preserved historical evidence.

## Scientific question

How do multiple generative model families compare for synthetic malware-image
generation under common dataset, seed, synthetic-budget, and evaluation
protocols?

## Benchmark scope

Paper 1 evaluates seven generative model families:

- GAN
- VAE
- diffusion
- autoregressive
- restricted Boltzmann machine
- Gaussian mixture
- masked autoregressive flow

The benchmark uses three scientific seeds:

- 42
- 43
- 44

The benchmark includes two malware-image datasets:

- USTC-TFC2016 malware-image representation
- CICMalDroid2020 Paper 1 representation

The principal synthetic budget is 2,000 generated samples per class.

## TrustForge migration doctrine

Historical Paper 1 executions are not retroactively classified as
TrustForge-native executions.

TrustForge preserves the distinction between:

1. scientific identity;
2. historical execution identity;
3. machine-specific execution paths;
4. historical artifacts and evidence;
5. current TrustForge representations.

A TrustForge representation may describe a historical execution without
changing that execution's provenance.

## Historical material retained in place

Important preserved material includes:

- `configs/paper1_final_rerun.yaml`
- `configs/paper1_cicmaldroid.yaml`
- `slurm/run_paper1.slurm`
- `papers/paper1_phase1_benchmark/logs/`
- historical Git commits identified in `lineage.yaml`
- external Paper 1 artifact roots identified in `evidence.yaml`

These materials must not be rewritten merely for portability.

## Evidence authority

The accepted USTC result table is:

`/home/bruno.fonkeng/gencys/artifacts_paper1_final_rerun/phase1_scores_dedup.csv`

with SHA256:

`c3c731f355f4cecfb930d024dd8ee42d97ce952d5abfca73c717af6b2a7a6031`

The accepted CICMalDroid result table is:

`/home/bruno.fonkeng/gencys/artifacts_paper1_cicmaldroid/phase1_scores_dedup.csv`

with SHA256:

`63fc2cd7b2cde81f2b964cc4407c6efdb1c50ee8bd61bfbbd4384d7eff8fc4f4`

An older USTC sidecar records the digest:

`9f48138cf5acc5188fb77286a61cbddc8fe1c07e9baef995bd7d76bd2bc2fefa`

That value is retained as historical provenance. It does not match the
current accepted USTC table, and the exact earlier table bytes corresponding
to that digest have not been identified among the current top-level Paper 1
CSV files.

See `evidence.yaml` for the complete evidence map.

## Migration boundary

M6 represents Paper 1 inside TrustForge.

M7 will separately determine whether current TrustForge-compatible execution
can scientifically reproduce the frozen Paper 1 benchmark.

Migration success does not imply scientific regression success.
