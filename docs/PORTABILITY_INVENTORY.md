# TrustForge Portability Inventory

## Purpose

This document classifies machine-specific path usage discovered during the
TrustForge foundation audit.

The purpose is to distinguish historical provenance from active portability
problems.

No path is authorized for modification merely because it appears in this
inventory.

## 1. Audit Summary

Initial broad scan:

- 11,193 path-related matches
- 1,035 files

Most matches occur in historical logs, summaries, manifests, snapshots, and
other evidence-bearing material.

Executable/config-focused scan:

- 824 matches
- 231 files

Most of those files belong to study-specific Paper 2, Paper 3, and Paper 4
research modules.

Therefore raw path-match count is not an appropriate measure of migration
work.

## 2. Classification A - Frozen Historical Compatibility

The following should not be rewritten in place during foundation migration:

- Paper 1-specific configs;
- Paper 1-specific Slurm runners;
- historical Paper 1 backfill scripts;
- root Paper 2 budget configs where superseded by the Paper 2 study module;
- study-local Paper 2 experiment configs and Slurm jobs;
- study-local Paper 3 experiment configs and Slurm jobs;
- study-local Paper 4 experiment configs and Slurm jobs;
- historical manifests;
- logs;
- summaries;
- frozen evidence tables.

Absolute paths inside historical records may be valid execution provenance.

## 3. Classification B - Transitional Shared Core

The following still participate in general repository operation and require
controlled TrustForge migration:

- eval/runner.py
- configs/config.yaml
- Makefile
- tests/test_smoke.py
- .github/workflows/smoke.yml
- model training entry points that default to configs/config.yaml

These files must not be changed by blind search-and-replace.

They should eventually consume a portable TrustForge path/configuration
interface.

## 4. eval/runner.py

eval/runner.py contains a Talon-specific dataset fallback.

However, it was still modified during Paper 3 development.

Therefore it is not purely historical Paper 1 code.

It is a mixed shared/historical component.

Migration rule:

Do not patch its historical path behavior until the TrustForge experiment,
path, and compatibility contracts exist.

Future architecture should separate portable path resolution from historical
evaluation semantics.

## 5. configs/config.yaml

configs/config.yaml is still a central repository default.

It is referenced by:

- Makefile;
- smoke tests;
- GitHub workflow;
- training modules;
- Slurm helpers;
- Runbook;
- README documentation.

Therefore it cannot currently be treated as an obsolete historical config.

Migration rule:

Create a TrustForge-native successor or portable configuration layer before
changing or retiring this file.

## 6. Historical Paper 1 Configs

Examples include:

- configs/paper1_config.yaml
- configs/paper1_final.yaml
- configs/paper1_final_rerun.yaml
- configs/paper1_real.yaml
- configs/paper1_smoke.yaml
- configs/paper1_cicmaldroid.yaml

These encode historical Talon dataset and artifact locations.

Those paths are part of recovered Paper 1 provenance.

Migration rule:

Preserve the original configs.

TrustForge reproduction may later use compatibility manifests or translated
study definitions rather than rewriting the originals.

## 7. Root Paper 2 Budget Configs

Examples:

- configs/paper2_500.yaml
- configs/paper2_1000.yaml
- configs/paper2_2000.yaml

Paper 2 later developed a dedicated study module with its own configurations
and evidence index.

These root configs should therefore be treated as legacy/historical until
their exact relationship to the dedicated Paper 2 study is formally recorded.

## 8. Root Slurm Infrastructure

Several root Slurm files contain fixed Talon repository paths and artifact
roots.

Examples include:

- slurm/synth_eval.sbatch
- slurm/synth_eval_cpu.sbatch
- slurm/synth_eval_gpu.sbatch
- slurm/eval_only_cpu.sbatch
- slurm/fid_kid_cpu.sbatch
- slurm/fid_kid_per_model_cpu.sbatch
- slurm/run_paper1.slurm

Most were last modified during Paper 1-era development.

Migration rule:

Do not make these the permanent TrustForge HPC interface.

Preserve them as historical or compatibility execution infrastructure.

TrustForge will later introduce a new HPC backend driven by experiment
definitions.

## 9. Legacy Utilities

The following are candidates for later classification as archive, compatibility
tool, or reusable utility:

- run_tuning.slurm
- scripts/backfill_kid_and_downstream.py
- scripts/metrics/print_cfid_table.py
- scripts/tuning_dashboard.py
- old one-off Slurm runners

No deletion or rewriting is authorized during foundation migration.

## 10. Study Modules

The dedicated study modules for Papers 2 through 4 contain many absolute
Talon paths.

These are not currently considered portability defects.

They are evidence-bearing study implementations and will be migrated only
during their dedicated migration stages.

## 11. TrustForge Migration Decision

TrustForge will not perform global path replacement.

Instead:

1. preserve historical path-bearing evidence;
2. define portable path and experiment contracts;
3. create TrustForge-native shared infrastructure;
4. route new and active work through that infrastructure;
5. migrate historical studies individually;
6. provide compatibility where historical reproduction requires old layouts.

## 12. Desired End State

New TrustForge execution should rely on logical roots such as:

TRUSTFORGE_DATA_ROOT

TRUSTFORGE_ARTIFACTS_ROOT

and explicit dataset/artifact manifests.

Historical records may continue to contain their original absolute paths.

Portability applies to future execution semantics, not to rewriting history.
