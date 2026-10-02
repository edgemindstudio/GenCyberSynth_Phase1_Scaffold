# M8.5 — TrustForge Old-Name / Old-Path Dependency Inventory

**Status:** `RENAME_DEPENDENCIES_INVENTORIED`
**M8.4 acceptance commit:** `a11445a`

## Authority boundary

This milestone inventories and classifies rename dependencies only.
It does not authorize repository rename, remote mutation, historical rewriting, paper migration, or compatibility removal.

## Principles

- `OLD_NAME_PRESENT != RENAME_BLOCKER`
- `HISTORICAL_PATH != CURRENT_CONFIGURATION`
- `HISTORICAL_REPOSITORY_IDENTITY != CURRENT_REPOSITORY_IDENTITY`
- `GCS_COMPATIBILITY != TRUSTFORGE_CANONICAL_IDENTITY`
- `PORTABILITY_BLOCKER != REPOSITORY_NAME_BLOCKER`
- `BULK_RENAME != GOVERNED_MIGRATION`

## Summary

Dependency groups: **16**
Rename blockers identified: **1**

| Classification | Groups |
|---|---:|
| `COMPATIBILITY_REFERENCE` | 2 |
| `DOCUMENTATION_ONLY` | 1 |
| `HISTORICAL_REFERENCE` | 4 |
| `PORTABILITY_BLOCKER` | 1 |
| `RENAME_BLOCKER` | 1 |
| `REQUIRES_REVIEW` | 2 |
| `SAFE_TO_RETAIN` | 3 |
| `TEST_ONLY` | 1 |
| `TOOLING_ONLY` | 1 |

## Rename blockers

- `M8.5-001`

## Dependency groups

### `M8.5-001`

- Classification: `RENAME_BLOCKER`
- Surface: Git remote origin
- Rename impact: A GitHub repository rename requires origin to resolve to the new canonical repository identity.
- Action before repository rename: Update origin only as part of the governed repository rename procedure after rename-readiness acceptance.
- Mutate now: `false`
- Rationale: This is the clearest live external identity tied directly to the old repository name.
- Examples:
  - `origin git@github.com:edgemindstudio/GenCyberSynth_Phase1_Scaffold.git`

### `M8.5-002`

- Classification: `HISTORICAL_REFERENCE`
- Surface: Paper 1 canonical experiment/study repository fields
- Rename impact: Historical provenance remains valid after repository rename.
- Action before repository rename: Preserve historical repository identity in accepted Paper 1 evidence; do not bulk-rewrite.
- Mutate now: `false`
- Rationale: Scientific execution/provenance identity must not be rewritten merely because the repository receives a new current name.
- Examples:
  - `studies/paper01_benchmark/experiments/** repository: edgemindstudio/GenCyberSynth_Phase1_Scaffold`
  - `studies/paper01_benchmark/study.yaml`
  - `studies/paper01_benchmark/lineage.yaml`
  - `schemas/examples/paper01/**`

### `M8.5-003`

- Classification: `HISTORICAL_REFERENCE`
- Surface: Paper 1 experiment generator repository constant
- Rename impact: The generator reproduces accepted historical Paper 1 contracts.
- Action before repository rename: Keep historical identity unless a future versioned generator explicitly separates historical_repository from current_repository.
- Mutate now: `false`
- Rationale: Changing this constant now would risk changing deterministic accepted Paper 1 contracts.
- Examples:
  - `scripts/generate_paper01_experiments.py: REPOSITORY = "edgemindstudio/GenCyberSynth_Phase1_Scaffold"`

### `M8.5-004`

- Classification: `HISTORICAL_REFERENCE`
- Surface: Paper 2 result/evidence absolute repository paths
- Rename impact: Paths may become non-resolving after local directory rename, but they remain historical evidence of where outputs were produced.
- Action before repository rename: Preserve in place. During Paper 2 migration, represent current portable locations separately from historical recorded paths.
- Mutate now: `false`
- Rationale: Observed absolute paths inside scientific outputs are evidence, not configuration defaults.
- Examples:
  - `papers/paper2_conditional_generation_done_right/results/tables/*.csv`
  - `papers/paper2_conditional_generation_done_right/results/tables/*.json`

### `M8.5-005`

- Classification: `PORTABILITY_BLOCKER`
- Surface: Paper 2 active scripts/configs/Slurm hard-coded external gencys roots
- Rename impact: These paths do not depend on the repository directory name, but they prevent portable execution and must be handled during Paper 2 migration.
- Action before repository rename: Do not change historical files now. Replace with TrustForge storage/path contracts in the future Paper 2 canonical migration.
- Mutate now: `false`
- Rationale: This is a portability problem rather than a repository-rename-name dependency.
- Examples:
  - `papers/paper2_conditional_generation_done_right/configs/**`
  - `papers/paper2_conditional_generation_done_right/scripts/**`
  - `papers/paper2_conditional_generation_done_right/slurm/**`

### `M8.5-006`

- Classification: `COMPATIBILITY_REFERENCE`
- Surface: gcs-core submodule and gcs_core imports
- Rename impact: The independent gcs-core identity does not inherently block renaming the parent repository.
- Action before repository rename: Retain until runtime/plugin migration explicitly replaces or retires the dependency.
- Mutate now: `false`
- Rationale: gcs-core is a separate compatibility/runtime dependency, not the parent repository identity.
- Examples:
  - `.gitmodules -> https://github.com/edgemindstudio/gcs-core.git`
  - `.github/workflows/smoke.yml`
  - `eval/runner.py`
  - `eval/val_common.py`
  - `model-template/**`

### `M8.5-007`

- Classification: `COMPATIBILITY_REFERENCE`
- Surface: Legacy GCS_* environment aliases
- Rename impact: Legacy aliases are intentionally supported alongside TRUSTFORGE_* variables.
- Action before repository rename: Retain through repository rename; retire only through a separately governed compatibility decision.
- Mutate now: `false`
- Rationale: Compatibility aliases are deliberate and already subordinate to canonical TRUSTFORGE_* variables.
- Examples:
  - `src/trustforge/storage/paths.py`
  - `scripts/trustforge_doctor.py`
  - `tests/trustforge/test_storage_paths.py`

### `M8.5-008`

- Classification: `SAFE_TO_RETAIN`
- Surface: External protected ~/gencys data/artifact root
- Rename impact: External storage root is independent of the repository directory name.
- Action before repository rename: Retain. Do not rename or reorganize protected historical storage as part of repository rename.
- Mutate now: `false`
- Rationale: The gencys storage identity is historical/external storage, not the Git repository name.
- Examples:
  - `/home/bruno.fonkeng/gencys/data`
  - `/home/bruno.fonkeng/gencys/artifacts*`
  - `artifacts -> /home/bruno.fonkeng/gencys/artifacts`

### `M8.5-009`

- Classification: `SAFE_TO_RETAIN`
- Surface: Dynamic repository-root discovery
- Rename impact: These mechanisms generally survive a repository directory rename because they derive location dynamically.
- Action before repository rename: Retain, while validating structural-depth assumptions during each paper migration.
- Mutate now: `false`
- Rationale: Dynamic location discovery is rename-safe unless directory depth changes.
- Examples:
  - `Path(__file__).resolve().parents[...]`
  - `git top-level discovery`
  - `Path.cwd()/--repo-root patterns`
  - `$(CURDIR)`
  - `$SLURM_SUBMIT_DIR`

### `M8.5-010`

- Classification: `DOCUMENTATION_ONLY`
- Surface: Current repository branding in README/Runbook/docs/comments
- Rename impact: Old branding does not block filesystem/Git rename but would make the renamed repository externally inconsistent.
- Action before repository rename: Update current-facing documentation and CLI branding during the governed rename transition, preserving explicitly historical references.
- Mutate now: `false`
- Rationale: These are presentation/current-identity surfaces rather than scientific evidence.
- Examples:
  - `README.md`
  - `Runbook.md`
  - `.github/git-commit-instructions.md`
  - `app/main.py descriptions/comments`
  - `common/data.py comments`
  - `adapters/base.py comments`

### `M8.5-011`

- Classification: `HISTORICAL_REFERENCE`
- Surface: Historical GenCyberSynth references in research lineage/architecture documentation
- Rename impact: Historical references remain semantically correct after rename.
- Action before repository rename: Preserve statements that explicitly describe the former GenCyberSynth identity; update only text that claims it is the current repository name.
- Mutate now: `false`
- Rationale: TrustForge architecture explicitly distinguishes historical GenCyberSynth identity from current framework identity.
- Examples:
  - `docs/RESEARCH_LINEAGE.md`
  - `docs/TRUSTFORGE_ARCHITECTURE.md`
  - `docs/STORAGE_AND_PATHS.md historical examples`

### `M8.5-012`

- Classification: `REQUIRES_REVIEW`
- Surface: Root Makefile current-facing GenCyberSynth branding and legacy defaults
- Rename impact: Branding should change eventually, while dependency/default lines may remain compatibility behavior.
- Action before repository rename: Split current-branding changes from compatibility semantics before rename; no bulk edit.
- Mutate now: `false`
- Rationale: One file contains both current-facing identity and historical/compatibility behavior.
- Examples:
  - `Makefile: "# Makefile — Unified main Makefile for GenCyberSynth"`
  - `Makefile: echo "GenCyberSynth unified Makefile"`
  - `Makefile: gcs-core schema path`
  - `Makefile: ~/gencys historical artifact example`

### `M8.5-013`

- Classification: `REQUIRES_REVIEW`
- Surface: model-template legacy product identity
- Rename impact: Template naming may communicate an obsolete product identity to new model/plugin projects.
- Action before repository rename: Review during plugin architecture normalization, not as an automatic repository rename substitution.
- Mutate now: `false`
- Rationale: The template combines package naming, dependency naming, citation identity, and job-name conventions.
- Examples:
  - `model-template/CITATION.cff`
  - `model-template/pyproject.toml`
  - `model-template/Makefile`
  - `model-template/scripts/slurm_array_example.sh`

### `M8.5-014`

- Classification: `SAFE_TO_RETAIN`
- Surface: gcs-core submodule URL
- Rename impact: Independent submodule URL is unaffected by parent repository rename.
- Action before repository rename: No action required for parent repository rename.
- Mutate now: `false`
- Rationale: Separate repository identity must not be renamed merely to match the parent.
- Examples:
  - `https://github.com/edgemindstudio/gcs-core.git`

### `M8.5-015`

- Classification: `TEST_ONLY`
- Surface: TrustForge tests referencing legacy aliases and historical identities
- Rename impact: Tests intentionally lock compatibility and historical behavior.
- Action before repository rename: Retain unless the governed behavior itself changes.
- Mutate now: `false`
- Rationale: Test references are not production dependency evidence by themselves.
- Examples:
  - `tests/trustforge/test_storage_paths.py`
  - `tests/trustforge/test_doctor.py`
  - `tests/trustforge/test_paper01_*`

### `M8.5-016`

- Classification: `TOOLING_ONLY`
- Surface: Migration/audit tooling references
- Rename impact: Tooling may reference both historical and canonical identities by design.
- Action before repository rename: Evaluate per tool; do not globally replace old-name strings.
- Mutate now: `false`
- Rationale: Audit tooling must often recognize legacy state in order to validate migration.
- Examples:
  - `tools/**`
  - `scripts/trustforge_doctor.py`
  - `scripts/trustforge_foundation_check.py`
