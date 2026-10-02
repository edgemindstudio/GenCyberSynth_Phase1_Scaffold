# M8.4.1 — TrustForge Repository Structural Normalization Plan

**Status:** `STRUCTURAL_DESIGN_DEFINED`
**M8.3 acceptance commit:** `c2e06fa`

## Authority boundary

This milestone defines normalized structural roles only.
No file move, rename, import rewrite, paper migration, or historical cleanup is authorized.

## Principles

- `NORMALIZED_TARGET != IMMEDIATE_MOVE`
- `COMPATIBILITY_SURFACE != NATIVE_CORE`
- `STUDY_LOCAL_SCIENCE_STAYS_WITH_STUDY`
- `MODEL_IMPLEMENTATION == PLUGIN`
- `INTERFACE_FIRST_BEFORE_IMPLEMENTATION_MIGRATION`
- `HISTORICAL_EVIDENCE_REMAINS_PRESERVED`

## Target repository shape

- `src/trustforge/`
- `studies/paper01_benchmark/`
- `studies/paper02_conditioning_audit/`
- `studies/paper03_augmentation_regimes/`
- `studies/paper04_selective_policies/`
- `configs/`
- `manifests/`
- `schemas/`
- `hpc/`
- `scripts/`
- `tools/`
- `tests/`
- `docs/`

## Summary

Components classified: **37**

| Normalization class | Components |
|---|---:|
| `COMPATIBILITY_ONLY` | 6 |
| `CREATE_NATIVE` | 7 |
| `DEFERRED` | 4 |
| `MIGRATE_LATER` | 5 |
| `PLUGIN` | 7 |
| `PRESERVE_CURRENT` | 8 |
| `STUDY_LOCAL` | 0 |

## Component decisions

### `trustforge_provenance`

- Current location: `src/trustforge/provenance/`
- Normalization class: `PRESERVE_CURRENT`
- Target role: Permanent paper-neutral provenance core.
- Target location: `src/trustforge/provenance/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: M8.2 already accepted this implementation as CORE_CONTRACT.
- Evidence:
  - Current TrustForge provenance package contains execution, Git, hashing, manifest, and validation primitives.

### `trustforge_storage`

- Current location: `src/trustforge/storage/`
- Normalization class: `PRESERVE_CURRENT`
- Target role: Permanent paper-neutral storage/path core.
- Target location: `src/trustforge/storage/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: M8.2 already accepted this implementation as CORE_CONTRACT.
- Evidence:
  - Current storage package resolves portable roots and canonical artifact layout.

### `trustforge_data_namespace`

- Current location: `src/trustforge/data/`
- Normalization class: `CREATE_NATIVE`
- Target role: Paper-neutral dataset identity/loading interfaces.
- Target location: `src/trustforge/data/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Namespace exists but contains only package scaffolding.
- Evidence:
  - M8.2 defines dataset abstraction as CORE_INTERFACE while common/data.py remains transitional.

### `trustforge_evaluation_namespace`

- Current location: `src/trustforge/evaluation/`
- Normalization class: `CREATE_NATIVE`
- Target role: Paper-neutral evaluator contracts and native implementations.
- Target location: `src/trustforge/evaluation/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Namespace exists but historical eval/runner.py is not accepted as core implementation.
- Evidence:
  - M8.2 classifies evaluation abstraction as CORE_INTERFACE and eval/runner.py as TRANSITIONAL_RUNTIME.

### `trustforge_orchestration_namespace`

- Current location: `src/trustforge/orchestration/`
- Normalization class: `CREATE_NATIVE`
- Target role: Paper-neutral train/synth/eval orchestration contracts.
- Target location: `src/trustforge/orchestration/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Namespace exists but current app/main.py remains transitional compatibility runtime.
- Evidence:
  - M8.2 distinguishes orchestration interface from current historical CLI implementation.

### `trustforge_models_namespace`

- Current location: `src/trustforge/models/`
- Normalization class: `CREATE_NATIVE`
- Target role: Plugin/adapter interfaces and model registration contracts, not model implementations.
- Target location: `src/trustforge/models/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Framework core should own plugin contracts while implementations remain plugins.
- Evidence:
  - M8.2 classifies model implementations as MODEL_PLUGIN rather than core.

### `trustforge_policies_namespace`

- Current location: `src/trustforge/policies/`
- Normalization class: `CREATE_NATIVE`
- Target role: Future paper-neutral policy interfaces only.
- Target location: `src/trustforge/policies/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Namespace exists but Paper 4 policy semantics remain study-owned.
- Evidence:
  - M8.2 classifies Paper 4 selective policies as STUDY_OWNED.

### `trustforge_augmentation_namespace`

- Current location: `src/trustforge/augmentation/`
- Normalization class: `CREATE_NATIVE`
- Target role: Future paper-neutral augmentation interfaces only.
- Target location: `src/trustforge/augmentation/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Namespace exists but Paper 3 augmentation semantics remain study-owned.
- Evidence:
  - M8.2 classifies Paper 3 augmentation behavior as STUDY_OWNED.

### `trustforge_uncertainty_namespace`

- Current location: `src/trustforge/uncertainty/`
- Normalization class: `DEFERRED`
- Target role: Future uncertainty/calibration capability.
- Target location: `src/trustforge/uncertainty/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Architecture reserves this area, but current Papers 1–4 migration does not yet require implementation.
- Evidence:
  - Namespace currently contains only __init__.py.

### `paper01_native_study`

- Current location: `studies/paper01_benchmark/`
- Normalization class: `PRESERVE_CURRENT`
- Target role: Canonical TrustForge-native Paper 1 study/evidence package.
- Target location: `studies/paper01_benchmark/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Paper 1 migration is complete and accepted.
- Evidence:
  - Current study contains canonical experiments, execution evidence, and audit history.

### `paper02_study`

- Current location: `papers/paper2_conditional_generation_done_right/`
- Normalization class: `MIGRATE_LATER`
- Target role: Paper 2 canonical TrustForge study package.
- Target location: `studies/paper02_conditioning_audit/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Paper 2 has rich configs/results/scripts/slurm structure but has not yet been migrated.
- Evidence:
  - Tracked structure includes configs, manifests, notes, paper, results, scripts, and slurm.

### `paper03_study`

- Current location: `papers/paper3_when_does_synth_help/`
- Normalization class: `MIGRATE_LATER`
- Target role: Paper 3 canonical TrustForge study package.
- Target location: `studies/paper03_augmentation_regimes/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Paper 3 has rich configs/results/scripts/slurm structure but remains historical active study.
- Evidence:
  - Tracked structure includes 82 configs, 122 results, 12 scripts, and 38 slurm files.

### `paper04_study`

- Current location: `papers/paper4_selective_synth_policies/`
- Normalization class: `MIGRATE_LATER`
- Target role: Paper 4 canonical TrustForge study package.
- Target location: `studies/paper04_selective_policies/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Paper 4 has rich configs/results/scripts/slurm structure but remains historical active study.
- Evidence:
  - Tracked structure includes configs, results, scripts, and slurm.

### `paper03_placeholder`

- Current location: `papers/paper03_when_does_synth_help/`
- Normalization class: `COMPATIBILITY_ONLY`
- Target role: Historical/scaffold placeholder retained until cleanup is explicitly authorized.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: M8.1 classified this location as scaffold-only, distinct from active Paper 3.
- Evidence:
  - Active Paper 3 is papers/paper3_when_does_synth_help/.

### `paper05_placeholder`

- Current location: `papers/paper05_shift_calibration/`
- Normalization class: `DEFERRED`
- Target role: Future study placeholder outside current Papers 1–4 migration.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Paper 5 is outside current migration scope.
- Evidence:
  - M8.1 classified this location as scaffold-only.

### `historical_cli_runtime`

- Current location: `app/`
- Normalization class: `COMPATIBILITY_ONLY`
- Target role: Historical orchestration compatibility surface.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: M8.2/M8.3 explicitly classify current CLI runtime as transitional, not native core.
- Evidence:
  - Papers 2–4 historically invoke python -m app.main.

### `historical_adapters`

- Current location: `adapters/`
- Normalization class: `COMPATIBILITY_ONLY`
- Target role: Historical concrete adapter compatibility surface.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Concrete adapters remain tied to historical model packages.
- Evidence:
  - M8.2 classifies adapter interface as core but concrete adapters as TRANSITIONAL_RUNTIME.

### `historical_dataset_runtime`

- Current location: `common/`
- Normalization class: `COMPATIBILITY_ONLY`
- Target role: Historical dataset/runtime compatibility surface.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: common/data.py is multi-paper shared runtime, not accepted native core.
- Evidence:
  - Papers/models import common.data directly.

### `historical_evaluator`

- Current location: `eval/`
- Normalization class: `COMPATIBILITY_ONLY`
- Target role: Historical evaluation compatibility surface.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: eval/runner.py mixes cross-paper and paper-era semantics.
- Evidence:
  - M8.2 classifies historical evaluator as TRANSITIONAL_RUNTIME.

### `gan_plugin`

- Current location: `gan/`
- Normalization class: `PLUGIN`
- Target role: GAN family implementation plugin.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Model implementation is reusable but not framework core.
- Evidence:
  - M8.2 classifies historical model implementations as MODEL_PLUGIN.

### `vae_plugin`

- Current location: `vae/`
- Normalization class: `PLUGIN`
- Target role: VAE family implementation plugin.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Model implementation is reusable but not framework core.
- Evidence:
  - M8.2 classifies historical model implementations as MODEL_PLUGIN.

### `diffusion_plugin`

- Current location: `diffusion/`
- Normalization class: `PLUGIN`
- Target role: Diffusion family implementation plugin.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Model implementation is reusable but not framework core.
- Evidence:
  - M8.2 classifies historical model implementations as MODEL_PLUGIN.

### `autoregressive_plugin`

- Current location: `autoregressive/`
- Normalization class: `PLUGIN`
- Target role: Autoregressive family implementation plugin.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Model implementation is reusable but not framework core.
- Evidence:
  - M8.2 classifies historical model implementations as MODEL_PLUGIN.

### `gaussianmixture_plugin`

- Current location: `gaussianmixture/`
- Normalization class: `PLUGIN`
- Target role: Gaussian-mixture implementation plugin.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Model implementation is reusable but not framework core.
- Evidence:
  - M8.2 classifies historical model implementations as MODEL_PLUGIN.

### `restrictedboltzmann_plugin`

- Current location: `restrictedboltzmann/`
- Normalization class: `PLUGIN`
- Target role: Restricted Boltzmann implementation plugin.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Model implementation is reusable but not framework core.
- Evidence:
  - M8.2 classifies historical model implementations as MODEL_PLUGIN.

### `maskedautoflow_plugin`

- Current location: `maskedautoflow/`
- Normalization class: `PLUGIN`
- Target role: Masked AutoFlow implementation plugin.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Model implementation is reusable but not framework core.
- Evidence:
  - M8.2 classifies historical model implementations as MODEL_PLUGIN.

### `generic_schemas`

- Current location: `schemas/`
- Normalization class: `PRESERVE_CURRENT`
- Target role: Repository-level TrustForge contract schemas.
- Target location: `schemas/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Generic study/experiment/manifest schemas are already accepted shared core contracts.
- Evidence:
  - Paper 1 examples remain explicitly separated under schemas/examples/paper01/.

### `manifests`

- Current location: `manifests/`
- Normalization class: `PRESERVE_CURRENT`
- Target role: Repository-level manifest schemas/registry area.
- Target location: `manifests/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Existing Paper 1 schema material is explicit and tracked.
- Evidence:
  - M8.1 distinguished generic schemas from Paper 1-specific manifest schema.

### `configs_root`

- Current location: `configs/`
- Normalization class: `MIGRATE_LATER`
- Target role: Future repository-level paper-neutral configuration area after mixed historical content is partitioned.
- Target location: `configs/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: The current root is mixed: some files are historical compatibility while the normalized repository still requires a native configs/ surface.
- Evidence:
  - M8.1 left many root configs ambiguous or historical compatibility.
  - M8.3 requires historical configs to remain compatibility-only rather than become native defaults.
  - The target architecture retains configs/ as a normalized repository-level surface.

### `slurm_root`

- Current location: `slurm/;run_gcs.slurm;run_suite.slurm;run_tuning.slurm`
- Normalization class: `COMPATIBILITY_ONLY`
- Target role: Historical/global HPC compatibility area.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Root Slurm contains legacy, one-off, and historical shared runners.
- Evidence:
  - M8.3 allows historical Slurm only as compatibility, not native default.

### `hpc_native_root`

- Current location: `hpc/`
- Normalization class: `CREATE_NATIVE`
- Target role: Future TrustForge-native backend-neutral/local/Slurm execution support.
- Target location: `hpc/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Target architecture calls for hpc/ and M8.2 defines execution backend as a core interface.
- Evidence:
  - No native hpc/ directory currently exists.

### `scripts_root`

- Current location: `scripts/`
- Normalization class: `MIGRATE_LATER`
- Target role: Repository-level general automation only; study-specific scripts should live with studies.
- Target location: `scripts/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Current scripts mix generic utilities, plots, metrics, and Paper 1-era functionality.
- Evidence:
  - M8.1 classified many scripts as ambiguous and some as Paper 1-specific.

### `tools_root`

- Current location: `tools/`
- Normalization class: `PRESERVE_CURRENT`
- Target role: Repository engineering/audit tooling.
- Target location: `tools/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: TrustForge migration and audit tooling already lives here.
- Evidence:
  - M8.1/M8.2/M8.3 generators are repository tooling, not scientific study code.

### `tests_root`

- Current location: `tests/`
- Normalization class: `PRESERVE_CURRENT`
- Target role: Repository-level tests, including TrustForge contract tests.
- Target location: `tests/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Current tests/trustforge suite validates foundation and migration contracts.
- Evidence:
  - Current TrustForge suite contains migration milestone tests.

### `docs_root`

- Current location: `docs/`
- Normalization class: `PRESERVE_CURRENT`
- Target role: Architecture, portability, reproducibility, and lineage documentation.
- Target location: `docs/`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: Current docs define accepted TrustForge architecture and research lineage.
- Evidence:
  - M8.1 classified core architecture docs as SHARED_TRUSTFORGE_CORE.

### `model_template`

- Current location: `model-template/`
- Normalization class: `DEFERRED`
- Target role: Developer/template infrastructure subject to later review.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: M8.1 classified it as operational infrastructure, not scientific core.
- Evidence:
  - No migration action has yet been authorized.

### `gcs_core_submodule`

- Current location: `gcs-core/`
- Normalization class: `DEFERRED`
- Target role: Legacy/external component requiring dedicated review.
- Target location: `deferred`
- Move authorized: `false`
- Rename authorized: `false`
- Rewrite authorized: `false`
- Rationale: M8.1 left this component ambiguous.
- Evidence:
  - Current repository tracks gcs-core as a separate top-level component.
