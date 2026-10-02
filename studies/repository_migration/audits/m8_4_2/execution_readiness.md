# M8.4.2 — TrustForge Normalization Execution Readiness

**Status:** `EXECUTION_READINESS_DEFINED`
**M8.4.1 acceptance commit:** `22f3df2`

## Authority boundary

M8.4.2 authorizes only explicitly identified safe scaffold creation.
It does not authorize historical moves, import rewrites, paper migrations, plugin relocation, or cleanup.

## Principles

- `SCAFFOLD_CREATION != MIGRATION`
- `DESTINATION_EXISTENCE != AUTHORITY_TRANSFER`
- `HISTORICAL_SOURCE_REMAINS_AUTHORITATIVE_UNTIL_MIGRATION_ACCEPTED`
- `NATIVE_NAMESPACE_EXISTENCE != IMPLEMENTATION_MIGRATION`
- `COMPATIBILITY_GUARD_PRECEDES_RUNTIME_REPLACEMENT`

## Summary

Actions classified: **21**

| Readiness class | Actions |
|---|---:|
| `AUTHORIZED_SAFE_SCAFFOLD` | 4 |
| `DEFERRED` | 1 |
| `GUARDED_COMPATIBILITY` | 6 |
| `MIGRATION_NOT_YET_AUTHORIZED` | 4 |
| `PRESERVE_AS_IS` | 6 |

## Actions

### `SAFE-001`

- Target: `src/trustforge/data/`
- Readiness class: `PRESERVE_AS_IS`
- Action type: `existing_native_scaffold`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The native namespace already exists and contains only tracked package scaffolding.
- Preconditions:
  - No implementation migration is required to preserve this namespace.
- Evidence:
  - Only src/trustforge/data/__init__.py is tracked.
  - No tracked Python import currently references trustforge.data.

### `SAFE-002`

- Target: `src/trustforge/evaluation/`
- Readiness class: `PRESERVE_AS_IS`
- Action type: `existing_native_scaffold`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The native namespace already exists and is not yet coupled to runtime behavior.
- Preconditions:
  - Historical eval/runner.py must remain untouched.
- Evidence:
  - Only src/trustforge/evaluation/__init__.py is tracked.
  - No tracked Python import currently references trustforge.evaluation.

### `SAFE-003`

- Target: `src/trustforge/orchestration/`
- Readiness class: `PRESERVE_AS_IS`
- Action type: `existing_native_scaffold`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The native namespace already exists without replacing app/main.py.
- Preconditions:
  - Historical app/main.py remains compatibility runtime.
- Evidence:
  - Only src/trustforge/orchestration/__init__.py is tracked.
  - No tracked Python import currently references trustforge.orchestration.

### `SAFE-004`

- Target: `src/trustforge/models/`
- Readiness class: `PRESERVE_AS_IS`
- Action type: `existing_native_scaffold`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The native namespace exists for future plugin contracts without moving model implementations.
- Preconditions:
  - Historical model packages remain in place.
- Evidence:
  - Only src/trustforge/models/__init__.py is tracked.
  - No tracked Python import currently references trustforge.models.

### `SAFE-005`

- Target: `src/trustforge/policies/`
- Readiness class: `PRESERVE_AS_IS`
- Action type: `existing_native_scaffold`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The namespace exists without absorbing Paper 4 policy semantics.
- Preconditions:
  - Paper 4 policy implementations remain study-owned.
- Evidence:
  - Only src/trustforge/policies/__init__.py is tracked.
  - No tracked Python import currently references trustforge.policies.

### `SAFE-006`

- Target: `src/trustforge/augmentation/`
- Readiness class: `PRESERVE_AS_IS`
- Action type: `existing_native_scaffold`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The namespace exists without absorbing Paper 3 augmentation semantics.
- Preconditions:
  - Paper 3 augmentation implementations remain study-owned.
- Evidence:
  - Only src/trustforge/augmentation/__init__.py is tracked.
  - No tracked Python import currently references trustforge.augmentation.

### `SAFE-007`

- Target: `hpc/`
- Readiness class: `AUTHORIZED_SAFE_SCAFFOLD`
- Action type: `create_directory_scaffold`
- Authorized: `true`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The native hpc/ root is part of accepted target architecture and does not currently exist.
- Preconditions:
  - Creation must contain no migrated historical Slurm scripts.
  - Creation must not change root slurm/ behavior.
  - Creation must not change execution semantics.
- Evidence:
  - docs/TRUSTFORGE_ARCHITECTURE.md already names hpc/ in the target structure.
  - M8.4.1 target structure includes hpc/.
  - No native hpc/ directory currently exists.

### `SAFE-008`

- Target: `studies/paper02_conditioning_audit/`
- Readiness class: `AUTHORIZED_SAFE_SCAFFOLD`
- Action type: `create_directory_scaffold`
- Authorized: `true`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The canonical Paper 2 study destination is accepted, but migration is not.
- Preconditions:
  - Scaffold must contain no copied/moved historical Paper 2 files.
  - Historical papers/paper2_conditional_generation_done_right/ remains authoritative.
- Evidence:
  - M8.4.1 defines studies/paper02_conditioning_audit/ as the future target.
  - The destination is currently absent.
  - Current references are planning/test references only.

### `SAFE-009`

- Target: `studies/paper03_augmentation_regimes/`
- Readiness class: `AUTHORIZED_SAFE_SCAFFOLD`
- Action type: `create_directory_scaffold`
- Authorized: `true`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The canonical Paper 3 study destination is accepted, but migration is not.
- Preconditions:
  - Scaffold must contain no copied/moved historical Paper 3 files.
  - Historical papers/paper3_when_does_synth_help/ remains authoritative.
- Evidence:
  - M8.4.1 defines studies/paper03_augmentation_regimes/ as the future target.
  - The destination is currently absent.
  - Current references are planning/test references only.

### `SAFE-010`

- Target: `studies/paper04_selective_policies/`
- Readiness class: `AUTHORIZED_SAFE_SCAFFOLD`
- Action type: `create_directory_scaffold`
- Authorized: `true`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: The canonical Paper 4 study destination is accepted, but migration is not.
- Preconditions:
  - Scaffold must contain no copied/moved historical Paper 4 files.
  - Historical papers/paper4_selective_synth_policies/ remains authoritative.
- Evidence:
  - M8.4.1 defines studies/paper04_selective_policies/ as the future target.
  - The destination is currently absent.
  - Current references are planning/test references only.

### `GUARD-001`

- Target: `app/`
- Readiness class: `GUARDED_COMPATIBILITY`
- Action type: `historical_runtime`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Shared CLI remains active compatibility runtime for Papers 2–4.
- Preconditions:
  - No relocation or import rewrite before paper migration contracts exist.
- Evidence:
  - M8.2 classifies app/main.py as TRANSITIONAL_RUNTIME.
  - M8.3 allows it only with migration debt.

### `GUARD-002`

- Target: `adapters/`
- Readiness class: `GUARDED_COMPATIBILITY`
- Action type: `historical_runtime`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Concrete adapters remain bound to historical model packages.
- Preconditions:
  - No relocation into src/trustforge/models/ yet.
- Evidence:
  - M8.2 separates adapter interface ownership from current concrete adapters.

### `GUARD-003`

- Target: `common/`
- Readiness class: `GUARDED_COMPATIBILITY`
- Action type: `historical_runtime`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Historical dataset runtime remains shared compatibility code.
- Preconditions:
  - No replacement by trustforge.data until consumer migrations are proven.
- Evidence:
  - M8.3 allows common/data.py only as transitional dependency.

### `GUARD-004`

- Target: `eval/`
- Readiness class: `GUARDED_COMPATIBILITY`
- Action type: `historical_runtime`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Historical evaluator remains compatibility runtime with paper-era semantics.
- Preconditions:
  - No replacement by trustforge.evaluation until evaluator contracts are implemented and paper migrations validated.
- Evidence:
  - M8.2 classifies eval/runner.py as TRANSITIONAL_RUNTIME.

### `GUARD-005`

- Target: `configs/`
- Readiness class: `GUARDED_COMPATIBILITY`
- Action type: `mixed_root`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Current root is mixed historical/native candidate space and must be partitioned later.
- Preconditions:
  - No historical config may be silently reclassified as native default.
- Evidence:
  - M8.4.1 classifies configs/ as MIGRATE_LATER.

### `GUARD-006`

- Target: `slurm/;run_gcs.slurm;run_suite.slurm;run_tuning.slurm`
- Readiness class: `GUARDED_COMPATIBILITY`
- Action type: `historical_hpc`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Historical/global Slurm surfaces remain compatibility infrastructure.
- Preconditions:
  - No script is moved into hpc/ merely because hpc/ now exists.
- Evidence:
  - M8.3 classifies historical Slurm usage as compatibility-only.

### `BLOCK-001`

- Target: `papers/paper2_conditional_generation_done_right/`
- Readiness class: `MIGRATION_NOT_YET_AUTHORIZED`
- Action type: `paper_migration`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Paper 2 historical study has not yet undergone canonical migration.
- Preconditions:
  - Paper 2 migration requires dedicated evidence mapping and compatibility validation.
- Evidence:
  - M8.4.1 classifies Paper 2 as MIGRATE_LATER.

### `BLOCK-002`

- Target: `papers/paper3_when_does_synth_help/`
- Readiness class: `MIGRATION_NOT_YET_AUTHORIZED`
- Action type: `paper_migration`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Paper 3 historical study has not yet undergone canonical migration.
- Preconditions:
  - Paper 3 migration requires dedicated evidence mapping and compatibility validation.
- Evidence:
  - M8.4.1 classifies Paper 3 as MIGRATE_LATER.

### `BLOCK-003`

- Target: `papers/paper4_selective_synth_policies/`
- Readiness class: `MIGRATION_NOT_YET_AUTHORIZED`
- Action type: `paper_migration`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Paper 4 historical study has not yet undergone canonical migration.
- Preconditions:
  - Paper 4 migration requires dedicated evidence mapping and compatibility validation.
- Evidence:
  - M8.4.1 classifies Paper 4 as MIGRATE_LATER.

### `BLOCK-004`

- Target: `gan/;vae/;diffusion/;autoregressive/;gaussianmixture/;restrictedboltzmann/;maskedautoflow/`
- Readiness class: `MIGRATION_NOT_YET_AUTHORIZED`
- Action type: `plugin_migration`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Model packages are conceptually plugins but relocation mechanics remain unresolved.
- Preconditions:
  - Plugin interface must be implemented before any model package relocation.
- Evidence:
  - M8.2 classifies model implementations as MODEL_PLUGIN.
  - M8.3 explicitly defers plugin migration mechanics.

### `DEFER-001`

- Target: `src/trustforge/uncertainty/`
- Readiness class: `DEFERRED`
- Action type: `future_capability`
- Authorized: `false`
- Historical move authorized: `false`
- Import rewrite authorized: `false`
- Paper migration authorized: `false`
- Rationale: Uncertainty/calibration is outside current Papers 1–4 normalization needs.
- Preconditions:
  - No implementation work required in M8.4.2.
- Evidence:
  - M8.4.1 classifies uncertainty namespace as DEFERRED.
