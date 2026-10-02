# M8.3 — TrustForge Paper Isolation Contracts

**Status:** `ISOLATION_BOUNDARIES_DEFINED`
**M8.2 acceptance commit:** `d42e022`

## Authority boundary

M8.3 defines dependency/isolation rules only.
It does not authorize file moves, runtime rewrites, or paper migrations.

## Principles

- `PAPER_SPECIFIC_SCIENCE != SHARED_FRAMEWORK_DEFAULT`
- `TRANSITIONAL_ALLOWED != PERMANENTLY_ALLOWED`
- `CROSS_PAPER_IMPORT != SHARED_CORE`
- `HISTORICAL_COMPATIBILITY != NATIVE_DEFAULT`
- `MODEL_PLUGIN != FRAMEWORK_CORE`

## Summary

Rules defined: **16**

| Dependency class | Rules |
|---|---:|
| `CORE_ALLOWED` | 4 |
| `DEFERRED` | 1 |
| `FORBIDDEN_CROSS_PAPER` | 3 |
| `HISTORICAL_COMPATIBILITY_ONLY` | 1 |
| `STUDY_LOCAL_ONLY` | 3 |
| `TRANSITIONAL_ALLOWED` | 4 |

## Isolation rules

### `P2-CORE-001`

- Paper: `paper02`
- Dependency class: `CORE_ALLOWED`
- Target: `src/trustforge/provenance/`
- Enforcement: `allowed`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 2 may depend on paper-neutral TrustForge provenance contracts.
- Evidence:
  - M8.2 classifies provenance as CORE_CONTRACT.

### `P2-CORE-002`

- Paper: `paper02`
- Dependency class: `CORE_ALLOWED`
- Target: `src/trustforge/storage/`
- Enforcement: `allowed`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 2 may depend on paper-neutral TrustForge storage/path contracts.
- Evidence:
  - M8.2 classifies storage resolution as CORE_CONTRACT.

### `P2-TRANS-001`

- Paper: `paper02`
- Dependency class: `TRANSITIONAL_ALLOWED`
- Target: `app/main.py`
- Enforcement: `allowed_with_migration_debt`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 2 historically invokes the shared CLI runtime; this remains temporary compatibility.
- Evidence:
  - M8.1 observed Paper 2 invocations of python -m app.main.
  - M8.2 classifies the historical shared CLI as TRANSITIONAL_RUNTIME.

### `P2-TRANS-002`

- Paper: `paper02`
- Dependency class: `TRANSITIONAL_ALLOWED`
- Target: `common/data.py`
- Enforcement: `allowed_with_migration_debt`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 2 historically imports shared dataset loading; this is not permanent core ownership.
- Evidence:
  - Paper 2 scripts directly import common.data.
  - M8.2 classifies historical shared dataset runtime as TRANSITIONAL_RUNTIME.

### `P2-LOCAL-001`

- Paper: `paper02`
- Dependency class: `STUDY_LOCAL_ONLY`
- Target: `gan/models_acgan.py;gan/train_acgan.py`
- Enforcement: `must_remain_explicitly_paper_owned`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: ACGAN and conditioning-intervention semantics are owned by Paper 2.
- Evidence:
  - M8.2 classifies Paper 2 conditioning semantics as STUDY_OWNED.

### `P2-XPAPER-001`

- Paper: `paper02`
- Dependency class: `FORBIDDEN_CROSS_PAPER`
- Target: `papers/paper3_when_does_synth_help/;papers/paper4_selective_synth_policies/`
- Enforcement: `forbidden`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 2 must not import implementation or scientific semantics from later studies.
- Evidence:
  - Paper-specific scientific meaning must remain isolated by study.

### `P3-CORE-001`

- Paper: `paper03`
- Dependency class: `CORE_ALLOWED`
- Target: `src/trustforge/provenance/;src/trustforge/storage/`
- Enforcement: `allowed`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 3 may depend on paper-neutral TrustForge contracts.
- Evidence:
  - M8.2 classifies storage and provenance as CORE_CONTRACT.

### `P3-TRANS-001`

- Paper: `paper03`
- Dependency class: `TRANSITIONAL_ALLOWED`
- Target: `app/main.py;eval/runner.py;common/data.py`
- Enforcement: `allowed_with_migration_debt`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 3 historically depends on the shared CLI/evaluator/data runtime.
- Evidence:
  - M8.1 observed extensive Paper 3 use of python -m app.main.
  - M8.2 classifies CLI, evaluator, and dataset runtime as TRANSITIONAL_RUNTIME.

### `P3-LOCAL-001`

- Paper: `paper03`
- Dependency class: `STUDY_LOCAL_ONLY`
- Target: `class-restricted synthesis;minority c4/c7 augmentation;budget-regime semantics`
- Enforcement: `must_remain_explicitly_paper_owned`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 3 scientific interventions must remain Paper 3-owned.
- Evidence:
  - M8.2 classifies Paper 3 augmentation semantics as STUDY_OWNED.

### `P3-XPAPER-001`

- Paper: `paper03`
- Dependency class: `FORBIDDEN_CROSS_PAPER`
- Target: `papers/paper2_conditional_generation_done_right/;papers/paper4_selective_synth_policies/`
- Enforcement: `forbidden`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 3 must not silently inherit Paper 2 conditioning or Paper 4 policy semantics.
- Evidence:
  - Paper-specific interventions are not framework defaults.

### `P4-CORE-001`

- Paper: `paper04`
- Dependency class: `CORE_ALLOWED`
- Target: `src/trustforge/provenance/;src/trustforge/storage/`
- Enforcement: `allowed`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 4 may depend on paper-neutral TrustForge contracts.
- Evidence:
  - M8.2 classifies storage and provenance as CORE_CONTRACT.

### `P4-TRANS-001`

- Paper: `paper04`
- Dependency class: `TRANSITIONAL_ALLOWED`
- Target: `app/main.py;eval/runner.py;common/data.py`
- Enforcement: `allowed_with_migration_debt`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 4 historically depends on shared runtime surfaces.
- Evidence:
  - M8.1 observed Paper 4 use of python -m app.main.
  - M8.2 classifies CLI/evaluator/data runtime as transitional.

### `P4-LOCAL-001`

- Paper: `paper04`
- Dependency class: `STUDY_LOCAL_ONLY`
- Target: `keep-all;confidence filtering;top-k;class-repair`
- Enforcement: `must_remain_explicitly_paper_owned`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Selective synthetic-data policy semantics are owned by Paper 4.
- Evidence:
  - M8.2 classifies Paper 4 policy semantics as STUDY_OWNED.

### `P4-XPAPER-001`

- Paper: `paper04`
- Dependency class: `FORBIDDEN_CROSS_PAPER`
- Target: `papers/paper2_conditional_generation_done_right/;papers/paper3_when_does_synth_help/`
- Enforcement: `forbidden`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper 4 must not import scientific implementation directly from Papers 2 or 3.
- Evidence:
  - Study ownership is distinct from shared framework ownership.

### `ALL-HIST-001`

- Paper: `all`
- Dependency class: `HISTORICAL_COMPATIBILITY_ONLY`
- Target: `historical configs;legacy Slurm scripts;historical path aliases`
- Enforcement: `compatibility_only`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Historical reproduction surfaces may be read/used for compatibility but must not become new native defaults.
- Evidence:
  - M8.2 classifies legacy path/config surfaces as HISTORICAL_COMPATIBILITY.

### `ALL-PLUGIN-001`

- Paper: `all`
- Dependency class: `DEFERRED`
- Target: `gan/;vae/;diffusion/;autoregressive/;gaussianmixture/;restrictedboltzmann/;maskedautoflow/`
- Enforcement: `deferred`
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Current model packages are plugins conceptually, but migration/isolation mechanics are deferred.
- Evidence:
  - M8.2 classifies historical model implementations as MODEL_PLUGIN.
