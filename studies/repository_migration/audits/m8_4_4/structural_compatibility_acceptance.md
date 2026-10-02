# M8.4.4 — TrustForge Structural Compatibility Acceptance

**Status:** `STRUCTURAL_COMPATIBILITY_ACCEPTED`
**M8.4.3 acceptance commit:** `f1db89a`
**M8.4.2 readiness commit:** `954ad5f`

## Acceptance

The M8.4.3 scaffold materialization is accepted as structurally compatible.

It changed repository structure only within the previously authorized scaffold scope and did not transfer scientific authority, migrate Papers 2–4, replace runtime imports, or alter historical execution paths.

## Acceptance claims

- Exact materialization scope: `true`
- Historical authority roots present: `true`
- Historical protected paths unchanged: `true`
- Runtime import migration observed: `false`
- Execution-path migration observed: `false`
- Historical authority transfer observed: `false`
- Full TrustForge suite green: `true`

## Principles

- `SCAFFOLD_EXISTENCE != AUTHORITY_TRANSFER`
- `REFERENCE_IN_AUDIT != RUNTIME_DEPENDENCY`
- `REFERENCE_IN_TEST != EXECUTION_PATH`
- `NO_HISTORICAL_DIFF == HISTORICAL_AUTHORITY_PRESERVED`
- `STRUCTURAL_COMPATIBILITY_ACCEPTANCE != PAPER_MIGRATION_ACCEPTANCE`

## Materialization scope

- `hpc/README.md`
- `studies/paper02_conditioning_audit/README.md`
- `studies/paper03_augmentation_regimes/README.md`
- `studies/paper04_selective_policies/README.md`
- `tests/trustforge/test_safe_scaffold_materialization_m8_4_3.py`
- `tools/materialize_safe_scaffolds_m8_4_3.py`

## Historical authority roots preserved

- `slurm`
- `papers/paper2_conditional_generation_done_right`
- `papers/paper3_when_does_synth_help`
- `papers/paper4_selective_synth_policies`

## Protected historical paths unchanged

- `slurm`
- `run_gcs.slurm`
- `run_suite.slurm`
- `run_tuning.slurm`
- `papers/paper2_conditional_generation_done_right`
- `papers/paper3_when_does_synth_help`
- `papers/paper4_selective_synth_policies`

## Validation observed

- TrustForge tests passed: **658**
- TrustForge subtests passed: **3**
- `git diff --check` clean: `true`

## Authority boundary

This acceptance closes structural compatibility for M8.4.3 only.
It does not authorize Paper 2–4 migration, runtime replacement, plugin relocation, historical cleanup, or repository rename.
