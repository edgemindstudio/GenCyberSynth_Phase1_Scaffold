# M8.2 — TrustForge Shared Core Boundary Definition

**Status:** `BOUNDARY_DEFINED`
**M8.1 baseline:** `d60f9c97e6b01b78cd615aa0fdb4aa61f3acce51`
**M8.1 acceptance commit:** `9487df0`

## Authority boundary

M8.2 defines architectural ownership only.
It does not authorize file moves, runtime rewrites, or paper migrations.

## Governing principles

- `MULTI_PAPER_SHARED_RUNTIME != SHARED_TRUSTFORGE_CORE`
- `REUSABLE != CORE`
- `INTERFACE_OWNERSHIP != CURRENT_IMPLEMENTATION_OWNERSHIP`
- `SCIENTIFIC_EXPERIMENT_IDENTITY != HISTORICAL_EXECUTION_IDENTITY`
- `HPC_BACKEND != SCIENTIFIC_EXPERIMENT_DEFINITION`
- `HISTORICAL_COMPATIBILITY != NATIVE_TRUSTFORGE_CORE`

## Summary

| Boundary class | Capabilities |
|---|---:|
| `CORE_CONTRACT` | 7 |
| `CORE_INTERFACE` | 6 |
| `DEFERRED` | 1 |
| `HISTORICAL_COMPATIBILITY` | 1 |
| `MODEL_PLUGIN` | 1 |
| `STUDY_OWNED` | 3 |
| `TRANSITIONAL_RUNTIME` | 4 |

## Capability decisions

### `study_identity_contract`

- Boundary class: `CORE_CONTRACT`
- Current role: Defines stable study identity and scientific scope.
- Future role: Permanent TrustForge study identity contract.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Study identity is framework-wide and not owned by one paper.
- Current locations:
  - `schemas/study.schema.yaml`
  - `docs/REPRODUCIBILITY_CONTRACT.md`
- Evidence:
  - STUDY is distinct from CODE, CONFIG, MANIFEST, and ARTIFACT.
  - Study identity is stable independently of manuscript filenames.

### `experiment_identity_contract`

- Boundary class: `CORE_CONTRACT`
- Current role: Defines exact scientific intent including dataset, model, seed, evaluation, and scientific parameters.
- Future role: Permanent TrustForge experiment-definition contract.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Experiment identity must remain paper-neutral and machine-independent.
- Current locations:
  - `schemas/experiment.schema.yaml`
  - `docs/REPRODUCIBILITY_CONTRACT.md`
- Evidence:
  - Experiment describes scientific intent.
  - Experiment identity is distinct from execution identity.

### `execution_identity_contract`

- Boundary class: `CORE_CONTRACT`
- Current role: Defines one concrete execution attempt.
- Future role: Permanent TrustForge execution-identity contract.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Execution identity is operational provenance and paper-neutral.
- Current locations:
  - `src/trustforge/provenance/execution.py`
  - `schemas/manifest.schema.yaml`
- Evidence:
  - Execution realizes an Experiment.
  - Execution identifiers are not scientific experiment identifiers.

### `storage_resolution`

- Boundary class: `CORE_CONTRACT`
- Current role: Resolves portable roots and canonical dataset/study/experiment/execution paths.
- Future role: Permanent TrustForge storage contract and implementation.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Storage semantics are already paper-neutral TrustForge code.
- Current locations:
  - `src/trustforge/storage/paths.py`
  - `src/trustforge/storage/layout.py`
- Evidence:
  - Scientific identity is separate from machine-specific paths.
  - Canonical paths derive from stable identifiers.

### `provenance_hashing`

- Boundary class: `CORE_CONTRACT`
- Current role: Provides SHA256 exact-file and canonical structured-data hashing.
- Future role: Permanent TrustForge provenance primitive.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Hashing semantics apply across all studies.
- Current locations:
  - `src/trustforge/provenance/hashing.py`
- Evidence:
  - Exact file hashes and canonical structured hashes answer different provenance questions.

### `git_provenance`

- Boundary class: `CORE_CONTRACT`
- Current role: Captures commit, branch, repository root, and dirty state read-only.
- Future role: Permanent TrustForge execution provenance primitive.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Git provenance is independent of study-specific science.
- Current locations:
  - `src/trustforge/provenance/git.py`
- Evidence:
  - Executions must record exact source identity.
  - Dirty state must not be represented as clean.

### `native_manifest_contract`

- Boundary class: `CORE_CONTRACT`
- Current role: Defines and validates execution reality.
- Future role: Permanent TrustForge manifest contract and native implementation.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Execution manifests are universal framework records.
- Current locations:
  - `schemas/manifest.schema.yaml`
  - `src/trustforge/provenance/manifest.py`
  - `src/trustforge/provenance/validation.py`
- Evidence:
  - CONFIG describes intended work; MANIFEST describes execution reality.
  - Manifest completion does not itself establish scientific acceptance.

### `accepted_evidence_contract`

- Boundary class: `DEFERRED`
- Current role: Paper 1 implementation separates production, evaluation, and authoritative results.
- Future role: Future paper-neutral accepted-evidence contract.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: The concept belongs in core, but current implementation is Paper 1-specific.
- Current locations:
  - `src/trustforge/paper01_execution_evidence.py`
  - `studies/paper01_benchmark/execution_evidence/`
- Evidence:
  - Operational completion is not accepted scientific evidence.
  - Paper 1 demonstrates producer/evaluator/result separation.

### `adapter_interface`

- Boundary class: `CORE_INTERFACE`
- Current role: Defines generic Adapter and synth(config)->manifest semantics.
- Future role: TrustForge-native model adapter interface.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: The abstraction is reusable even though the implementation is historical runtime.
- Current locations:
  - `adapters/base.py`
- Evidence:
  - Adapter defines a generic synthesis contract.

### `adapter_registry_interface`

- Boundary class: `CORE_INTERFACE`
- Current role: Registers and resolves adapters by model key.
- Future role: TrustForge-native plugin registry interface.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Registry semantics belong in core; historical registrations do not.
- Current locations:
  - `adapters/registry.py`
- Evidence:
  - Registry is implementation-agnostic at its public interface.

### `orchestration_interface`

- Boundary class: `CORE_INTERFACE`
- Current role: Routes train, synth, eval, configuration, paper scoping, and runtime metadata.
- Future role: TrustForge-native orchestration interface.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Train/synth/eval orchestration is core as an interface, not as the current CLI implementation.
- Current locations:
  - `app/main.py`
- Evidence:
  - Current app.main mixes generic orchestration with paper-era behavior.

### `evaluation_interface`

- Boundary class: `CORE_INTERFACE`
- Current role: Historical evaluator mixes metrics, manifests, paths, config interpretation, and paper-era behavior.
- Future role: TrustForge-native evaluator request/result interface.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Evaluation is cross-study, but the current monolith is not accepted wholesale as core.
- Current locations:
  - `eval/runner.py`
- Evidence:
  - eval/runner.py accumulated Paper 1, Paper 3, and Paper 4 behavior.

### `dataset_interface`

- Boundary class: `CORE_INTERFACE`
- Current role: Historical shared loading, normalization, label encoding, and dataset construction.
- Future role: TrustForge-native dataset representation/loading interface.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Dataset abstraction is core; current loader remains transitional.
- Current locations:
  - `common/data.py`
- Evidence:
  - Multiple historical model trainers import common.data.

### `execution_backend_interface`

- Boundary class: `CORE_INTERFACE`
- Current role: Historical Slurm scripts plus an architecture-level HPC contract.
- Future role: Backend-neutral execution plan with local and Slurm backends.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Execution backend abstraction is core; current Slurm scripts require later review.
- Current locations:
  - `docs/TRUSTFORGE_ARCHITECTURE.md`
  - `slurm/`
- Evidence:
  - HPC is an execution backend, not scientific experiment definition.

### `historical_shared_cli_runtime`

- Boundary class: `TRANSITIONAL_RUNTIME`
- Current role: Shared CLI incrementally modified across papers.
- Future role: Compatibility runtime during migration.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Shared historical use does not establish permanent core ownership.
- Current locations:
  - `app/main.py`
- Evidence:
  - Git history shows Paper 1, Paper 2, and Paper 3 changes.

### `historical_shared_evaluator`

- Boundary class: `TRANSITIONAL_RUNTIME`
- Current role: Cross-paper evaluator with accumulated historical semantics.
- Future role: Compatibility runtime until native evaluation contracts exist.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: The implementation is coupled to historical study semantics.
- Current locations:
  - `eval/runner.py`
- Evidence:
  - Git history includes Paper 1 fixes, Paper 3 changes, and Paper 4 behavior.

### `historical_shared_dataset_runtime`

- Boundary class: `TRANSITIONAL_RUNTIME`
- Current role: Shared historical loader consumed by multiple trainers.
- Future role: Compatibility runtime until native dataset interface exists.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Shared use does not prove permanent API suitability.
- Current locations:
  - `common/data.py`
- Evidence:
  - GAN, VAE, Diffusion, and Autoregressive trainers import common.data.

### `historical_concrete_adapters`

- Boundary class: `TRANSITIONAL_RUNTIME`
- Current role: Concrete bridges from shared CLI into historical model packages.
- Future role: Compatibility adapters or validated future plugins.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Concrete adapters are implementations, not the adapter abstraction.
- Current locations:
  - `adapters/gan_adapter.py`
  - `adapters/vae_adapter.py`
  - `adapters/diffusion_adapter.py`
  - `adapters/autoregressive_adapter.py`
  - `adapters/gaussianmixture_adapter.py`
  - `adapters/restrictedboltzmann_adapter.py`
  - `adapters/maskedautoflow_adapter.py`
- Evidence:
  - Concrete adapters import historical model-specific packages.

### `historical_model_implementations`

- Boundary class: `MODEL_PLUGIN`
- Current role: Seven historical generative-model implementations.
- Future role: Replaceable model plugins outside permanent core.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Reusable model code is a plugin, not framework core.
- Current locations:
  - `gan/`
  - `vae/`
  - `diffusion/`
  - `autoregressive/`
  - `gaussianmixture/`
  - `restrictedboltzmann/`
  - `maskedautoflow/`
- Evidence:
  - Model implementations are replaceable while TrustForge contracts remain stable.

### `paper02_conditioning_study`

- Boundary class: `STUDY_OWNED`
- Current role: Conditional-generation interventions and audits.
- Future role: Paper 2 study-owned implementation and evidence.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: These semantics answer Paper 2's scientific question.
- Current locations:
  - `papers/paper2_conditional_generation_done_right/`
  - `gan/models_acgan.py`
  - `gan/train_acgan.py`
- Evidence:
  - Paper 2 owns conditioning/class-faithfulness audit behavior.

### `paper03_augmentation_study`

- Boundary class: `STUDY_OWNED`
- Current role: Augmentation budgets, minority targeting, and class-restricted synthesis.
- Future role: Paper 3 study-owned implementation and evidence.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Paper-specific interventions must remain explicit study semantics.
- Current locations:
  - `papers/paper3_when_does_synth_help/`
  - `gan/sample.py`
  - `vae/sample.py`
  - `adapters/diffusion_adapter.py`
- Evidence:
  - Paper 3 owns augmentation regimes and minority-heavy c4/c7 behavior.

### `paper04_selective_policy_study`

- Boundary class: `STUDY_OWNED`
- Current role: Keep-all, confidence, top-k, and class-repair interventions.
- Future role: Paper 4 study-owned implementation and evidence.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Policy semantics belong to Paper 4 unless later generalized explicitly.
- Current locations:
  - `papers/paper4_selective_synth_policies/`
  - `eval/runner.py`
- Evidence:
  - Paper 4 treats synthetic inclusion as a policy-controlled intervention.

### `legacy_path_and_config_compatibility`

- Boundary class: `HISTORICAL_COMPATIBILITY`
- Current role: Historical reproduction and compatibility surfaces.
- Future role: Preserved compatibility until evidence permits retirement.
- Migration authorized: `false`
- Implementation move authorized: `false`
- Rationale: Historical path/config state may be provenance and must not be rewritten blindly.
- Current locations:
  - `configs/paper1_*.yaml`
  - `configs/paper2_500.yaml`
  - `configs/paper2_1000.yaml`
  - `configs/paper2_2000.yaml`
  - `slurm/legacy/`
  - `slurm/oneoffs/`
- Evidence:
  - Migration doctrine preserves historical path-bearing evidence.
