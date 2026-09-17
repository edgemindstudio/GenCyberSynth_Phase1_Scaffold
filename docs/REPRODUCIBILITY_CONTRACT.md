# TrustForge Reproducibility Contract

## Status

This document defines the reproducibility model for TrustForge research.

It establishes the distinction between:

- a study;
- an experiment;
- an execution;
- a manifest;
- an artifact;
- accepted scientific evidence.

The contract applies to future TrustForge-native research and guides migration
of historical studies.

It does not modify historical experiments or evidence.

---

## 1. Core Principle

A scientific claim must be traceable to an exact experimental identity.

TrustForge must be able to answer:

- What study produced this result?
- What experiment definition was used?
- What code version executed?
- What dataset representation was used?
- What random seed was used?
- What environment executed the experiment?
- What artifacts were produced?
- Which artifacts were accepted as scientific evidence?

A directory name alone is not sufficient scientific provenance.

---

## 2. Study

A Study represents a scientific investigation.

Examples:

paper01_benchmark

paper02_conditioning_audit

paper03_augmentation_regimes

paper04_selective_policies

A Study may contain many experiments.

A Study defines the scientific question, scope, and interpretation boundary.

It should have a stable identifier.

Example:

study_id: paper03_augmentation_regimes

A Study is not the same thing as a manuscript filename.

The manuscript may evolve while the scientific study identity remains stable.

---

## 3. Experiment

An Experiment is an exact scientific definition of work to be performed.

An experiment should identify, where relevant:

- study;
- experiment ID;
- dataset;
- dataset representation;
- split;
- model or generator;
- seed;
- preprocessing;
- augmentation regime;
- augmentation budget;
- policy;
- evaluator;
- metrics;
- training parameters;
- scientific configuration.

Example:

study_id: paper03_augmentation_regimes

experiment_id:
ustc_gan_minority_c4c7_b500_seed42

An experiment definition describes scientific intent.

It should not depend on a particular machine.

---

## 4. Execution

An Execution is one concrete attempt to run an Experiment.

The same Experiment may be executed:

- on Talon;
- on a personal machine;
- after interruption;
- after scheduler retry;
- under a reproduction test.

Each execution receives a unique execution identity.

Example:

execution_id:
20260917T143000Z_talon_job123456

Multiple executions do not create multiple scientific experiments unless the
scientific definition changes.

---

## 5. Configuration Identity

Every serious experiment should have a stable configuration representation.

TrustForge should calculate a configuration hash from the normalized
scientific configuration.

Example:

config_sha256:
<hash>

The configuration hash identifies the exact experiment definition.

Machine-specific runtime fields should not silently change the scientific
configuration hash.

---

## 6. Code Identity

Each execution must record the exact source-code identity used.

At minimum:

- Git commit;
- repository identity;
- working-tree state.

Preferred fields:

git_commit

git_branch

git_dirty

repository

release_tag

A dirty working tree must never be silently represented as a clean committed
state.

For authoritative scientific runs, a clean committed state is preferred.

---

## 7. Dataset Identity

Dataset identity must be separate from dataset path.

A dataset definition should eventually contain:

- dataset ID;
- source dataset;
- preparation stage;
- representation;
- shape;
- class mapping;
- preprocessing identity;
- split identity;
- checksum or manifest hash when practical.

Example:

dataset_id:
ustc_tfc2016_malware_nhwc_v1

The execution manifest may additionally record:

resolved_path:
/home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware_nhwc

The path records execution reality.

The dataset ID records scientific identity.

---

## 8. Seed Identity

Random seeds are part of scientific experiment identity whenever they affect
results.

Seed values must be explicit.

Example:

seed: 42

Seed must not be inferred only from directory names or scheduler array indexes.

If multiple random-number generators use separate seeds, those seeds should be
recorded explicitly when scientifically relevant.

---

## 9. Environment Identity

Each execution should record enough environment information to explain its
runtime.

Relevant information may include:

- operating system;
- hostname;
- Python version;
- dependency lock or environment hash;
- accelerator type;
- CUDA version;
- scheduler;
- scheduler job ID;
- CPU/GPU allocation.

The environment describes execution provenance.

It should not redefine experiment identity unless the scientific method
depends on the environment.

---

## 10. Artifact Identity

Every important output should be attributable to:

- study ID;
- experiment ID;
- execution ID;
- config hash;
- Git commit.

Artifacts may include:

- checkpoints;
- synthetic samples;
- metric summaries;
- evaluation results;
- tables;
- figures;
- logs;
- manifests.

Important artifacts should have checksums when practical.

---

## 11. Manifest

A Manifest records what actually happened during an execution.

The manifest is distinct from the experiment configuration.

CONFIG describes intended scientific work.

MANIFEST describes concrete execution reality.

A manifest should eventually contain fields such as:

schema_version

study_id

experiment_id

execution_id

status

started_at

finished_at

git_commit

git_dirty

config_sha256

dataset_id

resolved_dataset_path

artifact_root

seed

model

hostname

scheduler

scheduler_job_id

environment

artifacts

checksums

---

## 12. Execution Status

Executions should have explicit status.

Examples:

planned

running

completed

failed

cancelled

invalidated

completed does not automatically mean scientifically accepted.

A technically successful execution may still be scientifically invalid.

---

## 13. Scientific Acceptance

TrustForge distinguishes:

EXECUTION COMPLETED

from:

SCIENTIFIC EVIDENCE ACCEPTED

Accepted evidence must be explicitly identified.

This prevents temporary, exploratory, or superseded results from silently
becoming publication evidence.

An acceptance record should identify:

- study;
- experiment or experiment set;
- accepted artifact;
- checksum where practical;
- accepting evidence record;
- rationale or evidence boundary;
- relevant Git anchor.

---

## 14. latest.json Rule

latest.json may be useful operationally.

It must not be treated as authoritative scientific identity.

Scientific claims should reference immutable evidence such as:

- timestamped result files;
- hashes;
- manifests;
- frozen tables;
- evidence indexes;
- Git commits;
- scientific release tags.

---

## 15. Repeated and Failed Runs

TrustForge must preserve the difference between:

- experiment identity;
- execution attempts.

A failed execution does not alter the experiment definition.

A rerun with identical scientific configuration should normally retain the
same experiment ID but receive a new execution ID.

A change to scientifically meaningful parameters requires a new experiment
identity or configuration hash.

---

## 16. Historical Studies

Historical Papers 1 through 4 were created before the TrustForge contract.

Their provenance will therefore be reconstructed rather than rewritten.

Historical configs, commits, tags, evidence indexes, hashes, logs, and artifact
paths will be mapped into TrustForge lineage records.

The original historical records must remain preserved.

---

## 17. Reproduction

A reproduction attempt must identify what is being reproduced.

Possible targets include:

- exact historical execution;
- exact historical scientific configuration;
- published table;
- published figure;
- aggregate result;
- qualitative scientific conclusion.

TrustForge should not claim exact reproduction when only conceptual
reproduction has occurred.

---

## 18. Regression Validation

Migration of a historical study requires comparison against frozen scientific
evidence.

Validation may include:

- artifact hash equality;
- table equality;
- metric equality within defined tolerance;
- figure-data equality;
- expected experiment counts;
- expected seed coverage;
- expected class coverage.

The acceptance criterion must be defined before declaring a historical study
successfully migrated.

---

## 19. Immutable Scientific Boundary

Once evidence is frozen for a study, later infrastructure changes must not
silently alter its meaning.

Later software may improve:

- engineering quality;
- portability;
- performance;
- orchestration;
- diagnostics.

But scientific reinterpretation requires an explicit new experiment or study
boundary.

---

## 20. TrustForge Goal

The desired provenance chain is:

STUDY
  -> EXPERIMENT
  -> CONFIG HASH
  -> EXECUTION
  -> MANIFEST
  -> ARTIFACTS
  -> ACCEPTED EVIDENCE
  -> SCIENTIFIC CLAIM

Every important scientific result should eventually be traceable through this
chain.
