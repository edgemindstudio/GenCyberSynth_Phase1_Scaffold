# TrustForge Architecture

## Status

TrustForge is the permanent research framework that will evolve from the
historical GenCyberSynth research repository.

This document defines architecture only. It does not authorize deletion,
movement, or rewriting of historical scientific evidence.

## 1. Purpose

TrustForge supports trustworthy probabilistic and generative machine learning
for cybersecurity, including:

- synthetic cybersecurity data;
- malware-image classification;
- conditioning and class-faithfulness audits;
- synthetic augmentation;
- selective synthetic-data policies;
- uncertainty and calibration;
- abstention and risk-aware decisions;
- distribution shift;
- probabilistic telemetry modeling;
- reproducibility and provenance;
- HPC-scale experimentation.

GenCyberSynth remains a historical research identity where required for
existing papers, artifacts, commits, and provenance.

## 2. Core Principle

No machine owns TrustForge.

The Git repository defines the research system.

External storage holds large datasets, checkpoints, generated samples,
execution logs, and other large artifacts.

Scientific provenance must bind claims to exact code, configuration, data,
environment, execution, and evidence.

## 3. Scientific Separation

TrustForge preserves:

CODE != CONFIG != MANIFEST != ARTIFACT != STUDY

CODE:
Reusable implementation.

CONFIG:
Exact scientific experiment definition.

MANIFEST:
Record of what actually executed.

ARTIFACT:
Generated evidence such as checkpoints, synthetic samples, metrics, tables,
figures, and logs.

STUDY:
Scientific interpretation and paper-specific evidence.

## 4. State Model

FROZEN:
Historical evidence already supporting an established scientific result.
It must not silently change.

ACTIVE:
Research currently under development.

SHARED:
Reusable TrustForge infrastructure that may evolve subject to validation.

## 5. Historical Reproducibility

A later software improvement must never silently redefine an earlier
experiment.

Historical studies may therefore retain:

- exact Git anchors;
- frozen configurations;
- compatibility code;
- artifact hashes;
- evidence indexes;
- scientific release tags.

## 6. Target Repository Architecture

The long-term structure is expected to converge toward:

trustforge/
  src/trustforge/
  studies/
  configs/
  manifests/
  hpc/
  scripts/
  tests/
  docs/
  README.md

Existing files will not be moved into this structure until their scientific
lineage and migration requirements are validated.

## 7. Storage Contract

Git is not an artifact warehouse.

Large data and artifacts remain outside the repository.

TrustForge will use portable logical roots such as:

TRUSTFORGE_DATA_ROOT

TRUSTFORGE_ARTIFACTS_ROOT

Machine-specific absolute paths must not define scientific meaning.

## 8. HPC Contract

HPC is an execution backend, not the scientific experiment definition.

The intended flow is:

experiment definition
  -> execution plan
  -> local or Slurm backend

Slurm may define resources and scheduling, but must not silently define the
scientific semantics of the experiment.

## 9. Migration Sequence

M0 - Protect current repository state
M1 - Architecture documentation
M2 - Storage and path contract
M3 - Study, experiment, and manifest contracts
M4 - Shared TrustForge package skeleton
M5 - Engineering validation tooling
M6 - Paper 1 migration
M7 - Paper 1 scientific regression validation
M8 - Paper 2 migration
M9 - Paper 3 migration
M10 - Paper 4 migration
M11 - Future-study template
M12 - Repository rename to TrustForge
M13 - Multi-machine synchronization

## 10. Protection Boundary

Pre-TrustForge tag:

pre-trustforge-foundation-20260916

Pre-TrustForge commit:

8358f57339cd9278f593053c46677ee84e445b2a

Foundation branch:

migration/trustforge-foundation

Historical paper evidence and existing scientific tags must remain preserved
throughout migration.

