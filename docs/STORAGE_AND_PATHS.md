# TrustForge Storage and Path Contract

## Status

This document defines the storage and path model for TrustForge.

It is a contract for future migration work.

It does not authorize moving, renaming, deleting, or rewriting existing
datasets or artifacts.

---

## 1. Core Principle

Scientific meaning must not depend on a particular computer or absolute
filesystem path.

TrustForge distinguishes between:

- repository location;
- dataset location;
- artifact location;
- temporary execution storage;
- machine-specific implementation details.

A study should describe what data and artifacts it requires without encoding
where a particular user mounted or stored them.

---

## 2. Repository Location

The TrustForge Git repository may be cloned into different locations on
different machines.

Examples include:

Talon:

~/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold

Mac:

~/PycharmProjects/GenCyberSynth_Phase1_Scaffold

Windows:

C:\Users\fonke\Desktop\GenCyberSynth_Phase1_Scaffold

Future Linux desktop:

location to be chosen when cloned.

These physical repository locations are operational details.

They are not part of scientific experiment identity.

---

## 3. Data Root

TrustForge will define a logical dataset root:

TRUSTFORGE_DATA_ROOT

All large research datasets should ultimately be resolved relative to this
logical root or through dataset manifests.

Code must not require hard-coded paths such as:

/home/bruno.fonkeng/...

/Users/poisedre/...

C:\Users\fonke\...

---

## 4. Artifact Root

TrustForge will define a logical artifact root:

TRUSTFORGE_ARTIFACTS_ROOT

Generated research artifacts include:

- checkpoints;
- synthetic samples;
- raw summaries;
- evaluation outputs;
- execution logs;
- tensorboard data;
- large intermediate outputs;
- temporary experiment evidence.

These artifacts normally remain outside Git.

---

## 5. Talon Storage

Current Talon external research storage is:

~/gencys

This location currently contains historical data and artifacts for Papers
1 through 4.

Examples include:

~/gencys/data

~/gencys/artifacts_paper1_*

~/gencys/artifacts_paper2

~/gencys/artifacts_paper3

~/gencys/artifacts_paper4

~/gencys/artifacts_paper4_cicmaldroid

~/gencys/paper1_evidence

~/gencys/runs

These directories must not be moved, renamed, merged, or deleted during the
foundation stages.

They contain historical scientific provenance.

---

## 6. Personal Machines

Personal machines should use separate external locations for:

- datasets;
- generated artifacts.

The exact physical directory may differ by operating system.

During migration, existing directories may remain in place.

TrustForge logical variables will provide portability without requiring all
machines to use identical filesystem layouts.

---

## 7. Historical Path Compatibility

Historical configs, scripts, summaries, or manifests may contain absolute
paths.

Those paths are historical evidence and must not be blindly rewritten.

Migration should distinguish between:

1. historical immutable records;
2. active configuration requiring portability;
3. shared code requiring portability.

Historical records may preserve their original absolute paths as evidence of
where an experiment executed.

Active and future code should resolve logical paths dynamically.

---

## 8. Path Resolution Order

Future TrustForge code should resolve storage using an explicit order.

For datasets:

1. dataset manifest path, when explicitly supplied;
2. TRUSTFORGE_DATA_ROOT;
3. documented compatibility fallback, if temporarily required;
4. otherwise fail clearly.

For artifacts:

1. explicitly supplied artifact destination;
2. TRUSTFORGE_ARTIFACTS_ROOT;
3. documented compatibility fallback, if temporarily required;
4. otherwise fail clearly.

Silent machine-specific guessing should be avoided.

---

## 9. Configuration Rule

Portable experiment configs should contain logical or relative identifiers.

Preferred:

dataset: ustc_tfc2016
artifact_namespace: paper03/minority_heavy/gan/seed42

Avoid:

dataset_path: /home/bruno.fonkeng/gencys/data/USTC-TFC2016_malware_nhwc

The physical path should be resolved by the execution environment.

---

## 10. Manifest Rule

Execution manifests may record the resolved absolute path that was actually
used.

This is desirable provenance.

Therefore:

CONFIG describes logical intent.

MANIFEST records concrete execution reality.

A manifest may safely contain:

- resolved dataset path;
- resolved artifact path;
- hostname;
- scheduler job ID;
- execution timestamp.

Such fields describe provenance and should not define portability.

---

## 11. Repository Content Rule

The repository should contain:

- source code;
- experiment configs;
- study definitions;
- manifests intended for preservation;
- small evidence tables;
- publication figures when appropriate;
- documentation;
- tests;
- orchestration logic.

The repository should normally not contain:

- large raw datasets;
- large generated datasets;
- large checkpoints;
- tensorboard trees;
- bulk execution logs;
- large temporary intermediates.

---

## 12. Dataset Identity

A dataset must eventually be identified by more than a filesystem path.

Dataset identity should include, where practical:

- canonical dataset name;
- version or preparation stage;
- shape or representation;
- preprocessing identity;
- split identity;
- checksums or manifest hashes.

Two directories with similar names must not automatically be assumed to
represent the same scientific dataset.

---

## 13. Artifact Identity

Artifact identity should eventually include:

- study ID;
- experiment ID;
- seed;
- model or generator;
- configuration hash;
- Git commit or release;
- creation timestamp;
- artifact manifest.

Directory names are useful organizational aids but are not sufficient
scientific identity by themselves.

---

## 14. Compatibility During Migration

TrustForge migration will initially support historical layouts.

The first goal is not to reorganize storage.

The first goal is to make storage meaning explicit.

Only after study lineage, path resolution, and artifact identity are validated
may physical storage cleanup be considered.

---

## 15. Legacy Naming

Historical directories may continue to contain names such as:

GenCyberSynth

gencys

paper1

paper2

paper3

paper4

These names may remain where they serve provenance.

New framework-level interfaces should use TrustForge terminology.

Legacy environment variables, if discovered, may temporarily be supported as
compatibility aliases, but new documentation and code should use:

TRUSTFORGE_DATA_ROOT

TRUSTFORGE_ARTIFACTS_ROOT

---

## 16. Safety Rule

No storage migration may:

- overwrite historical evidence;
- silently change artifact identity;
- invalidate manuscript evidence;
- destroy hashes;
- merge distinct dataset preparation stages;
- replace historical paths inside frozen evidence;
- assume latest.json is authoritative scientific evidence.

Any future physical reorganization must be separately planned, verified, and
reversible.
