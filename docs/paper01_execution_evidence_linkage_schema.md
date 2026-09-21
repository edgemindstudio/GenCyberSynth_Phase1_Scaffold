# Paper 1 Execution–Evidence Linkage Schema

**Migration stage:** M6.4.2A
**Project:** TrustForge
**Study:** `paper01_benchmark`

## 1. Purpose

This document defines the canonical representation for linking a scientific Paper 1 experiment to its recovered historical executions and surviving evidence.

The linkage model MUST preserve the distinction between scientific experiment identity and historical execution identity.

It MUST NOT rewrite, normalize, merge, or otherwise modify historical Paper 1 artifacts.

M6.4.2 linkage records are provenance records. They describe recovered evidence; they do not retroactively alter historical execution state.

---

## 2. Governing Principles

The following distinctions are mandatory:

`SCIENTIFIC EXPERIMENT IDENTITY ≠ HISTORICAL EXECUTION IDENTITY`

`ARTIFACT PRODUCER ≠ ACCEPTED EVALUATOR ≠ AUTHORITATIVE RESULT ROW`

`MISSING SEED MANIFEST ≠ MISSING EXPERIMENT`

`MULTIPLE COMPLETED HISTORICAL EXECUTIONS ≠ ONE AUTOMATICALLY AUTHORITATIVE EXECUTION`

`OBSERVED EVIDENCE ≠ PROVEN LINEAGE`

`TIMESTAMP ALIGNMENT ≠ EXECUTION PROOF`

`latest.json ≠ AUTHORITATIVE EVIDENCE`

A linkage record MUST preserve historical multiplicity rather than collapse several executions into one fictional run.

---

## 3. Canonical Record Scope

There will ultimately be exactly one canonical linkage record for each scientific experiment defined by:

`studies/paper01_benchmark/experiment_index.yaml`

Paper 1 contains:

* 2 datasets;
* 7 model families;
* 3 seeds;
* 42 scientific experiments.

A linkage record represents one scientific experiment and may reference multiple historical executions.

---

## 4. Proposed Canonical Record Structure

Each linkage record SHALL contain the following top-level structure:

```yaml
schema_version: trustforge.paper01.execution_evidence_linkage.v1

experiment:
  experiment_id: cicmaldroid2020_gaussianmixture_b2000_seed42
  dataset: cicmaldroid2020
  family: gaussianmixture
  seed: 42
  budget_per_class: 2000
  historical_run_id: gaussianmixture_s42

authoritative_result:
  status: verified
  table_path: /historical/path/phase1_scores_dedup.csv
  table_sha256: "<sha256>"
  row_key:
    run_id: gaussianmixture_s42
  accepted_metric_match:
    status: exact
    summary_count: 1

accepted_evaluation:
  status: verified
  execution:
    job_id: "246862"
    task_id: "15"
    host: talon35
    config_sha1: 0d8e51e3e3a54b2816ac72bed5ee001973498cbb
    git_commit: a9d6317a34e63df1e4ee445b618c29169e7a26e4
    do_train: false
    do_synth: false
    do_eval: true
    stages:
      - eval
    log_path: papers/paper1_phase1_benchmark/logs/slurm_raw/slurm-paper1.246846_15.out

  accepted_summary:
    path: /historical/path/summary_20260410_091359.json
    authority: accepted_timestamped_summary
    metric_match: exact
    config_sha1_match_to_execution: true

artifact_production:
  status: strong_candidate
  executions:
    - job_id: "246427"
      task_id: "15"
      host: talon35
      config_sha1: 5d601b08f992905c7764f13d2e8c811550461eda
      git_commit: a9d6317a34e63df1e4ee445b618c29169e7a26e4
      do_train: true
      do_synth: true
      do_eval: true
      log_path: papers/paper1_phase1_benchmark/logs/slurm_raw/slurm-paper1.246407_15.out
      evidence_basis:
        - synthesis_stage_observed
        - surviving_artifact_timestamps_align
        - checkpoint_timestamps_align

  lineage_claim:
    level: strong_candidate
    proven: false
    reason: >
      Historical execution explicitly synthesized artifacts and surviving
      artifact timestamps strongly align, but timestamp alignment alone does
      not prove exclusive artifact lineage.

historical_executions:
  - job_id: "246364"
    role:
      - historical_execution
    config_sha1: eae6301fea41a1a5fd9b8fe410a1145ae5a12316
    git_commit: a9d6317a34e63df1e4ee445b618c29169e7a26e4

  - job_id: "246427"
    role:
      - historical_execution
      - artifact_producer_candidate
    config_sha1: 5d601b08f992905c7764f13d2e8c811550461eda
    git_commit: a9d6317a34e63df1e4ee445b618c29169e7a26e4

  - job_id: "246862"
    role:
      - historical_execution
      - accepted_evaluator
    config_sha1: 0d8e51e3e3a54b2816ac72bed5ee001973498cbb
    git_commit: a9d6317a34e63df1e4ee445b618c29169e7a26e4

manifest_evidence:
  seed_specific:
    status: absent
    path: null

  shared_root:
    status: present
    path: /historical/path/gaussianmixture/synthetic/manifest.json
    authority: compatibility_evidence

synthetic_evidence:
  status: present
  file_count: 10000
  layout: class_then_seed
  layout_counts:
    class_then_seed: 10000

checkpoint_evidence:
  status: present
  file_count: 6

non_authoritative_aliases:
  latest_summary:
    present: true
    authority: none
    authoritative: false
    reason: compatibility_alias

provenance_assertions:
  accepted_summary_to_evaluation_execution:
    status: proven
    basis:
      - matching_config_sha1
      - execution_log_reports_summary_path

  accepted_summary_to_authoritative_result:
    status: proven
    basis:
      - exact_accepted_metric_match

  artifact_execution_to_surviving_artifacts:
    status: strong_candidate
    basis:
      - synthesis_stage_observed
      - timestamp_alignment
    limitations:
      - timestamp_alignment_is_not_execution_proof

source_inventory:
  audit_version: M6.4.1-v3.2
  inventory_path: studies/paper01_benchmark/audits/m6_4_1/paper01_execution_evidence_inventory.json
```

The example above is illustrative of the schema structure. Canonical records MUST be generated from validated evidence rather than copied manually from this example.

---

## 5. Evidence Status Vocabulary

Linkage assertions SHALL use a controlled evidence-status vocabulary.

### `verified`

Use only when direct evidence establishes the fact.

Examples:

* authoritative score row exists;
* timestamped summary exactly matches accepted metrics;
* config SHA1 directly links summary and Slurm execution;
* log explicitly reports model and seed.

### `strong_candidate`

Use when several independent observations strongly support a relationship but direct lineage proof is unavailable.

Example:

* an execution explicitly performed synthesis;
* surviving artifact timestamps align with that execution;
* checkpoints align with that execution;
* but no immutable producer identifier is embedded in each surviving artifact.

### `observed`

Use for filesystem or historical-state observations that establish presence but not lineage.

Examples:

* a checkpoint exists;
* a shared manifest exists;
* a particular directory layout exists.

### `unresolved`

Use when available evidence does not support a stronger assertion.

### `absent`

Use when an expected evidence object was explicitly searched for and was not found.

`absent` MUST NOT automatically mean that the scientific experiment itself is absent or invalid.

---

## 6. Accepted Evaluation Invariants

Every canonical Paper 1 linkage record MUST satisfy the following:

1. Exactly one authoritative score-table row is identified.

2. Exactly one timestamped historical summary matches the authoritative accepted metrics.

3. `latest.json` MUST NOT count toward the accepted historical summary count.

4. The accepted timestamped summary MUST link to at least one historical execution through explicit evidence.

5. A config-SHA1 match is acceptable direct linkage evidence when both the summary and execution log independently report that SHA1.

6. The accepted evaluator MAY differ from the artifact producer.

7. An evaluation-only execution is valid as the accepted evaluator.

---

## 7. Historical Execution Invariants

All explicitly identified historical executions for the scientific experiment MUST remain representable.

A linkage generator MUST NOT:

* discard earlier execution waves merely because a later accepted evaluator exists;
* combine distinct job IDs into one synthetic execution;
* copy fields from one historical execution into another;
* assume identical model and seed imply identical execution;
* infer execution identity solely from timestamp proximity.

Historical execution entries SHOULD retain, when available:

* job ID;
* array task ID;
* host;
* model;
* seed;
* effective config SHA1;
* Git commit;
* stage toggles;
* observed stages;
* log path;
* reported summary path.

---

## 8. Artifact-Producer Invariants

Artifact production is a distinct provenance role.

The linkage system MUST support:

* one proven producer;
* one strong producer candidate;
* multiple producer candidates;
* unresolved producer lineage.

A historical execution MUST NOT be labeled `artifact_producer: verified` solely because:

* it occurred before evaluation;
* its timestamp is close to surviving artifacts;
* it used the same model and seed.

Timestamp correlation MAY support `strong_candidate` when combined with explicit synthesis evidence.

---

## 9. Manifest Invariants

Seed-specific manifest absence MUST NOT invalidate a scientific experiment when other historical evidence establishes its execution and outputs.

Paper 1 specifically contains three known seed-manifest exceptions:

* `cicmaldroid2020_gaussianmixture_b2000_seed42`
* `cicmaldroid2020_gaussianmixture_b2000_seed43`
* `cicmaldroid2020_gaussianmixture_b2000_seed44`

These experiments retain synthetic outputs under the historical `class_then_seed` layout and have a shared compatibility manifest.

The linkage representation MUST preserve this condition rather than synthesize nonexistent seed-specific manifests.

---

## 10. Artifact Layout Invariants

Historical artifact layouts are provenance evidence.

The current M6.4.1 census establishes:

* 36 experiments with `seed_directory`;
* 3 CICMalDroid GMM experiments with `class_then_seed`;
* 3 USTC GMM experiments with coexisting `class_then_seed` and `seed_directory` layouts.

The canonical linkage layer MUST describe these layouts but MUST NOT normalize, move, duplicate, delete, or reorganize historical artifact files.

---

## 11. `latest.json` Rule

`latest.json` is a historical compatibility/convenience alias.

It MAY be recorded as observed evidence.

It MUST NOT:

* become the accepted summary;
* create a second accepted evaluation;
* increase accepted-summary cardinality;
* determine canonical execution identity;
* override a timestamped historical summary.

Permanent rule:

`latest.json ≠ AUTHORITATIVE EVIDENCE`

---

## 12. Path Representation

Repository-owned evidence SHOULD use repository-relative paths.

Example:

```yaml
log_path: papers/paper1_phase1_benchmark/logs/slurm_raw/slurm-paper1.246846_15.out
```

Historical external artifacts MAY retain absolute historical paths when those paths are part of the recovered evidence.

Absolute historical paths are references to preserved historical state; they are not TrustForge storage configuration.

Canonical linkage files themselves MUST reside inside the repository.

They MUST NOT be written into historical artifact roots.

---

## 13. Hash and Identity Rules

Different hashes MUST retain their original semantics.

Examples:

* authoritative result-table SHA256;
* canonical Git-blob SHA256 for historical source configs;
* effective task-config SHA1 recorded by historical Slurm execution;
* future linkage-record content hashes.

These hashes MUST NOT be substituted for one another.

An effective config SHA1 identifies an execution-time configuration artifact.

It is not equivalent to the canonical Git-blob SHA256 of the historical source configuration file.

---

## 14. Generation Rules

Future linkage generation MUST be deterministic.

Given:

* the same M6.4.1 evidence inventory;
* the same experiment index;
* the same authoritative score-table identities;
* the same linkage schema version;

the generator MUST produce semantically identical linkage records.

Generation MUST NOT depend on:

* current wall-clock time;
* filesystem traversal order;
* `latest.json`;
* implicit “most recent file” selection;
* undocumented heuristics.

---

## 15. Validation Rules

Before canonical linkage records are accepted, validation MUST establish at minimum:

* exactly 42 linkage records;
* exactly one record per scientific experiment;
* no duplicate experiment IDs;
* every record references an authoritative score row;
* every record identifies exactly one accepted timestamped summary;
* every accepted summary has exact accepted-metric correspondence;
* no `latest.json` is authoritative;
* every accepted evaluator is backed by explicit execution evidence;
* all historical executions remain distinguishable;
* artifact-producer confidence is explicit;
* missing manifests remain explicitly represented;
* model-specific layouts remain preserved;
* no historical artifacts are modified.

---

## 16. Migration Boundary

M6.4.2 SHALL create new TrustForge provenance representation only.

It SHALL NOT:

* patch historical Paper 1 result tables;
* rewrite historical manifests;
* rename historical summaries;
* change Slurm logs;
* reorganize synthetic images;
* remove duplicate historical layouts;
* alter checkpoints;
* rewrite historical source configurations.

Historical evidence is immutable input to the migration.

---

## 17. Next Step

After this schema contract is reviewed and regression-tested, M6.4.2B will implement:

1. a machine-readable linkage schema;
2. a deterministic linkage-record generator;
3. linkage validation tests;
4. generation of 42 canonical Paper 1 linkage records;
5. a study-level linkage index.

No canonical linkage record should be generated before the schema and its invariants are validated.
