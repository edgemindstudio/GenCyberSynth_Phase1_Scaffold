# TrustForge Research Lineage

## Purpose

This document records the scientific lineage of the completed studies being
preserved during the TrustForge migration.

Detailed study records will eventually live under:

docs/research-lineage/

## Paper 1 - Generative Benchmark

Historical identity:
GenCyberSynth Phase-1

Scientific role:
Benchmark foundation for synthetic malware-image generation.

Key properties:

- seven generator families;
- USTC-TFC2016;
- CICMalDroid2020;
- seeds 42, 43, 44;
- utility-first evaluation;
- KID and MS-SSIM as supporting diagnostics.

Recovered Git anchors:

- 37c6092
  Standardize seed-aware artifacts for Paper 1 reruns

- cbf7150f351a4978301c095c0e69633ab552af64
  Exact pre-KID evidence code state

- a9d6317a34e63df1e4ee445b618c29169e7a26e4
  Exact post-KID evidence code state

- b75fce9
  CICMalDroid and cross-dataset integration

Authoritative USTC and CIC result CSV hashes were matched against the preserved
Paper 1 evidence package.

## Paper 2 - Conditional Generation Audit

Study root:

papers/paper2_conditional_generation_done_right/

Scientific role:

Determine whether class-conditional synthetic samples are externally
recognizable as their requested classes.

Evidence includes:

- seed stability;
- audit-budget stability;
- multiple independent audit classifiers;
- CICMalDroid validation;
- cross-dataset stability.

Final evidence index commit:

262998a

No later Paper 2 study commits were observed after 2026-05-11.

## Paper 3 - Synthetic Augmentation Regimes

Study root:

papers/paper3_when_does_synth_help/

Scientific role:

Determine when synthetic augmentation improves or harms downstream malware
classification.

Experimental dimensions include:

- GAN;
- VAE;
- Diffusion;
- multiple budgets;
- balanced augmentation;
- minority-heavy c4/c7 augmentation;
- seeds 42, 43, 44.

Original frozen result boundary:

6e9dd1b

Journal evidence inventory:

9942640

Journal extensions include:

- c4/c7 per-class diagnostics;
- precision-recall analysis;
- synthetic class-alignment auditing;
- ResNet classifier sensitivity.

Final journal-development boundary:

852cf18

No later Paper 3 study commits were observed after 2026-05-25.

## Paper 4 - Selective Synthetic Data Policies

Study root:

papers/paper4_selective_synth_policies/

Scientific role:

Treat synthetic-data inclusion as a policy-controlled intervention rather than
an automatic keep-all operation.

Policy families include:

- keep-all;
- confidence filtering;
- confidence-ranked top-k;
- class-repair allocation.

Journal evidence includes:

- quality-utility diagnostics;
- strict-confidence collapse;
- top-k budget sensitivity;
- class-repair allocation;
- CICMalDroid cross-dataset validation.

Important scientific tags include:

- paper4-baseline-keepall-20260520
- paper4-confidence-policies-20260520
- paper4-class-repair-allocation-20260520
- paper4-journal-quality-utility-20260522
- paper4-journal-class-repair-allocation-20260522
- paper4-journal-strict-conf-collapse-20260522
- paper4-journal-topk-budget-sensitivity-20260522
- paper4-journal-cicmaldroid-keepall-20260525
- paper4-journal-cicmaldroid-policies-20260526
- paper4-journal-evidence-checkpoint-20260526
- paper4-journal-manuscript-evidence-map-20260526
- paper4-journal-main-tables-cross-dataset-figures-20260526

## Lineage Policy

TrustForge migration must preserve:

- historical Git commits;
- scientific tags;
- frozen configurations;
- evidence indexes;
- frozen tables and figures;
- artifact identities;
- hashes where available;
- conference versus journal evidence boundaries.

Future shared infrastructure must not overwrite or silently reinterpret these
historical scientific states.
