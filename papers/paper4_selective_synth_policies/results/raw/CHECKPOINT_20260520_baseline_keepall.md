# Paper 4 Checkpoint: Full Baseline + keep_all Identity Policy

Date: 2026-05-20

This checkpoint freezes the Paper 4 GAN baseline and identity-policy validation layer.

## Evidence table

Primary paper table:

- `paper4_results_paper.csv`

Frozen snapshot:

- `paper4_results_paper_full_baseline_plus_keepall_allseeds_20260520_103938.csv`

## Coverage

The current paper table contains 21 rows:

- 12 baseline rows
- 9 keep_all identity-policy rows

## Baseline coverage

Real-only:

- budget 0, seeds 42, 43, 44

Naive fixed-budget GAN augmentation:

- budget 25, seeds 42, 43, 44
- budget 500, seeds 42, 43, 44
- budget 2000, seeds 42, 43, 44

## keep_all identity-policy coverage

Policy-filtered keep_all manifests:

- budget 25, seeds 42, 43, 44
- budget 500, seeds 42, 43, 44
- budget 2000, seeds 42, 43, 44

## Technical milestone

The shared `eval/runner.py` manifest loader was updated to resolve relative manifest paths robustly for both baseline manifests and deeper policy manifests such as:

`synthetic/<config_id>/seed<seed>/policy/<policy_id>/manifest.json`

This enables policy manifests to load synthetic image paths correctly without overwriting baseline manifests.

## Scientific meaning

The keep_all policy is an identity-policy validation layer. It is not intended as a selective improvement policy. Its purpose is to prove that policy-specific manifests can be evaluated through the same downstream utility pipeline as baseline augmentation.

Next step: implement the first real selective policy, likely `confidence_accept`, starting with budget 2000 across seeds 42, 43, and 44.
