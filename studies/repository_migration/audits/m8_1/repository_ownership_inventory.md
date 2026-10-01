# M8.1 — Repository Ownership & Dependency Inventory

**Mode:** READ_ONLY_AUDIT
**Baseline commit:** `d60f9c97e6b01b78cd615aa0fdb4aa61f3acce51`
**Generation branch:** `migration/trustforge-foundation`

## Snapshot boundary

This audit is pinned to the accepted pre-M8 repository snapshot.
Its classification counts remain stable after M8.1 itself is committed.

## Authority boundary

This audit records observed ownership/dependency evidence only.
It does not assign future owners, authorize moves, or perform migration.

## Classification summary

| Classification | Files |
|---|---:|
| `AMBIGUOUS_REQUIRES_REVIEW` | 86 |
| `HISTORICAL_COMPATIBILITY` | 21 |
| `MULTI_PAPER_SHARED_RUNTIME` | 56 |
| `OPERATIONAL_INFRASTRUCTURE` | 29 |
| `PAPER01_SPECIFIC` | 175 |
| `PAPER02_SPECIFIC` | 201 |
| `PAPER03_SPECIFIC` | 268 |
| `PAPER04_SPECIFIC` | 158 |
| `SCAFFOLD_ONLY` | 2 |
| `SHARED_TRUSTFORGE_CORE` | 28 |

## Top-level component rollup

| Component | Classification | Tracked files |
|---|---|---:|
| `.gitattributes` | `OPERATIONAL_INFRASTRUCTURE` | 1 |
| `.github` | `OPERATIONAL_INFRASTRUCTURE` | 4 |
| `.gitignore` | `OPERATIONAL_INFRASTRUCTURE` | 1 |
| `.gitmodules` | `OPERATIONAL_INFRASTRUCTURE` | 1 |
| `Makefile` | `MULTI_PAPER_SHARED_RUNTIME` | 1 |
| `README.md` | `OPERATIONAL_INFRASTRUCTURE` | 1 |
| `Runbook.md` | `OPERATIONAL_INFRASTRUCTURE` | 1 |
| `adapters` | `MULTI_PAPER_SHARED_RUNTIME` | 10 |
| `app` | `MULTI_PAPER_SHARED_RUNTIME` | 2 |
| `autoregressive` | `MULTI_PAPER_SHARED_RUNTIME` | 5 |
| `common` | `MULTI_PAPER_SHARED_RUNTIME` | 2 |
| `configs` | `AMBIGUOUS_REQUIRES_REVIEW` | 19 |
| `configs` | `HISTORICAL_COMPATIBILITY` | 9 |
| `configs` | `MULTI_PAPER_SHARED_RUNTIME` | 1 |
| `demo` | `AMBIGUOUS_REQUIRES_REVIEW` | 1 |
| `diffusion` | `MULTI_PAPER_SHARED_RUNTIME` | 5 |
| `docs` | `PAPER01_SPECIFIC` | 1 |
| `docs` | `SHARED_TRUSTFORGE_CORE` | 5 |
| `eval` | `MULTI_PAPER_SHARED_RUNTIME` | 3 |
| `gan` | `MULTI_PAPER_SHARED_RUNTIME` | 7 |
| `gaussianmixture` | `MULTI_PAPER_SHARED_RUNTIME` | 5 |
| `gcs-core` | `AMBIGUOUS_REQUIRES_REVIEW` | 1 |
| `manifests` | `PAPER01_SPECIFIC` | 1 |
| `maskedautoflow` | `MULTI_PAPER_SHARED_RUNTIME` | 5 |
| `model-template` | `OPERATIONAL_INFRASTRUCTURE` | 16 |
| `papers` | `PAPER02_SPECIFIC` | 201 |
| `papers` | `PAPER03_SPECIFIC` | 268 |
| `papers` | `PAPER04_SPECIFIC` | 158 |
| `papers` | `SCAFFOLD_ONLY` | 2 |
| `repo_tree.py` | `OPERATIONAL_INFRASTRUCTURE` | 1 |
| `requirements.ci.txt` | `OPERATIONAL_INFRASTRUCTURE` | 1 |
| `requirements.txt` | `OPERATIONAL_INFRASTRUCTURE` | 1 |
| `requirements.txt.bak` | `AMBIGUOUS_REQUIRES_REVIEW` | 1 |
| `restrictedboltzmann` | `MULTI_PAPER_SHARED_RUNTIME` | 5 |
| `run_gcs.slurm` | `AMBIGUOUS_REQUIRES_REVIEW` | 1 |
| `run_suite.slurm` | `AMBIGUOUS_REQUIRES_REVIEW` | 1 |
| `run_tuning.slurm` | `AMBIGUOUS_REQUIRES_REVIEW` | 1 |
| `schemas` | `PAPER01_SPECIFIC` | 3 |
| `schemas` | `SHARED_TRUSTFORGE_CORE` | 3 |
| `scripts` | `AMBIGUOUS_REQUIRES_REVIEW` | 39 |
| `scripts` | `PAPER01_SPECIFIC` | 8 |
| `scripts` | `SHARED_TRUSTFORGE_CORE` | 2 |
| `slurm` | `AMBIGUOUS_REQUIRES_REVIEW` | 10 |
| `slurm` | `HISTORICAL_COMPATIBILITY` | 12 |
| `src` | `PAPER01_SPECIFIC` | 4 |
| `src` | `SHARED_TRUSTFORGE_CORE` | 18 |
| `studies` | `PAPER01_SPECIFIC` | 117 |
| `tests` | `AMBIGUOUS_REQUIRES_REVIEW` | 9 |
| `tests` | `OPERATIONAL_INFRASTRUCTURE` | 1 |
| `tests` | `PAPER01_SPECIFIC` | 25 |
| `tools` | `AMBIGUOUS_REQUIRES_REVIEW` | 3 |
| `tools` | `PAPER01_SPECIFIC` | 16 |
| `vae` | `MULTI_PAPER_SHARED_RUNTIME` | 5 |

## Dependency signals

- TrustForge imports observed: 56
- Root-runtime imports observed: 57
- `python -m app.main` invocations observed: 136

### Paper-reference counts by top-level component

| Component | Reference lines |
|---|---:|
| `Makefile` | 35 |
| `adapters` | 1 |
| `app` | 3 |
| `configs` | 109 |
| `docs` | 86 |
| `eval` | 13 |
| `gan` | 7 |
| `manifests` | 4 |
| `papers` | 2012 |
| `schemas` | 23 |
| `scripts` | 253 |
| `slurm` | 15 |
| `src` | 181 |
| `studies` | 1477 |
| `tests` | 298 |
| `tools` | 209 |
| `vae` | 1 |

## M8.1 interpretation guardrails

- `MULTI_PAPER_SHARED_RUNTIME` does not imply `SHARED_TRUSTFORGE_CORE`.
- `OPERATIONAL_INFRASTRUCTURE` is not a scientific ownership assignment.
- Current location does not determine future ownership.
- Imports/invocations are dependency evidence, not migration authorization.
- Historical study files and path-bearing evidence remain untouched.
- `future_owner`, `migration_action`, and `authority_decision` remain null.
