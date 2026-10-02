# M8.6 — TrustForge Repository Rename-Readiness Acceptance

**Status:** `READY_FOR_CONTROLLED_RENAME`
**M8.5 acceptance commit:** `ff5f617`
**M8.4 acceptance commit:** `a11445a`

## Decision

The repository is accepted as ready for a controlled rename transaction to TrustForge.

The rename is an identity/location transaction only. It is not a scientific migration, historical rewrite, runtime replacement, plugin migration, or compatibility cleanup.

## Identity transition

- Repository: `GenCyberSynth_Phase1_Scaffold` → `TrustForge`
- Remote: `git@github.com:edgemindstudio/GenCyberSynth_Phase1_Scaffold.git` → `git@github.com:edgemindstudio/TrustForge.git`
- Local directory: `/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold` → `/home/bruno.fonkeng/ProbabilisticModels/TrustForge`

## Readiness findings

- Structural normalization accepted: `true`
- Paper isolation defined: `true`
- Historical authority preserved: `true`
- Rename dependency inventory complete: `true`
- Direct rename blocker groups: **1**
- Direct blocker: `Git remote origin`
- Blocker resolvable inside transaction: `true`
- Historical rewrite required: `false`
- Paper migration required before rename: `false`
- gcs-core rename required: `false`
- Protected gencys storage rename required: `false`

## Principles

- `REPOSITORY_RENAME != HISTORICAL_REWRITE`
- `REPOSITORY_RENAME != PAPER_MIGRATION`
- `CURRENT_IDENTITY != HISTORICAL_EXECUTION_IDENTITY`
- `REMOTE_UPDATE_BELONGS_TO_RENAME_TRANSACTION`
- `PROTECTED_STORAGE_IDENTITY != REPOSITORY_IDENTITY`
- `GCS_CORE_IDENTITY != PARENT_REPOSITORY_IDENTITY`
- `TREE_CONTENT_MUST_REMAIN_STABLE_DURING_IDENTITY_RENAME`

## Preflight

- Verify branch migration/trustforge-foundation.
- Capture git rev-parse HEAD and treat it as RENAME_PREFLIGHT_HEAD.
- Capture git rev-parse HEAD^{tree} and treat it as RENAME_PREFLIGHT_TREE.
- Capture git ls-files count.
- Capture git remote -v.
- Capture git submodule status.
- Confirm only known untracked scratch/bundle files are present.

## Ordered rename transaction

1. Rename the GitHub repository from GenCyberSynth_Phase1_Scaffold to TrustForge under edgemindstudio.
2. Rename the local repository directory from GenCyberSynth_Phase1_Scaffold to TrustForge.
3. Set origin to git@github.com:edgemindstudio/TrustForge.git.
4. If TRUSTFORGE_REPO_ROOT is explicitly configured, update it to the new local repository directory.

## Forbidden during the rename transaction

- Do not rewrite Paper 1 repository/provenance fields.
- Do not rewrite Paper 2 historical result/evidence absolute paths.
- Do not rename or reorganize /home/bruno.fonkeng/gencys.
- Do not rename gcs-core or gcs_core.
- Do not remove GCS_* compatibility aliases.
- Do not migrate Papers 2–4.
- Do not move app/, adapters/, common/, eval/, or model packages.
- Do not combine current-facing branding edits with the identity/location rename transaction.

## Postflight validation

- Verify git rev-parse HEAD matches RENAME_PREFLIGHT_HEAD.
- Verify git rev-parse HEAD^{tree} matches RENAME_PREFLIGHT_TREE.
- Verify git ls-files count matches preflight.
- Verify git rev-parse --show-toplevel resolves to /home/bruno.fonkeng/ProbabilisticModels/TrustForge.
- Verify origin resolves to git@github.com:edgemindstudio/TrustForge.git.
- Verify git submodule status is unchanged.
- Run PYTHONPATH=src python -m pytest -q tests/trustforge.
- Run PYTHONPATH=src python scripts/trustforge_doctor.py.
- Run git diff --check.

## Preserve without rewrite

- Accepted Paper 1 repository identity fields naming edgemindstudio/GenCyberSynth_Phase1_Scaffold.
- Paper 1 deterministic generator historical repository constant.
- Paper 2 historical result/evidence absolute paths containing GenCyberSynth_Phase1_Scaffold.
- Explicit historical GenCyberSynth lineage documentation.
- /home/bruno.fonkeng/gencys protected data/artifact storage.
- gcs-core submodule URL and gcs_core compatibility imports.
- Legacy GCS_* environment aliases.

## Follow-up after rename

- **Current-facing repository branding** — README.md, Runbook.md, .github/git-commit-instructions.md, current CLI/help/comments, current Makefile branding (`separate post-rename commit`)
- **Mixed Makefile compatibility semantics** — Keep gcs-core and historical-storage behavior separate from branding updates. (`review after rename`)
- **model-template legacy identity** — Review with plugin architecture; do not bulk-substitute during rename. (`deferred`)
- **Paper 2 portability debt** — Replace active hard-coded gencys roots only during canonical Paper 2 migration. (`Paper 2 migration`)

## Authority boundary

- Controlled repository rename transaction: `authorized`
- Historical evidence rewriting: `not authorized`
- Papers 2–4 migration: `not authorized`
- Runtime replacement: `not authorized`
- Plugin relocation: `not authorized`
- gcs-core rename: `not authorized`
- Protected storage rename: `not authorized`
- Legacy alias removal: `not authorized`
