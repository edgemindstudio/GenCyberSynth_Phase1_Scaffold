#!/usr/bin/env python3
"""
M8.6 — TrustForge repository rename-readiness acceptance.

This milestone determines whether the repository is ready for a controlled
identity/location rename from GenCyberSynth_Phase1_Scaffold to TrustForge.

It does NOT perform the rename.

The accepted transaction may change:
- GitHub repository name,
- local repository directory name,
- origin URL,
- explicitly configured TRUSTFORGE_REPO_ROOT.

The transaction must NOT rewrite historical scientific evidence, protected
~/gencys storage, legacy compatibility aliases, gcs-core identity, Papers 2–4,
or model implementations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


OUTPUT_REL = Path("studies/repository_migration/audits/m8_6")
SCHEMA_REL = OUTPUT_REL / "rename_readiness_acceptance.schema.yaml"

M8_5_ACCEPTANCE_COMMIT = "ff5f617"
M8_4_ACCEPTANCE_COMMIT = "a11445a"

OLD_REPOSITORY_NAME = "GenCyberSynth_Phase1_Scaffold"
NEW_REPOSITORY_NAME = "TrustForge"

OLD_REMOTE = "git@github.com:edgemindstudio/GenCyberSynth_Phase1_Scaffold.git"
NEW_REMOTE = "git@github.com:edgemindstudio/TrustForge.git"

OLD_LOCAL_DIR = "/home/bruno.fonkeng/ProbabilisticModels/GenCyberSynth_Phase1_Scaffold"
NEW_LOCAL_DIR = "/home/bruno.fonkeng/ProbabilisticModels/TrustForge"


def build_acceptance() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "milestone": "M8.6",
        "title": "TrustForge Repository Rename-Readiness Acceptance",
        "status": "READY_FOR_CONTROLLED_RENAME",
        "source_acceptance": {
            "m8_5_acceptance_commit": M8_5_ACCEPTANCE_COMMIT,
            "m8_4_acceptance_commit": M8_4_ACCEPTANCE_COMMIT,
        },
        "identity_transition": {
            "old_repository_name": OLD_REPOSITORY_NAME,
            "new_repository_name": NEW_REPOSITORY_NAME,
            "old_remote": OLD_REMOTE,
            "new_remote": NEW_REMOTE,
            "old_local_directory": OLD_LOCAL_DIR,
            "new_local_directory": NEW_LOCAL_DIR,
        },
        "readiness_findings": {
            "structural_normalization_accepted": True,
            "paper_isolation_defined": True,
            "historical_authority_preserved": True,
            "rename_dependency_inventory_complete": True,
            "direct_rename_blocker_count": 1,
            "direct_rename_blocker": "Git remote origin",
            "blocker_resolvable_inside_transaction": True,
            "historical_rewrite_required": False,
            "paper_migration_required_before_rename": False,
            "gcs_core_rename_required": False,
            "protected_gencys_storage_rename_required": False,
        },
        "authority_boundary": {
            "repository_rename_transaction_authorized": True,
            "historical_evidence_rewrite_authorized": False,
            "paper2_migration_authorized": False,
            "paper3_migration_authorized": False,
            "paper4_migration_authorized": False,
            "runtime_replacement_authorized": False,
            "plugin_relocation_authorized": False,
            "gcs_core_rename_authorized": False,
            "protected_storage_rename_authorized": False,
            "legacy_alias_removal_authorized": False,
        },
        "principles": [
            "REPOSITORY_RENAME != HISTORICAL_REWRITE",
            "REPOSITORY_RENAME != PAPER_MIGRATION",
            "CURRENT_IDENTITY != HISTORICAL_EXECUTION_IDENTITY",
            "REMOTE_UPDATE_BELONGS_TO_RENAME_TRANSACTION",
            "PROTECTED_STORAGE_IDENTITY != REPOSITORY_IDENTITY",
            "GCS_CORE_IDENTITY != PARENT_REPOSITORY_IDENTITY",
            "TREE_CONTENT_MUST_REMAIN_STABLE_DURING_IDENTITY_RENAME",
        ],
        "transaction": {
            "preflight": [
                "Verify branch migration/trustforge-foundation.",
                "Capture git rev-parse HEAD and treat it as RENAME_PREFLIGHT_HEAD.",
                "Capture git rev-parse HEAD^{tree} and treat it as RENAME_PREFLIGHT_TREE.",
                "Capture git ls-files count.",
                "Capture git remote -v.",
                "Capture git submodule status.",
                "Confirm only known untracked scratch/bundle files are present.",
            ],
            "ordered_actions": [
                {
                    "step": 1,
                    "action": "Rename the GitHub repository from GenCyberSynth_Phase1_Scaffold to TrustForge under edgemindstudio.",
                    "content_mutation": False,
                },
                {
                    "step": 2,
                    "action": "Rename the local repository directory from GenCyberSynth_Phase1_Scaffold to TrustForge.",
                    "content_mutation": False,
                },
                {
                    "step": 3,
                    "action": "Set origin to git@github.com:edgemindstudio/TrustForge.git.",
                    "content_mutation": False,
                },
                {
                    "step": 4,
                    "action": "If TRUSTFORGE_REPO_ROOT is explicitly configured, update it to the new local repository directory.",
                    "content_mutation": False,
                },
            ],
            "forbidden_during_transaction": [
                "Do not rewrite Paper 1 repository/provenance fields.",
                "Do not rewrite Paper 2 historical result/evidence absolute paths.",
                "Do not rename or reorganize /home/bruno.fonkeng/gencys.",
                "Do not rename gcs-core or gcs_core.",
                "Do not remove GCS_* compatibility aliases.",
                "Do not migrate Papers 2–4.",
                "Do not move app/, adapters/, common/, eval/, or model packages.",
                "Do not combine current-facing branding edits with the identity/location rename transaction.",
            ],
            "postflight": [
                "Verify git rev-parse HEAD matches RENAME_PREFLIGHT_HEAD.",
                "Verify git rev-parse HEAD^{tree} matches RENAME_PREFLIGHT_TREE.",
                "Verify git ls-files count matches preflight.",
                "Verify git rev-parse --show-toplevel resolves to /home/bruno.fonkeng/ProbabilisticModels/TrustForge.",
                "Verify origin resolves to git@github.com:edgemindstudio/TrustForge.git.",
                "Verify git submodule status is unchanged.",
                "Run PYTHONPATH=src python -m pytest -q tests/trustforge.",
                "Run PYTHONPATH=src python scripts/trustforge_doctor.py.",
                "Run git diff --check.",
            ],
        },
        "preserve_without_rewrite": [
            "Accepted Paper 1 repository identity fields naming edgemindstudio/GenCyberSynth_Phase1_Scaffold.",
            "Paper 1 deterministic generator historical repository constant.",
            "Paper 2 historical result/evidence absolute paths containing GenCyberSynth_Phase1_Scaffold.",
            "Explicit historical GenCyberSynth lineage documentation.",
            "/home/bruno.fonkeng/gencys protected data/artifact storage.",
            "gcs-core submodule URL and gcs_core compatibility imports.",
            "Legacy GCS_* environment aliases.",
        ],
        "post_rename_followup": [
            {
                "item": "Current-facing repository branding",
                "scope": "README.md, Runbook.md, .github/git-commit-instructions.md, current CLI/help/comments, current Makefile branding",
                "timing": "separate post-rename commit",
            },
            {
                "item": "Mixed Makefile compatibility semantics",
                "scope": "Keep gcs-core and historical-storage behavior separate from branding updates.",
                "timing": "review after rename",
            },
            {
                "item": "model-template legacy identity",
                "scope": "Review with plugin architecture; do not bulk-substitute during rename.",
                "timing": "deferred",
            },
            {
                "item": "Paper 2 portability debt",
                "scope": "Replace active hard-coded gencys roots only during canonical Paper 2 migration.",
                "timing": "Paper 2 migration",
            },
        ],
    }


def load_schema(repo: Path) -> dict[str, Any]:
    data = yaml.safe_load((repo / SCHEMA_REL).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("M8.6 schema must be a mapping")
    return data


def validate(data: dict[str, Any], schema: dict[str, Any]) -> None:
    for key in schema["required_top_level"]:
        if key not in data:
            raise ValueError(f"missing top-level field: {key}")

    if data["status"] != "READY_FOR_CONTROLLED_RENAME":
        raise ValueError("M8.6 must explicitly record controlled rename readiness")

    findings = data["readiness_findings"]
    required_true = [
        "structural_normalization_accepted",
        "paper_isolation_defined",
        "historical_authority_preserved",
        "rename_dependency_inventory_complete",
        "blocker_resolvable_inside_transaction",
    ]
    for key in required_true:
        if findings[key] is not True:
            raise ValueError(f"readiness finding must be true: {key}")

    if findings["direct_rename_blocker_count"] != 1:
        raise ValueError("M8.5 established exactly one direct rename blocker group")

    if findings["historical_rewrite_required"] is not False:
        raise ValueError("historical rewrite must not be required")
    if findings["paper_migration_required_before_rename"] is not False:
        raise ValueError("paper migration must not be required before rename")
    if findings["gcs_core_rename_required"] is not False:
        raise ValueError("gcs-core rename must not be required")
    if findings["protected_gencys_storage_rename_required"] is not False:
        raise ValueError("protected storage rename must not be required")

    authority = data["authority_boundary"]
    if authority["repository_rename_transaction_authorized"] is not True:
        raise ValueError("controlled repository rename transaction must be authorized")

    forbidden_authority = {
        key: value
        for key, value in authority.items()
        if key != "repository_rename_transaction_authorized"
    }
    if any(forbidden_authority.values()):
        raise ValueError("M8.6 must not authorize migration/rewrites beyond repository rename")

    steps = data["transaction"]["ordered_actions"]
    if [step["step"] for step in steps] != [1, 2, 3, 4]:
        raise ValueError("rename transaction steps must be deterministic and ordered")
    if any(step["content_mutation"] for step in steps):
        raise ValueError("identity/location rename transaction must not mutate tracked content")

    required_forbidden = {
        "Do not rewrite Paper 1 repository/provenance fields.",
        "Do not rewrite Paper 2 historical result/evidence absolute paths.",
        "Do not rename or reorganize /home/bruno.fonkeng/gencys.",
        "Do not rename gcs-core or gcs_core.",
        "Do not migrate Papers 2–4.",
    }
    if not required_forbidden.issubset(set(data["transaction"]["forbidden_during_transaction"])):
        raise ValueError("rename transaction is missing required preservation guards")


def render_markdown(data: dict[str, Any]) -> str:
    ident = data["identity_transition"]
    findings = data["readiness_findings"]
    tx = data["transaction"]

    lines = [
        "# M8.6 — TrustForge Repository Rename-Readiness Acceptance",
        "",
        f"**Status:** `{data['status']}`",
        f"**M8.5 acceptance commit:** `{data['source_acceptance']['m8_5_acceptance_commit']}`",
        f"**M8.4 acceptance commit:** `{data['source_acceptance']['m8_4_acceptance_commit']}`",
        "",
        "## Decision",
        "",
        "The repository is accepted as ready for a controlled rename transaction to TrustForge.",
        "",
        "The rename is an identity/location transaction only. It is not a scientific migration, historical rewrite, runtime replacement, plugin migration, or compatibility cleanup.",
        "",
        "## Identity transition",
        "",
        f"- Repository: `{ident['old_repository_name']}` → `{ident['new_repository_name']}`",
        f"- Remote: `{ident['old_remote']}` → `{ident['new_remote']}`",
        f"- Local directory: `{ident['old_local_directory']}` → `{ident['new_local_directory']}`",
        "",
        "## Readiness findings",
        "",
        f"- Structural normalization accepted: `{str(findings['structural_normalization_accepted']).lower()}`",
        f"- Paper isolation defined: `{str(findings['paper_isolation_defined']).lower()}`",
        f"- Historical authority preserved: `{str(findings['historical_authority_preserved']).lower()}`",
        f"- Rename dependency inventory complete: `{str(findings['rename_dependency_inventory_complete']).lower()}`",
        f"- Direct rename blocker groups: **{findings['direct_rename_blocker_count']}**",
        f"- Direct blocker: `{findings['direct_rename_blocker']}`",
        f"- Blocker resolvable inside transaction: `{str(findings['blocker_resolvable_inside_transaction']).lower()}`",
        f"- Historical rewrite required: `{str(findings['historical_rewrite_required']).lower()}`",
        f"- Paper migration required before rename: `{str(findings['paper_migration_required_before_rename']).lower()}`",
        f"- gcs-core rename required: `{str(findings['gcs_core_rename_required']).lower()}`",
        f"- Protected gencys storage rename required: `{str(findings['protected_gencys_storage_rename_required']).lower()}`",
        "",
        "## Principles",
        "",
    ]
    lines += [f"- `{p}`" for p in data["principles"]]

    lines += ["", "## Preflight", ""]
    lines += [f"- {x}" for x in tx["preflight"]]

    lines += ["", "## Ordered rename transaction", ""]
    for step in tx["ordered_actions"]:
        lines.append(f"{step['step']}. {step['action']}")

    lines += ["", "## Forbidden during the rename transaction", ""]
    lines += [f"- {x}" for x in tx["forbidden_during_transaction"]]

    lines += ["", "## Postflight validation", ""]
    lines += [f"- {x}" for x in tx["postflight"]]

    lines += ["", "## Preserve without rewrite", ""]
    lines += [f"- {x}" for x in data["preserve_without_rewrite"]]

    lines += ["", "## Follow-up after rename", ""]
    lines += [
        f"- **{x['item']}** — {x['scope']} (`{x['timing']}`)"
        for x in data["post_rename_followup"]
    ]

    lines += [
        "",
        "## Authority boundary",
        "",
        "- Controlled repository rename transaction: `authorized`",
        "- Historical evidence rewriting: `not authorized`",
        "- Papers 2–4 migration: `not authorized`",
        "- Runtime replacement: `not authorized`",
        "- Plugin relocation: `not authorized`",
        "- gcs-core rename: `not authorized`",
        "- Protected storage rename: `not authorized`",
        "- Legacy alias removal: `not authorized`",
        "",
    ]

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(repo: Path, data: dict[str, Any]) -> list[Path]:
    out = repo / OUTPUT_REL
    out.mkdir(parents=True, exist_ok=True)

    yaml_path = out / "rename_readiness_acceptance.yaml"
    md_path = out / "rename_readiness_acceptance.md"

    yaml_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(data), encoding="utf-8")
    return [yaml_path, md_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()

    data = build_acceptance()
    validate(data, load_schema(repo))

    print(f"[m8.6] status={data['status']}")
    print(f"[m8.6] old_repository={data['identity_transition']['old_repository_name']}")
    print(f"[m8.6] new_repository={data['identity_transition']['new_repository_name']}")
    print(f"[m8.6] direct_rename_blocker_count={data['readiness_findings']['direct_rename_blocker_count']}")
    print(
        "[m8.6] repository_rename_transaction_authorized="
        f"{data['authority_boundary']['repository_rename_transaction_authorized']}"
    )

    if args.check_only:
        print("[m8.6] check-only: no generated artifacts written")
        return 0

    for path in write_outputs(repo, data):
        print(f"[m8.6] wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
