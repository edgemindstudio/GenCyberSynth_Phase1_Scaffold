#!/usr/bin/env python3
"""
M8.5 — TrustForge old-name / old-path dependency inventory.

This milestone classifies rename-related references without mutating them.

It distinguishes:
- actual repository rename blockers,
- portability blockers,
- historical provenance that must be preserved,
- compatibility references,
- documentation/test/tooling references,
- safe-to-retain legacy names,
- items requiring later review.

It does not rename the repository, remote, submodule, files, packages, or
historical evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


OUTPUT_REL = Path("studies/repository_migration/audits/m8_5")
SCHEMA_REL = OUTPUT_REL / "old_name_path_dependency_inventory.schema.yaml"

M8_4_ACCEPTANCE_COMMIT = "a11445a"

CLASSES = {
    "RENAME_BLOCKER",
    "PORTABILITY_BLOCKER",
    "HISTORICAL_REFERENCE",
    "COMPATIBILITY_REFERENCE",
    "DOCUMENTATION_ONLY",
    "TEST_ONLY",
    "TOOLING_ONLY",
    "SAFE_TO_RETAIN",
    "REQUIRES_REVIEW",
}


def record(
    dependency_id: str,
    classification: str,
    surface: str,
    examples: list[str],
    rename_impact: str,
    action_before_repo_rename: str,
    mutate_now: bool,
    rationale: str,
) -> dict[str, Any]:
    return {
        "dependency_id": dependency_id,
        "classification": classification,
        "surface": surface,
        "examples": examples,
        "rename_impact": rename_impact,
        "action_before_repo_rename": action_before_repo_rename,
        "mutate_now": mutate_now,
        "rationale": rationale,
    }


DEPENDENCIES = [
    record(
        "M8.5-001",
        "RENAME_BLOCKER",
        "Git remote origin",
        [
            "origin git@github.com:edgemindstudio/GenCyberSynth_Phase1_Scaffold.git",
        ],
        "A GitHub repository rename requires origin to resolve to the new canonical repository identity.",
        "Update origin only as part of the governed repository rename procedure after rename-readiness acceptance.",
        False,
        "This is the clearest live external identity tied directly to the old repository name.",
    ),
    record(
        "M8.5-002",
        "HISTORICAL_REFERENCE",
        "Paper 1 canonical experiment/study repository fields",
        [
            "studies/paper01_benchmark/experiments/** repository: edgemindstudio/GenCyberSynth_Phase1_Scaffold",
            "studies/paper01_benchmark/study.yaml",
            "studies/paper01_benchmark/lineage.yaml",
            "schemas/examples/paper01/**",
        ],
        "Historical provenance remains valid after repository rename.",
        "Preserve historical repository identity in accepted Paper 1 evidence; do not bulk-rewrite.",
        False,
        "Scientific execution/provenance identity must not be rewritten merely because the repository receives a new current name.",
    ),
    record(
        "M8.5-003",
        "HISTORICAL_REFERENCE",
        "Paper 1 experiment generator repository constant",
        [
            'scripts/generate_paper01_experiments.py: REPOSITORY = "edgemindstudio/GenCyberSynth_Phase1_Scaffold"',
        ],
        "The generator reproduces accepted historical Paper 1 contracts.",
        "Keep historical identity unless a future versioned generator explicitly separates historical_repository from current_repository.",
        False,
        "Changing this constant now would risk changing deterministic accepted Paper 1 contracts.",
    ),
    record(
        "M8.5-004",
        "HISTORICAL_REFERENCE",
        "Paper 2 result/evidence absolute repository paths",
        [
            "papers/paper2_conditional_generation_done_right/results/tables/*.csv",
            "papers/paper2_conditional_generation_done_right/results/tables/*.json",
        ],
        "Paths may become non-resolving after local directory rename, but they remain historical evidence of where outputs were produced.",
        "Preserve in place. During Paper 2 migration, represent current portable locations separately from historical recorded paths.",
        False,
        "Observed absolute paths inside scientific outputs are evidence, not configuration defaults.",
    ),
    record(
        "M8.5-005",
        "PORTABILITY_BLOCKER",
        "Paper 2 active scripts/configs/Slurm hard-coded external gencys roots",
        [
            "papers/paper2_conditional_generation_done_right/configs/**",
            "papers/paper2_conditional_generation_done_right/scripts/**",
            "papers/paper2_conditional_generation_done_right/slurm/**",
        ],
        "These paths do not depend on the repository directory name, but they prevent portable execution and must be handled during Paper 2 migration.",
        "Do not change historical files now. Replace with TrustForge storage/path contracts in the future Paper 2 canonical migration.",
        False,
        "This is a portability problem rather than a repository-rename-name dependency.",
    ),
    record(
        "M8.5-006",
        "COMPATIBILITY_REFERENCE",
        "gcs-core submodule and gcs_core imports",
        [
            ".gitmodules -> https://github.com/edgemindstudio/gcs-core.git",
            ".github/workflows/smoke.yml",
            "eval/runner.py",
            "eval/val_common.py",
            "model-template/**",
        ],
        "The independent gcs-core identity does not inherently block renaming the parent repository.",
        "Retain until runtime/plugin migration explicitly replaces or retires the dependency.",
        False,
        "gcs-core is a separate compatibility/runtime dependency, not the parent repository identity.",
    ),
    record(
        "M8.5-007",
        "COMPATIBILITY_REFERENCE",
        "Legacy GCS_* environment aliases",
        [
            "src/trustforge/storage/paths.py",
            "scripts/trustforge_doctor.py",
            "tests/trustforge/test_storage_paths.py",
        ],
        "Legacy aliases are intentionally supported alongside TRUSTFORGE_* variables.",
        "Retain through repository rename; retire only through a separately governed compatibility decision.",
        False,
        "Compatibility aliases are deliberate and already subordinate to canonical TRUSTFORGE_* variables.",
    ),
    record(
        "M8.5-008",
        "SAFE_TO_RETAIN",
        "External protected ~/gencys data/artifact root",
        [
            "/home/bruno.fonkeng/gencys/data",
            "/home/bruno.fonkeng/gencys/artifacts*",
            "artifacts -> /home/bruno.fonkeng/gencys/artifacts",
        ],
        "External storage root is independent of the repository directory name.",
        "Retain. Do not rename or reorganize protected historical storage as part of repository rename.",
        False,
        "The gencys storage identity is historical/external storage, not the Git repository name.",
    ),
    record(
        "M8.5-009",
        "SAFE_TO_RETAIN",
        "Dynamic repository-root discovery",
        [
            "Path(__file__).resolve().parents[...]",
            "git top-level discovery",
            "Path.cwd()/--repo-root patterns",
            "$(CURDIR)",
            "$SLURM_SUBMIT_DIR",
        ],
        "These mechanisms generally survive a repository directory rename because they derive location dynamically.",
        "Retain, while validating structural-depth assumptions during each paper migration.",
        False,
        "Dynamic location discovery is rename-safe unless directory depth changes.",
    ),
    record(
        "M8.5-010",
        "DOCUMENTATION_ONLY",
        "Current repository branding in README/Runbook/docs/comments",
        [
            "README.md",
            "Runbook.md",
            ".github/git-commit-instructions.md",
            "app/main.py descriptions/comments",
            "common/data.py comments",
            "adapters/base.py comments",
        ],
        "Old branding does not block filesystem/Git rename but would make the renamed repository externally inconsistent.",
        "Update current-facing documentation and CLI branding during the governed rename transition, preserving explicitly historical references.",
        False,
        "These are presentation/current-identity surfaces rather than scientific evidence.",
    ),
    record(
        "M8.5-011",
        "HISTORICAL_REFERENCE",
        "Historical GenCyberSynth references in research lineage/architecture documentation",
        [
            "docs/RESEARCH_LINEAGE.md",
            "docs/TRUSTFORGE_ARCHITECTURE.md",
            "docs/STORAGE_AND_PATHS.md historical examples",
        ],
        "Historical references remain semantically correct after rename.",
        "Preserve statements that explicitly describe the former GenCyberSynth identity; update only text that claims it is the current repository name.",
        False,
        "TrustForge architecture explicitly distinguishes historical GenCyberSynth identity from current framework identity.",
    ),
    record(
        "M8.5-012",
        "REQUIRES_REVIEW",
        "Root Makefile current-facing GenCyberSynth branding and legacy defaults",
        [
            'Makefile: "# Makefile — Unified main Makefile for GenCyberSynth"',
            'Makefile: echo "GenCyberSynth unified Makefile"',
            "Makefile: gcs-core schema path",
            "Makefile: ~/gencys historical artifact example",
        ],
        "Branding should change eventually, while dependency/default lines may remain compatibility behavior.",
        "Split current-branding changes from compatibility semantics before rename; no bulk edit.",
        False,
        "One file contains both current-facing identity and historical/compatibility behavior.",
    ),
    record(
        "M8.5-013",
        "REQUIRES_REVIEW",
        "model-template legacy product identity",
        [
            "model-template/CITATION.cff",
            "model-template/pyproject.toml",
            "model-template/Makefile",
            "model-template/scripts/slurm_array_example.sh",
        ],
        "Template naming may communicate an obsolete product identity to new model/plugin projects.",
        "Review during plugin architecture normalization, not as an automatic repository rename substitution.",
        False,
        "The template combines package naming, dependency naming, citation identity, and job-name conventions.",
    ),
    record(
        "M8.5-014",
        "SAFE_TO_RETAIN",
        "gcs-core submodule URL",
        [
            "https://github.com/edgemindstudio/gcs-core.git",
        ],
        "Independent submodule URL is unaffected by parent repository rename.",
        "No action required for parent repository rename.",
        False,
        "Separate repository identity must not be renamed merely to match the parent.",
    ),
    record(
        "M8.5-015",
        "TEST_ONLY",
        "TrustForge tests referencing legacy aliases and historical identities",
        [
            "tests/trustforge/test_storage_paths.py",
            "tests/trustforge/test_doctor.py",
            "tests/trustforge/test_paper01_*",
        ],
        "Tests intentionally lock compatibility and historical behavior.",
        "Retain unless the governed behavior itself changes.",
        False,
        "Test references are not production dependency evidence by themselves.",
    ),
    record(
        "M8.5-016",
        "TOOLING_ONLY",
        "Migration/audit tooling references",
        [
            "tools/**",
            "scripts/trustforge_doctor.py",
            "scripts/trustforge_foundation_check.py",
        ],
        "Tooling may reference both historical and canonical identities by design.",
        "Evaluate per tool; do not globally replace old-name strings.",
        False,
        "Audit tooling must often recognize legacy state in order to validate migration.",
    ),
]


def build_inventory() -> dict[str, Any]:
    counts = {name: 0 for name in sorted(CLASSES)}
    for item in DEPENDENCIES:
        counts[item["classification"]] += 1

    blockers = [
        item["dependency_id"]
        for item in DEPENDENCIES
        if item["classification"] == "RENAME_BLOCKER"
    ]

    return {
        "schema_version": 1,
        "milestone": "M8.5",
        "title": "TrustForge Old-Name and Old-Path Dependency Inventory",
        "status": "RENAME_DEPENDENCIES_INVENTORIED",
        "source_acceptance": {
            "m8_4_acceptance_commit": M8_4_ACCEPTANCE_COMMIT,
        },
        "authority_boundary": {
            "repository_rename_authorized": False,
            "remote_update_authorized": False,
            "historical_rewrite_authorized": False,
            "paper_migration_authorized": False,
            "compatibility_removal_authorized": False,
        },
        "principles": [
            "OLD_NAME_PRESENT != RENAME_BLOCKER",
            "HISTORICAL_PATH != CURRENT_CONFIGURATION",
            "HISTORICAL_REPOSITORY_IDENTITY != CURRENT_REPOSITORY_IDENTITY",
            "GCS_COMPATIBILITY != TRUSTFORGE_CANONICAL_IDENTITY",
            "PORTABILITY_BLOCKER != REPOSITORY_NAME_BLOCKER",
            "BULK_RENAME != GOVERNED_MIGRATION",
        ],
        "summary": {
            "dependency_group_count": len(DEPENDENCIES),
            "class_counts": counts,
            "rename_blocker_ids": blockers,
            "rename_blocker_count": len(blockers),
        },
        "dependencies": DEPENDENCIES,
    }


def load_schema(repo: Path) -> dict[str, Any]:
    data = yaml.safe_load((repo / SCHEMA_REL).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("M8.5 schema must be a mapping")
    return data


def validate(data: dict[str, Any], schema: dict[str, Any]) -> None:
    for key in schema["required_top_level"]:
        if key not in data:
            raise ValueError(f"missing top-level field: {key}")

    allowed = set(schema["allowed_classifications"])
    seen: set[str] = set()

    for item in data["dependencies"]:
        for key in schema["dependency_required"]:
            if key not in item:
                raise ValueError(f"{item.get('dependency_id')}: missing {key}")
        if item["dependency_id"] in seen:
            raise ValueError(f"duplicate dependency id: {item['dependency_id']}")
        seen.add(item["dependency_id"])

        if item["classification"] not in allowed:
            raise ValueError(f"unknown classification: {item['classification']}")
        if item["mutate_now"] is not False:
            raise ValueError("M8.5 inventory must remain read-only")
        if not item["examples"]:
            raise ValueError(f"{item['dependency_id']}: examples required")

    if any(data["authority_boundary"].values()):
        raise ValueError("M8.5 inventory must not authorize rename or migration")

    if "M8.5-001" not in data["summary"]["rename_blocker_ids"]:
        raise ValueError("Git remote origin must be identified as rename blocker")


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# M8.5 — TrustForge Old-Name / Old-Path Dependency Inventory",
        "",
        f"**Status:** `{data['status']}`",
        f"**M8.4 acceptance commit:** `{data['source_acceptance']['m8_4_acceptance_commit']}`",
        "",
        "## Authority boundary",
        "",
        "This milestone inventories and classifies rename dependencies only.",
        "It does not authorize repository rename, remote mutation, historical rewriting, paper migration, or compatibility removal.",
        "",
        "## Principles",
        "",
    ]
    lines += [f"- `{p}`" for p in data["principles"]]

    lines += [
        "",
        "## Summary",
        "",
        f"Dependency groups: **{data['summary']['dependency_group_count']}**",
        f"Rename blockers identified: **{data['summary']['rename_blocker_count']}**",
        "",
        "| Classification | Groups |",
        "|---|---:|",
    ]
    lines += [
        f"| `{name}` | {count} |"
        for name, count in data["summary"]["class_counts"].items()
    ]

    lines += [
        "",
        "## Rename blockers",
        "",
    ]
    lines += [f"- `{x}`" for x in data["summary"]["rename_blocker_ids"]]

    lines += ["", "## Dependency groups", ""]

    for item in data["dependencies"]:
        lines += [
            f"### `{item['dependency_id']}`",
            "",
            f"- Classification: `{item['classification']}`",
            f"- Surface: {item['surface']}",
            f"- Rename impact: {item['rename_impact']}",
            f"- Action before repository rename: {item['action_before_repo_rename']}",
            "- Mutate now: `false`",
            f"- Rationale: {item['rationale']}",
            "- Examples:",
        ]
        lines += [f"  - `{x}`" for x in item["examples"]]
        lines += [""]

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(repo: Path, data: dict[str, Any]) -> list[Path]:
    out = repo / OUTPUT_REL
    out.mkdir(parents=True, exist_ok=True)

    yaml_path = out / "old_name_path_dependency_inventory.yaml"
    md_path = out / "old_name_path_dependency_inventory.md"

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

    data = build_inventory()
    validate(data, load_schema(repo))

    print(f"[m8.5] status={data['status']}")
    print(f"[m8.5] dependency_group_count={data['summary']['dependency_group_count']}")
    print(f"[m8.5] rename_blocker_count={data['summary']['rename_blocker_count']}")
    for cls, count in data["summary"]["class_counts"].items():
        print(f"[m8.5] {cls}={count}")

    if args.check_only:
        print("[m8.5] check-only: no generated artifacts written")
        return 0

    for path in write_outputs(repo, data):
        print(f"[m8.5] wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
