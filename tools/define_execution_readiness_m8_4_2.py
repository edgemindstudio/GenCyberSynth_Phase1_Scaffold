#!/usr/bin/env python3
"""
M8.4.2 — TrustForge normalization execution readiness.

This milestone classifies which structural normalization actions are safe to
authorize now, which remain guarded compatibility work, and which remain
migration-prohibited.

It does not move historical runtime or migrate Papers 2–4.

Generated artifacts:
    studies/repository_migration/audits/m8_4_2/execution_readiness.yaml
    studies/repository_migration/audits/m8_4_2/execution_readiness.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


OUTPUT_REL = Path("studies/repository_migration/audits/m8_4_2")
SCHEMA_REL = OUTPUT_REL / "execution_readiness.schema.yaml"

M8_4_1_ACCEPTANCE_COMMIT = "22f3df2"

READINESS_CLASSES = {
    "AUTHORIZED_SAFE_SCAFFOLD",
    "PRESERVE_AS_IS",
    "GUARDED_COMPATIBILITY",
    "MIGRATION_NOT_YET_AUTHORIZED",
    "DEFERRED",
}


def action(
    action_id: str,
    target: str,
    readiness_class: str,
    action_type: str,
    rationale: str,
    preconditions: list[str],
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "target": target,
        "readiness_class": readiness_class,
        "action_type": action_type,
        "authorized": readiness_class == "AUTHORIZED_SAFE_SCAFFOLD",
        "historical_move_authorized": False,
        "import_rewrite_authorized": False,
        "paper_migration_authorized": False,
        "rationale": rationale,
        "preconditions": preconditions,
        "evidence": evidence,
    }


ACTIONS: list[dict[str, Any]] = [
    action(
        "SAFE-001",
        "src/trustforge/data/",
        "PRESERVE_AS_IS",
        "existing_native_scaffold",
        "The native namespace already exists and contains only tracked package scaffolding.",
        ["No implementation migration is required to preserve this namespace."],
        ["Only src/trustforge/data/__init__.py is tracked.",
         "No tracked Python import currently references trustforge.data."],
    ),
    action(
        "SAFE-002",
        "src/trustforge/evaluation/",
        "PRESERVE_AS_IS",
        "existing_native_scaffold",
        "The native namespace already exists and is not yet coupled to runtime behavior.",
        ["Historical eval/runner.py must remain untouched."],
        ["Only src/trustforge/evaluation/__init__.py is tracked.",
         "No tracked Python import currently references trustforge.evaluation."],
    ),
    action(
        "SAFE-003",
        "src/trustforge/orchestration/",
        "PRESERVE_AS_IS",
        "existing_native_scaffold",
        "The native namespace already exists without replacing app/main.py.",
        ["Historical app/main.py remains compatibility runtime."],
        ["Only src/trustforge/orchestration/__init__.py is tracked.",
         "No tracked Python import currently references trustforge.orchestration."],
    ),
    action(
        "SAFE-004",
        "src/trustforge/models/",
        "PRESERVE_AS_IS",
        "existing_native_scaffold",
        "The native namespace exists for future plugin contracts without moving model implementations.",
        ["Historical model packages remain in place."],
        ["Only src/trustforge/models/__init__.py is tracked.",
         "No tracked Python import currently references trustforge.models."],
    ),
    action(
        "SAFE-005",
        "src/trustforge/policies/",
        "PRESERVE_AS_IS",
        "existing_native_scaffold",
        "The namespace exists without absorbing Paper 4 policy semantics.",
        ["Paper 4 policy implementations remain study-owned."],
        ["Only src/trustforge/policies/__init__.py is tracked.",
         "No tracked Python import currently references trustforge.policies."],
    ),
    action(
        "SAFE-006",
        "src/trustforge/augmentation/",
        "PRESERVE_AS_IS",
        "existing_native_scaffold",
        "The namespace exists without absorbing Paper 3 augmentation semantics.",
        ["Paper 3 augmentation implementations remain study-owned."],
        ["Only src/trustforge/augmentation/__init__.py is tracked.",
         "No tracked Python import currently references trustforge.augmentation."],
    ),
    action(
        "SAFE-007",
        "hpc/",
        "AUTHORIZED_SAFE_SCAFFOLD",
        "create_directory_scaffold",
        "The native hpc/ root is part of accepted target architecture and does not currently exist.",
        [
            "Creation must contain no migrated historical Slurm scripts.",
            "Creation must not change root slurm/ behavior.",
            "Creation must not change execution semantics.",
        ],
        [
            "docs/TRUSTFORGE_ARCHITECTURE.md already names hpc/ in the target structure.",
            "M8.4.1 target structure includes hpc/.",
            "No native hpc/ directory currently exists.",
        ],
    ),
    action(
        "SAFE-008",
        "studies/paper02_conditioning_audit/",
        "AUTHORIZED_SAFE_SCAFFOLD",
        "create_directory_scaffold",
        "The canonical Paper 2 study destination is accepted, but migration is not.",
        [
            "Scaffold must contain no copied/moved historical Paper 2 files.",
            "Historical papers/paper2_conditional_generation_done_right/ remains authoritative.",
        ],
        [
            "M8.4.1 defines studies/paper02_conditioning_audit/ as the future target.",
            "The destination is currently absent.",
            "Current references are planning/test references only.",
        ],
    ),
    action(
        "SAFE-009",
        "studies/paper03_augmentation_regimes/",
        "AUTHORIZED_SAFE_SCAFFOLD",
        "create_directory_scaffold",
        "The canonical Paper 3 study destination is accepted, but migration is not.",
        [
            "Scaffold must contain no copied/moved historical Paper 3 files.",
            "Historical papers/paper3_when_does_synth_help/ remains authoritative.",
        ],
        [
            "M8.4.1 defines studies/paper03_augmentation_regimes/ as the future target.",
            "The destination is currently absent.",
            "Current references are planning/test references only.",
        ],
    ),
    action(
        "SAFE-010",
        "studies/paper04_selective_policies/",
        "AUTHORIZED_SAFE_SCAFFOLD",
        "create_directory_scaffold",
        "The canonical Paper 4 study destination is accepted, but migration is not.",
        [
            "Scaffold must contain no copied/moved historical Paper 4 files.",
            "Historical papers/paper4_selective_synth_policies/ remains authoritative.",
        ],
        [
            "M8.4.1 defines studies/paper04_selective_policies/ as the future target.",
            "The destination is currently absent.",
            "Current references are planning/test references only.",
        ],
    ),
    action(
        "GUARD-001",
        "app/",
        "GUARDED_COMPATIBILITY",
        "historical_runtime",
        "Shared CLI remains active compatibility runtime for Papers 2–4.",
        ["No relocation or import rewrite before paper migration contracts exist."],
        ["M8.2 classifies app/main.py as TRANSITIONAL_RUNTIME.",
         "M8.3 allows it only with migration debt."],
    ),
    action(
        "GUARD-002",
        "adapters/",
        "GUARDED_COMPATIBILITY",
        "historical_runtime",
        "Concrete adapters remain bound to historical model packages.",
        ["No relocation into src/trustforge/models/ yet."],
        ["M8.2 separates adapter interface ownership from current concrete adapters."],
    ),
    action(
        "GUARD-003",
        "common/",
        "GUARDED_COMPATIBILITY",
        "historical_runtime",
        "Historical dataset runtime remains shared compatibility code.",
        ["No replacement by trustforge.data until consumer migrations are proven."],
        ["M8.3 allows common/data.py only as transitional dependency."],
    ),
    action(
        "GUARD-004",
        "eval/",
        "GUARDED_COMPATIBILITY",
        "historical_runtime",
        "Historical evaluator remains compatibility runtime with paper-era semantics.",
        ["No replacement by trustforge.evaluation until evaluator contracts are implemented and paper migrations validated."],
        ["M8.2 classifies eval/runner.py as TRANSITIONAL_RUNTIME."],
    ),
    action(
        "GUARD-005",
        "configs/",
        "GUARDED_COMPATIBILITY",
        "mixed_root",
        "Current root is mixed historical/native candidate space and must be partitioned later.",
        ["No historical config may be silently reclassified as native default."],
        ["M8.4.1 classifies configs/ as MIGRATE_LATER."],
    ),
    action(
        "GUARD-006",
        "slurm/;run_gcs.slurm;run_suite.slurm;run_tuning.slurm",
        "GUARDED_COMPATIBILITY",
        "historical_hpc",
        "Historical/global Slurm surfaces remain compatibility infrastructure.",
        ["No script is moved into hpc/ merely because hpc/ now exists."],
        ["M8.3 classifies historical Slurm usage as compatibility-only."],
    ),
    action(
        "BLOCK-001",
        "papers/paper2_conditional_generation_done_right/",
        "MIGRATION_NOT_YET_AUTHORIZED",
        "paper_migration",
        "Paper 2 historical study has not yet undergone canonical migration.",
        ["Paper 2 migration requires dedicated evidence mapping and compatibility validation."],
        ["M8.4.1 classifies Paper 2 as MIGRATE_LATER."],
    ),
    action(
        "BLOCK-002",
        "papers/paper3_when_does_synth_help/",
        "MIGRATION_NOT_YET_AUTHORIZED",
        "paper_migration",
        "Paper 3 historical study has not yet undergone canonical migration.",
        ["Paper 3 migration requires dedicated evidence mapping and compatibility validation."],
        ["M8.4.1 classifies Paper 3 as MIGRATE_LATER."],
    ),
    action(
        "BLOCK-003",
        "papers/paper4_selective_synth_policies/",
        "MIGRATION_NOT_YET_AUTHORIZED",
        "paper_migration",
        "Paper 4 historical study has not yet undergone canonical migration.",
        ["Paper 4 migration requires dedicated evidence mapping and compatibility validation."],
        ["M8.4.1 classifies Paper 4 as MIGRATE_LATER."],
    ),
    action(
        "BLOCK-004",
        "gan/;vae/;diffusion/;autoregressive/;gaussianmixture/;restrictedboltzmann/;maskedautoflow/",
        "MIGRATION_NOT_YET_AUTHORIZED",
        "plugin_migration",
        "Model packages are conceptually plugins but relocation mechanics remain unresolved.",
        ["Plugin interface must be implemented before any model package relocation."],
        ["M8.2 classifies model implementations as MODEL_PLUGIN.",
         "M8.3 explicitly defers plugin migration mechanics."],
    ),
    action(
        "DEFER-001",
        "src/trustforge/uncertainty/",
        "DEFERRED",
        "future_capability",
        "Uncertainty/calibration is outside current Papers 1–4 normalization needs.",
        ["No implementation work required in M8.4.2."],
        ["M8.4.1 classifies uncertainty namespace as DEFERRED."],
    ),
]


def build_readiness() -> dict[str, Any]:
    counts = {name: 0 for name in sorted(READINESS_CLASSES)}
    for record in ACTIONS:
        counts[record["readiness_class"]] += 1

    return {
        "schema_version": 1,
        "milestone": "M8.4.2",
        "title": "TrustForge Normalization Execution Readiness",
        "status": "EXECUTION_READINESS_DEFINED",
        "source_structural_plan": {
            "m8_4_1_acceptance_commit": M8_4_1_ACCEPTANCE_COMMIT,
        },
        "authority_boundary": {
            "safe_scaffold_creation_authorized": True,
            "historical_moves_authorized": False,
            "import_rewrites_authorized": False,
            "paper_migrations_authorized": False,
            "plugin_relocations_authorized": False,
            "historical_cleanup_authorized": False,
        },
        "principles": [
            "SCAFFOLD_CREATION != MIGRATION",
            "DESTINATION_EXISTENCE != AUTHORITY_TRANSFER",
            "HISTORICAL_SOURCE_REMAINS_AUTHORITATIVE_UNTIL_MIGRATION_ACCEPTED",
            "NATIVE_NAMESPACE_EXISTENCE != IMPLEMENTATION_MIGRATION",
            "COMPATIBILITY_GUARD_PRECEDES_RUNTIME_REPLACEMENT",
        ],
        "summary": {
            "action_count": len(ACTIONS),
            "class_counts": counts,
        },
        "actions": ACTIONS,
    }


def load_schema(repo: Path) -> dict[str, Any]:
    data = yaml.safe_load((repo / SCHEMA_REL).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("M8.4.2 schema must be a mapping")
    return data


def validate(data: dict[str, Any], schema: dict[str, Any]) -> None:
    for key in schema["required_top_level"]:
        if key not in data:
            raise ValueError(f"missing top-level field: {key}")

    allowed = set(schema["allowed_readiness_classes"])
    seen: set[str] = set()

    for record in data["actions"]:
        for key in schema["action_required"]:
            if key not in record:
                raise ValueError(f"{record.get('action_id')}: missing {key}")

        if record["action_id"] in seen:
            raise ValueError(f"duplicate action id: {record['action_id']}")
        seen.add(record["action_id"])

        if record["readiness_class"] not in allowed:
            raise ValueError(f"unknown readiness class: {record['readiness_class']}")

        expected_authorized = (
            record["readiness_class"] == "AUTHORIZED_SAFE_SCAFFOLD"
        )
        if record["authorized"] is not expected_authorized:
            raise ValueError(
                f"{record['action_id']}: authorized flag disagrees with readiness class"
            )

        if record["historical_move_authorized"] is not False:
            raise ValueError("M8.4.2 cannot authorize historical moves")
        if record["import_rewrite_authorized"] is not False:
            raise ValueError("M8.4.2 cannot authorize import rewrites")
        if record["paper_migration_authorized"] is not False:
            raise ValueError("M8.4.2 cannot authorize paper migration")

        if not record["target"].strip():
            raise ValueError(f"{record['action_id']}: target required")
        if not record["preconditions"]:
            raise ValueError(f"{record['action_id']}: preconditions required")
        if not record["evidence"]:
            raise ValueError(f"{record['action_id']}: evidence required")


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# M8.4.2 — TrustForge Normalization Execution Readiness",
        "",
        f"**Status:** `{data['status']}`",
        f"**M8.4.1 acceptance commit:** `{data['source_structural_plan']['m8_4_1_acceptance_commit']}`",
        "",
        "## Authority boundary",
        "",
        "M8.4.2 authorizes only explicitly identified safe scaffold creation.",
        "It does not authorize historical moves, import rewrites, paper migrations, plugin relocation, or cleanup.",
        "",
        "## Principles",
        "",
    ]
    lines += [f"- `{p}`" for p in data["principles"]]

    lines += [
        "",
        "## Summary",
        "",
        f"Actions classified: **{data['summary']['action_count']}**",
        "",
        "| Readiness class | Actions |",
        "|---|---:|",
    ]
    lines += [
        f"| `{name}` | {count} |"
        for name, count in data["summary"]["class_counts"].items()
    ]

    lines += ["", "## Actions", ""]

    for record in data["actions"]:
        lines += [
            f"### `{record['action_id']}`",
            "",
            f"- Target: `{record['target']}`",
            f"- Readiness class: `{record['readiness_class']}`",
            f"- Action type: `{record['action_type']}`",
            f"- Authorized: `{str(record['authorized']).lower()}`",
            "- Historical move authorized: `false`",
            "- Import rewrite authorized: `false`",
            "- Paper migration authorized: `false`",
            f"- Rationale: {record['rationale']}",
            "- Preconditions:",
        ]
        lines += [f"  - {x}" for x in record["preconditions"]]
        lines += ["- Evidence:"]
        lines += [f"  - {x}" for x in record["evidence"]]
        lines += [""]

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(repo: Path, data: dict[str, Any]) -> list[Path]:
    out = repo / OUTPUT_REL
    out.mkdir(parents=True, exist_ok=True)

    yaml_path = out / "execution_readiness.yaml"
    md_path = out / "execution_readiness.md"

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

    data = build_readiness()
    schema = load_schema(repo)
    validate(data, schema)

    print(f"[m8.4.2] status={data['status']}")
    print(f"[m8.4.2] action_count={data['summary']['action_count']}")
    for cls, count in data["summary"]["class_counts"].items():
        print(f"[m8.4.2] {cls}={count}")

    if args.check_only:
        print("[m8.4.2] check-only: no generated artifacts written")
        return 0

    for path in write_outputs(repo, data):
        print(f"[m8.4.2] wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
