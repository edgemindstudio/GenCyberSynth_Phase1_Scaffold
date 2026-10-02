#!/usr/bin/env python3
"""
M8.3 — TrustForge paper isolation contracts.

This milestone defines explicit dependency boundaries for Papers 2–4.
It does not move code, rewrite historical runtime, or migrate study files.

Generated artifacts:
    studies/repository_migration/audits/m8_3/paper_isolation_contracts.yaml
    studies/repository_migration/audits/m8_3/paper_isolation_contracts.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


OUTPUT_REL = Path("studies/repository_migration/audits/m8_3")
SCHEMA_REL = OUTPUT_REL / "paper_isolation_contracts.schema.yaml"

M8_2_ACCEPTANCE_COMMIT = "d42e022"

DEPENDENCY_CLASSES = {
    "CORE_ALLOWED",
    "TRANSITIONAL_ALLOWED",
    "STUDY_LOCAL_ONLY",
    "FORBIDDEN_CROSS_PAPER",
    "HISTORICAL_COMPATIBILITY_ONLY",
    "DEFERRED",
}


def rule(
    rule_id: str,
    paper: str,
    dependency_class: str,
    target: str,
    rationale: str,
    enforcement: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "paper": paper,
        "dependency_class": dependency_class,
        "target": target,
        "rationale": rationale,
        "enforcement": enforcement,
        "migration_authorized": False,
        "implementation_move_authorized": False,
        "evidence": evidence,
    }


RULES: list[dict[str, Any]] = [
    rule(
        "P2-CORE-001",
        "paper02",
        "CORE_ALLOWED",
        "src/trustforge/provenance/",
        "Paper 2 may depend on paper-neutral TrustForge provenance contracts.",
        "allowed",
        ["M8.2 classifies provenance as CORE_CONTRACT."],
    ),
    rule(
        "P2-CORE-002",
        "paper02",
        "CORE_ALLOWED",
        "src/trustforge/storage/",
        "Paper 2 may depend on paper-neutral TrustForge storage/path contracts.",
        "allowed",
        ["M8.2 classifies storage resolution as CORE_CONTRACT."],
    ),
    rule(
        "P2-TRANS-001",
        "paper02",
        "TRANSITIONAL_ALLOWED",
        "app/main.py",
        "Paper 2 historically invokes the shared CLI runtime; this remains temporary compatibility.",
        "allowed_with_migration_debt",
        ["M8.1 observed Paper 2 invocations of python -m app.main.",
         "M8.2 classifies the historical shared CLI as TRANSITIONAL_RUNTIME."],
    ),
    rule(
        "P2-TRANS-002",
        "paper02",
        "TRANSITIONAL_ALLOWED",
        "common/data.py",
        "Paper 2 historically imports shared dataset loading; this is not permanent core ownership.",
        "allowed_with_migration_debt",
        ["Paper 2 scripts directly import common.data.",
         "M8.2 classifies historical shared dataset runtime as TRANSITIONAL_RUNTIME."],
    ),
    rule(
        "P2-LOCAL-001",
        "paper02",
        "STUDY_LOCAL_ONLY",
        "gan/models_acgan.py;gan/train_acgan.py",
        "ACGAN and conditioning-intervention semantics are owned by Paper 2.",
        "must_remain_explicitly_paper_owned",
        ["M8.2 classifies Paper 2 conditioning semantics as STUDY_OWNED."],
    ),
    rule(
        "P2-XPAPER-001",
        "paper02",
        "FORBIDDEN_CROSS_PAPER",
        "papers/paper3_when_does_synth_help/;papers/paper4_selective_synth_policies/",
        "Paper 2 must not import implementation or scientific semantics from later studies.",
        "forbidden",
        ["Paper-specific scientific meaning must remain isolated by study."],
    ),
    rule(
        "P3-CORE-001",
        "paper03",
        "CORE_ALLOWED",
        "src/trustforge/provenance/;src/trustforge/storage/",
        "Paper 3 may depend on paper-neutral TrustForge contracts.",
        "allowed",
        ["M8.2 classifies storage and provenance as CORE_CONTRACT."],
    ),
    rule(
        "P3-TRANS-001",
        "paper03",
        "TRANSITIONAL_ALLOWED",
        "app/main.py;eval/runner.py;common/data.py",
        "Paper 3 historically depends on the shared CLI/evaluator/data runtime.",
        "allowed_with_migration_debt",
        ["M8.1 observed extensive Paper 3 use of python -m app.main.",
         "M8.2 classifies CLI, evaluator, and dataset runtime as TRANSITIONAL_RUNTIME."],
    ),
    rule(
        "P3-LOCAL-001",
        "paper03",
        "STUDY_LOCAL_ONLY",
        "class-restricted synthesis;minority c4/c7 augmentation;budget-regime semantics",
        "Paper 3 scientific interventions must remain Paper 3-owned.",
        "must_remain_explicitly_paper_owned",
        ["M8.2 classifies Paper 3 augmentation semantics as STUDY_OWNED."],
    ),
    rule(
        "P3-XPAPER-001",
        "paper03",
        "FORBIDDEN_CROSS_PAPER",
        "papers/paper2_conditional_generation_done_right/;papers/paper4_selective_synth_policies/",
        "Paper 3 must not silently inherit Paper 2 conditioning or Paper 4 policy semantics.",
        "forbidden",
        ["Paper-specific interventions are not framework defaults."],
    ),
    rule(
        "P4-CORE-001",
        "paper04",
        "CORE_ALLOWED",
        "src/trustforge/provenance/;src/trustforge/storage/",
        "Paper 4 may depend on paper-neutral TrustForge contracts.",
        "allowed",
        ["M8.2 classifies storage and provenance as CORE_CONTRACT."],
    ),
    rule(
        "P4-TRANS-001",
        "paper04",
        "TRANSITIONAL_ALLOWED",
        "app/main.py;eval/runner.py;common/data.py",
        "Paper 4 historically depends on shared runtime surfaces.",
        "allowed_with_migration_debt",
        ["M8.1 observed Paper 4 use of python -m app.main.",
         "M8.2 classifies CLI/evaluator/data runtime as transitional."],
    ),
    rule(
        "P4-LOCAL-001",
        "paper04",
        "STUDY_LOCAL_ONLY",
        "keep-all;confidence filtering;top-k;class-repair",
        "Selective synthetic-data policy semantics are owned by Paper 4.",
        "must_remain_explicitly_paper_owned",
        ["M8.2 classifies Paper 4 policy semantics as STUDY_OWNED."],
    ),
    rule(
        "P4-XPAPER-001",
        "paper04",
        "FORBIDDEN_CROSS_PAPER",
        "papers/paper2_conditional_generation_done_right/;papers/paper3_when_does_synth_help/",
        "Paper 4 must not import scientific implementation directly from Papers 2 or 3.",
        "forbidden",
        ["Study ownership is distinct from shared framework ownership."],
    ),
    rule(
        "ALL-HIST-001",
        "all",
        "HISTORICAL_COMPATIBILITY_ONLY",
        "historical configs;legacy Slurm scripts;historical path aliases",
        "Historical reproduction surfaces may be read/used for compatibility but must not become new native defaults.",
        "compatibility_only",
        ["M8.2 classifies legacy path/config surfaces as HISTORICAL_COMPATIBILITY."],
    ),
    rule(
        "ALL-PLUGIN-001",
        "all",
        "DEFERRED",
        "gan/;vae/;diffusion/;autoregressive/;gaussianmixture/;restrictedboltzmann/;maskedautoflow/",
        "Current model packages are plugins conceptually, but migration/isolation mechanics are deferred.",
        "deferred",
        ["M8.2 classifies historical model implementations as MODEL_PLUGIN."],
    ),
]


def build_contract() -> dict[str, Any]:
    counts = {name: 0 for name in sorted(DEPENDENCY_CLASSES)}
    for item in RULES:
        counts[item["dependency_class"]] += 1

    return {
        "schema_version": 1,
        "milestone": "M8.3",
        "title": "TrustForge Paper Isolation Contracts",
        "status": "ISOLATION_BOUNDARIES_DEFINED",
        "source_boundary": {
            "m8_2_acceptance_commit": M8_2_ACCEPTANCE_COMMIT,
        },
        "authority_boundary": {
            "file_moves_authorized": False,
            "runtime_rewrites_authorized": False,
            "paper_migrations_authorized": False,
            "cross_paper_reuse_authorized_by_default": False,
        },
        "principles": [
            "PAPER_SPECIFIC_SCIENCE != SHARED_FRAMEWORK_DEFAULT",
            "TRANSITIONAL_ALLOWED != PERMANENTLY_ALLOWED",
            "CROSS_PAPER_IMPORT != SHARED_CORE",
            "HISTORICAL_COMPATIBILITY != NATIVE_DEFAULT",
            "MODEL_PLUGIN != FRAMEWORK_CORE",
        ],
        "dependency_class_definitions": {
            "CORE_ALLOWED": "Paper may depend on paper-neutral TrustForge core contracts/interfaces.",
            "TRANSITIONAL_ALLOWED": "Existing historical dependency may remain temporarily, but creates migration debt.",
            "STUDY_LOCAL_ONLY": "Scientific behavior must remain explicitly owned by the paper/study.",
            "FORBIDDEN_CROSS_PAPER": "Direct dependency on another paper's scientific implementation is prohibited.",
            "HISTORICAL_COMPATIBILITY_ONLY": "May be used only to reproduce or interpret historical execution/evidence.",
            "DEFERRED": "Boundary recognized, but migration mechanics are intentionally unresolved.",
        },
        "summary": {
            "rule_count": len(RULES),
            "class_counts": counts,
        },
        "rules": RULES,
    }


def load_schema(repo: Path) -> dict[str, Any]:
    path = repo / SCHEMA_REL
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("M8.3 schema must be a mapping")
    return data


def validate(data: dict[str, Any], schema: dict[str, Any]) -> None:
    for key in schema["required_top_level"]:
        if key not in data:
            raise ValueError(f"missing top-level field: {key}")

    allowed = set(schema["allowed_dependency_classes"])
    seen: set[str] = set()

    for item in data["rules"]:
        for key in schema["rule_required"]:
            if key not in item:
                raise ValueError(f"{item.get('rule_id')}: missing {key}")

        if item["rule_id"] in seen:
            raise ValueError(f"duplicate rule id: {item['rule_id']}")
        seen.add(item["rule_id"])

        if item["dependency_class"] not in allowed:
            raise ValueError(f"unknown dependency class: {item['dependency_class']}")

        if item["migration_authorized"] is not False:
            raise ValueError("M8.3 must not authorize migration")
        if item["implementation_move_authorized"] is not False:
            raise ValueError("M8.3 must not authorize implementation moves")

        if not item["target"].strip():
            raise ValueError(f"{item['rule_id']}: empty target")
        if not item["evidence"]:
            raise ValueError(f"{item['rule_id']}: evidence required")
        if not item["rationale"].strip():
            raise ValueError(f"{item['rule_id']}: rationale required")


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# M8.3 — TrustForge Paper Isolation Contracts",
        "",
        f"**Status:** `{data['status']}`",
        f"**M8.2 acceptance commit:** `{data['source_boundary']['m8_2_acceptance_commit']}`",
        "",
        "## Authority boundary",
        "",
        "M8.3 defines dependency/isolation rules only.",
        "It does not authorize file moves, runtime rewrites, or paper migrations.",
        "",
        "## Principles",
        "",
    ]

    lines += [f"- `{p}`" for p in data["principles"]]
    lines += [
        "",
        "## Summary",
        "",
        f"Rules defined: **{data['summary']['rule_count']}**",
        "",
        "| Dependency class | Rules |",
        "|---|---:|",
    ]
    lines += [
        f"| `{name}` | {count} |"
        for name, count in data["summary"]["class_counts"].items()
    ]

    lines += ["", "## Isolation rules", ""]

    for item in data["rules"]:
        lines += [
            f"### `{item['rule_id']}`",
            "",
            f"- Paper: `{item['paper']}`",
            f"- Dependency class: `{item['dependency_class']}`",
            f"- Target: `{item['target']}`",
            f"- Enforcement: `{item['enforcement']}`",
            "- Migration authorized: `false`",
            "- Implementation move authorized: `false`",
            f"- Rationale: {item['rationale']}",
            "- Evidence:",
        ]
        lines += [f"  - {entry}" for entry in item["evidence"]]
        lines += [""]

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(repo: Path, data: dict[str, Any]) -> list[Path]:
    out = repo / OUTPUT_REL
    out.mkdir(parents=True, exist_ok=True)

    yaml_path = out / "paper_isolation_contracts.yaml"
    md_path = out / "paper_isolation_contracts.md"

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

    schema = load_schema(repo)
    data = build_contract()
    validate(data, schema)

    print(f"[m8.3] status={data['status']}")
    print(f"[m8.3] rule_count={data['summary']['rule_count']}")
    for cls, count in data["summary"]["class_counts"].items():
        print(f"[m8.3] {cls}={count}")

    if args.check_only:
        print("[m8.3] check-only: no generated artifacts written")
        return 0

    for path in write_outputs(repo, data):
        print(f"[m8.3] wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
