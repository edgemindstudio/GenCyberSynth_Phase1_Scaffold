#!/usr/bin/env python3
"""
M8.4.4 — TrustForge structural compatibility acceptance.

This milestone records that the M8.4.3 scaffold materialization:
- changed no historical authority roots,
- changed no historical Paper 2–4 or Slurm paths,
- introduced no runtime import migration,
- introduced no execution-path migration,
- preserved the full TrustForge test suite.

It does not authorize further migration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


OUTPUT_REL = Path("studies/repository_migration/audits/m8_4_4")
SCHEMA_REL = OUTPUT_REL / "structural_compatibility_acceptance.schema.yaml"

M8_4_3_ACCEPTANCE_COMMIT = "f1db89a"
M8_4_2_ACCEPTANCE_COMMIT = "954ad5f"

EXPECTED_MATERIALIZATION_PATHS = [
    "hpc/README.md",
    "studies/paper02_conditioning_audit/README.md",
    "studies/paper03_augmentation_regimes/README.md",
    "studies/paper04_selective_policies/README.md",
    "tests/trustforge/test_safe_scaffold_materialization_m8_4_3.py",
    "tools/materialize_safe_scaffolds_m8_4_3.py",
]

HISTORICAL_AUTHORITY_ROOTS = [
    "slurm",
    "papers/paper2_conditional_generation_done_right",
    "papers/paper3_when_does_synth_help",
    "papers/paper4_selective_synth_policies",
]

PROTECTED_HISTORICAL_PATHS = [
    "slurm",
    "run_gcs.slurm",
    "run_suite.slurm",
    "run_tuning.slurm",
    "papers/paper2_conditional_generation_done_right",
    "papers/paper3_when_does_synth_help",
    "papers/paper4_selective_synth_policies",
]


def acceptance_record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "milestone": "M8.4.4",
        "title": "TrustForge Structural Compatibility Acceptance",
        "status": "STRUCTURAL_COMPATIBILITY_ACCEPTED",
        "source_materialization": {
            "m8_4_3_acceptance_commit": M8_4_3_ACCEPTANCE_COMMIT,
            "prior_readiness_commit": M8_4_2_ACCEPTANCE_COMMIT,
        },
        "acceptance_claims": {
            "materialization_scope_exact": True,
            "historical_authority_roots_present": True,
            "historical_paths_unchanged": True,
            "runtime_import_migration_observed": False,
            "execution_path_migration_observed": False,
            "historical_authority_transfer_observed": False,
            "full_trustforge_suite_green": True,
        },
        "authority_boundary": {
            "paper_migration_authorized": False,
            "runtime_replacement_authorized": False,
            "plugin_relocation_authorized": False,
            "historical_cleanup_authorized": False,
            "repository_rename_authorized": False,
        },
        "principles": [
            "SCAFFOLD_EXISTENCE != AUTHORITY_TRANSFER",
            "REFERENCE_IN_AUDIT != RUNTIME_DEPENDENCY",
            "REFERENCE_IN_TEST != EXECUTION_PATH",
            "NO_HISTORICAL_DIFF == HISTORICAL_AUTHORITY_PRESERVED",
            "STRUCTURAL_COMPATIBILITY_ACCEPTANCE != PAPER_MIGRATION_ACCEPTANCE",
        ],
        "materialization_scope": EXPECTED_MATERIALIZATION_PATHS,
        "historical_authority_roots": HISTORICAL_AUTHORITY_ROOTS,
        "protected_historical_paths": PROTECTED_HISTORICAL_PATHS,
        "observed_validation": {
            "trustforge_tests_passed": 658,
            "trustforge_subtests_passed": 3,
            "git_diff_check_clean": True,
        },
    }


def load_schema(repo: Path) -> dict[str, Any]:
    data = yaml.safe_load((repo / SCHEMA_REL).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("M8.4.4 schema must be a mapping")
    return data


def validate(data: dict[str, Any], schema: dict[str, Any]) -> None:
    for key in schema["required_top_level"]:
        if key not in data:
            raise ValueError(f"missing top-level field: {key}")

    claims = data["acceptance_claims"]
    if claims["materialization_scope_exact"] is not True:
        raise ValueError("materialization scope must be exact")
    if claims["historical_authority_roots_present"] is not True:
        raise ValueError("historical authority roots must remain present")
    if claims["historical_paths_unchanged"] is not True:
        raise ValueError("historical paths must remain unchanged")
    if claims["runtime_import_migration_observed"] is not False:
        raise ValueError("runtime import migration must not be observed")
    if claims["execution_path_migration_observed"] is not False:
        raise ValueError("execution path migration must not be observed")
    if claims["historical_authority_transfer_observed"] is not False:
        raise ValueError("authority transfer must not be observed")
    if claims["full_trustforge_suite_green"] is not True:
        raise ValueError("TrustForge suite must remain green")

    if data["materialization_scope"] != EXPECTED_MATERIALIZATION_PATHS:
        raise ValueError("materialization scope mismatch")

    if data["historical_authority_roots"] != HISTORICAL_AUTHORITY_ROOTS:
        raise ValueError("historical authority roots mismatch")

    if data["protected_historical_paths"] != PROTECTED_HISTORICAL_PATHS:
        raise ValueError("protected historical paths mismatch")

    for value in data["authority_boundary"].values():
        if value is not False:
            raise ValueError("M8.4.4 must not authorize downstream migration actions")


def render_markdown(data: dict[str, Any]) -> str:
    claims = data["acceptance_claims"]
    obs = data["observed_validation"]

    lines = [
        "# M8.4.4 — TrustForge Structural Compatibility Acceptance",
        "",
        f"**Status:** `{data['status']}`",
        f"**M8.4.3 acceptance commit:** `{data['source_materialization']['m8_4_3_acceptance_commit']}`",
        f"**M8.4.2 readiness commit:** `{data['source_materialization']['prior_readiness_commit']}`",
        "",
        "## Acceptance",
        "",
        "The M8.4.3 scaffold materialization is accepted as structurally compatible.",
        "",
        "It changed repository structure only within the previously authorized scaffold scope and did not transfer scientific authority, migrate Papers 2–4, replace runtime imports, or alter historical execution paths.",
        "",
        "## Acceptance claims",
        "",
        f"- Exact materialization scope: `{str(claims['materialization_scope_exact']).lower()}`",
        f"- Historical authority roots present: `{str(claims['historical_authority_roots_present']).lower()}`",
        f"- Historical protected paths unchanged: `{str(claims['historical_paths_unchanged']).lower()}`",
        f"- Runtime import migration observed: `{str(claims['runtime_import_migration_observed']).lower()}`",
        f"- Execution-path migration observed: `{str(claims['execution_path_migration_observed']).lower()}`",
        f"- Historical authority transfer observed: `{str(claims['historical_authority_transfer_observed']).lower()}`",
        f"- Full TrustForge suite green: `{str(claims['full_trustforge_suite_green']).lower()}`",
        "",
        "## Principles",
        "",
    ]
    lines += [f"- `{x}`" for x in data["principles"]]

    lines += ["", "## Materialization scope", ""]
    lines += [f"- `{x}`" for x in data["materialization_scope"]]

    lines += ["", "## Historical authority roots preserved", ""]
    lines += [f"- `{x}`" for x in data["historical_authority_roots"]]

    lines += ["", "## Protected historical paths unchanged", ""]
    lines += [f"- `{x}`" for x in data["protected_historical_paths"]]

    lines += [
        "",
        "## Validation observed",
        "",
        f"- TrustForge tests passed: **{obs['trustforge_tests_passed']}**",
        f"- TrustForge subtests passed: **{obs['trustforge_subtests_passed']}**",
        f"- `git diff --check` clean: `{str(obs['git_diff_check_clean']).lower()}`",
        "",
        "## Authority boundary",
        "",
        "This acceptance closes structural compatibility for M8.4.3 only.",
        "It does not authorize Paper 2–4 migration, runtime replacement, plugin relocation, historical cleanup, or repository rename.",
        "",
    ]

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(repo: Path, data: dict[str, Any]) -> list[Path]:
    out = repo / OUTPUT_REL
    out.mkdir(parents=True, exist_ok=True)

    yaml_path = out / "structural_compatibility_acceptance.yaml"
    md_path = out / "structural_compatibility_acceptance.md"

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

    data = acceptance_record()
    validate(data, load_schema(repo))

    print(f"[m8.4.4] status={data['status']}")
    print(f"[m8.4.4] materialization_paths={len(data['materialization_scope'])}")
    print(f"[m8.4.4] historical_authority_roots={len(data['historical_authority_roots'])}")
    print(f"[m8.4.4] trustforge_tests_passed={data['observed_validation']['trustforge_tests_passed']}")
    print(f"[m8.4.4] trustforge_subtests_passed={data['observed_validation']['trustforge_subtests_passed']}")

    if args.check_only:
        print("[m8.4.4] check-only: no generated artifacts written")
        return 0

    for path in write_outputs(repo, data):
        print(f"[m8.4.4] wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
