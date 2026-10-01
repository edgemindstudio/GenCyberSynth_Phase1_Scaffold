#!/usr/bin/env python3
"""
M7.5 — Final Paper 1 Migration Acceptance.

Read-only acceptance audit.

This audit consolidates governed evidence from:
- M6.5.5 migration closure;
- M7.1 P2 interface audit;
- M7.2A canonical-derived JSONL compatibility;
- M7.2B filesystem-coupled runtime compatibility;
- M7.3 Makefile safety/authority review;
- M7.4 cross-dataset/cross-seed regression matrix.

It does not rerun experiments, select new scientific authority, rewrite
historical artifacts, or modify Makefile orchestration.

Acceptance question:
    Has Paper 1 been migrated far enough that TrustForge can formally accept
    the canonical evidence architecture while preserving historical execution
    artifacts and legacy workflows under their existing authority boundaries?
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


REPORTS = {
    "m6_5_5": Path(
        "studies/paper01_benchmark/audits/m6_5_5/"
        "paper01_migration_closure_audit.json"
    ),
    "m7_1": Path(
        "studies/paper01_benchmark/audits/m7_1/"
        "paper01_p2_interface_audit.json"
    ),
    "m7_2a_ustc": Path(
        "studies/paper01_benchmark/audits/m7_2a/"
        "paper01_jsonl_compat_ustc_tfc2016_seed42.json"
    ),
    "m7_2a_cic": Path(
        "studies/paper01_benchmark/audits/m7_2a/"
        "paper01_jsonl_compat_cicmaldroid2020_seed42.json"
    ),
    "m7_2b_ustc": Path(
        "studies/paper01_benchmark/audits/m7_2b/"
        "paper01_filesystem_runtime_ustc_tfc2016_seed42.json"
    ),
    "m7_2b_cic": Path(
        "studies/paper01_benchmark/audits/m7_2b/"
        "paper01_filesystem_runtime_cicmaldroid2020_seed42.json"
    ),
    "m7_3": Path(
        "studies/paper01_benchmark/audits/m7_3/"
        "paper01_makefile_safety_review.json"
    ),
    "m7_4": Path(
        "studies/paper01_benchmark/audits/m7_4/"
        "paper01_cross_dataset_seed_matrix.json"
    ),
}


class AcceptanceAuditError(RuntimeError):
    """Raised when required governed acceptance evidence is unavailable."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _load_report(
    repo_root: Path,
    key: str,
) -> dict[str, Any]:
    path = repo_root / REPORTS[key]
    if not path.is_file():
        raise AcceptanceAuditError(
            f"missing governed report {key}: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise AcceptanceAuditError(
            f"invalid governed report {key}: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise AcceptanceAuditError(
            f"governed report {key} root must be an object"
        )

    return payload


def _status_check(
    key: str,
    report: dict[str, Any],
) -> CheckResult:
    status = report.get("status")
    return CheckResult(
        name=f"status:{key}",
        passed=status == "PASS",
        detail=f"status={status}",
    )


def _summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    value = report.get("summary")
    return value if isinstance(value, dict) else {}


def _exact(
    *,
    name: str,
    observed: Any,
    expected: Any,
) -> CheckResult:
    return CheckResult(
        name=name,
        passed=observed == expected,
        detail=(
            f"observed={observed!r} "
            f"expected={expected!r}"
        ),
    )


def run_audit(
    repo_root: Path,
) -> dict[str, Any]:
    reports = {
        key: _load_report(repo_root, key)
        for key in REPORTS
    }

    checks: list[CheckResult] = []

    # Every governed prerequisite report must itself be PASS.
    for key, report in reports.items():
        checks.append(
            _status_check(key, report)
        )

    closure = _summary(
        reports["m6_5_5"]
    )
    checks.extend(
        [
            _exact(
                name="m6_5_5_checks",
                observed=(
                    closure.get("checks_passed"),
                    closure.get("checks_total"),
                ),
                expected=(26, 26),
            ),
            _exact(
                name="m6_5_5_migrated_counts",
                observed=(
                    closure.get("p0_count"),
                    closure.get("p1_count"),
                    closure.get("p2_deferred_count"),
                ),
                expected=(2, 10, 13),
            ),
        ]
    )

    p2 = _summary(
        reports["m7_1"]
    )
    checks.extend(
        [
            _exact(
                name="m7_1_p2_partition",
                observed=(
                    p2.get("p2_total"),
                    p2.get("jsonl_consumers"),
                    p2.get("jsonl_only_consumers"),
                    p2.get("filesystem_coupled_consumers"),
                    p2.get("makefile_review_surfaces"),
                ),
                expected=(13, 12, 10, 2, 1),
            ),
            _exact(
                name="m7_1_checks",
                observed=(
                    p2.get("checks_passed"),
                    p2.get("checks_total"),
                ),
                expected=(26, 26),
            ),
        ]
    )

    for key in (
        "m7_2a_ustc",
        "m7_2a_cic",
    ):
        compat = _summary(
            reports[key]
        )
        checks.extend(
            [
                _exact(
                    name=f"{key}_rows",
                    observed=compat.get("rows"),
                    expected=7,
                ),
                _exact(
                    name=f"{key}_checks",
                    observed=(
                        compat.get("checks_passed"),
                        compat.get("checks_total"),
                    ),
                    expected=(37, 37),
                ),
                _exact(
                    name=f"{key}_p2_consumers",
                    observed=compat.get(
                        "p2_consumers"
                    ),
                    expected=12,
                ),
            ]
        )

    for key in (
        "m7_2b_ustc",
        "m7_2b_cic",
    ):
        runtime = _summary(
            reports[key]
        )
        checks.extend(
            [
                _exact(
                    name=f"{key}_rows_models",
                    observed=(
                        runtime.get("rows"),
                        runtime.get("models"),
                    ),
                    expected=(7, 7),
                ),
                _exact(
                    name=f"{key}_checks",
                    observed=(
                        runtime.get("checks_passed"),
                        runtime.get("checks_total"),
                    ),
                    expected=(58, 58),
                ),
            ]
        )

    makefile = reports["m7_3"]
    make_summary = _summary(makefile)
    decision = makefile.get("decision")
    if not isinstance(decision, dict):
        decision = {}

    checks.extend(
        [
            _exact(
                name="m7_3_target_partition",
                observed=(
                    make_summary.get("targets_reviewed"),
                    make_summary.get("canonical_safe"),
                    make_summary.get("historical_preserved"),
                    make_summary.get("mixed_authority"),
                    make_summary.get("destructive"),
                ),
                expected=(21, 3, 4, 12, 2),
            ),
            _exact(
                name="m7_3_checks",
                observed=(
                    make_summary.get("checks_passed"),
                    make_summary.get("checks_total"),
                ),
                expected=(27, 27),
            ),
            _exact(
                name="m7_3_no_makefile_change_authorized",
                observed=decision.get(
                    "makefile_modification_authorized"
                ),
                expected=False,
            ),
            _exact(
                name="m7_3_recommendation",
                observed=decision.get(
                    "recommendation"
                ),
                expected=(
                    "NO_CHANGE_DURING_M7_3_REVIEW"
                ),
            ),
        ]
    )

    matrix = _summary(
        reports["m7_4"]
    )
    checks.extend(
        [
            _exact(
                name="m7_4_views",
                observed=(
                    matrix.get("views_passed"),
                    matrix.get("views"),
                ),
                expected=(6, 6),
            ),
            _exact(
                name="m7_4_experiments",
                observed=matrix.get(
                    "experiments"
                ),
                expected=42,
            ),
            _exact(
                name="m7_4_matrix_checks",
                observed=(
                    matrix.get("checks_passed"),
                    matrix.get("checks_total"),
                ),
                expected=(6, 6),
            ),
            _exact(
                name="m7_4_jsonl_checks",
                observed=(
                    matrix.get(
                        "jsonl_checks_passed"
                    ),
                    matrix.get(
                        "jsonl_checks_total"
                    ),
                ),
                expected=(222, 222),
            ),
            _exact(
                name="m7_4_runtime_checks",
                observed=(
                    matrix.get(
                        "runtime_checks_passed"
                    ),
                    matrix.get(
                        "runtime_checks_total"
                    ),
                ),
                expected=(348, 348),
            ),
        ]
    )

    passed = all(
        check.passed
        for check in checks
    )

    acceptance = (
        "ACCEPTED"
        if passed
        else "NOT_ACCEPTED"
    )

    return {
        "audit": "M7.5",
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "acceptance": acceptance,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(
                check.passed
                for check in checks
            ),
            "checks_failed": sum(
                not check.passed
                for check in checks
            ),
            "governed_reports": len(
                reports
            ),
            "canonical_experiments": 42,
            "dataset_seed_views": 6,
        },
        "checks": [
            asdict(check)
            for check in checks
        ],
        "evidence": {
            key: {
                "path": str(REPORTS[key]),
                "status": reports[key].get(
                    "status"
                ),
            }
            for key in REPORTS
        },
        "acceptance_statement": (
            "Paper 1 canonical evidence architecture is formally accepted "
            "for TrustForge migration. Canonical scientific authority is "
            "explicit and validated across all 42 experiments; downstream "
            "consumer compatibility is verified; historical execution "
            "artifacts and legacy workflows remain preserved under their "
            "existing authority boundaries; and no Makefile reinterpretation "
            "or historical artifact rewrite is authorized by this acceptance."
            if passed
            else (
                "Paper 1 migration is not accepted because one or more "
                "governed prerequisite checks failed."
            )
        ),
        "preserved_distinctions": [
            "SCIENTIFIC_EXPERIMENT_IDENTITY != HISTORICAL_EXECUTION_IDENTITY",
            "ARTIFACT_PRODUCER != ACCEPTED_EVALUATOR != AUTHORITATIVE_RESULT_ROW",
            "latest.json != AUTHORITATIVE_EVIDENCE",
            "FILESYSTEM_FALLBACK != SCIENTIFIC_AUTHORITY",
            "OPERATIONAL_COMPLETION_STATUS != ACCEPTED_SCIENTIFIC_EVIDENCE",
            "MAKEFILE_ORCHESTRATION != SCIENTIFIC_AUTHORITY",
        ],
    }


def _render_markdown(
    report: dict[str, Any],
) -> str:
    summary = report["summary"]

    lines = [
        "# M7.5 — Final Paper 1 Migration Acceptance",
        "",
        f"**Status:** {report['status']}",
        f"**Acceptance:** {report['acceptance']}",
        "",
        "## Acceptance summary",
        "",
        (
            f"- Governed prerequisite reports: "
            f"{summary['governed_reports']}"
        ),
        (
            f"- Acceptance checks: "
            f"{summary['checks_passed']}/"
            f"{summary['checks_total']} passed"
        ),
        (
            f"- Dataset/seed views accepted: "
            f"{summary['dataset_seed_views']}/6"
        ),
        (
            f"- Canonical experiments covered: "
            f"{summary['canonical_experiments']}/42"
        ),
        "",
        "## Governed evidence",
        "",
        "| Stage | Status | Report |",
        "|---|---|---|",
    ]

    for key, evidence in report[
        "evidence"
    ].items():
        lines.append(
            f"| `{key}` | "
            f"{evidence['status']} | "
            f"`{evidence['path']}` |"
        )

    lines.extend(
        [
            "",
            "## Acceptance statement",
            "",
            report["acceptance_statement"],
            "",
            "## Preserved authority distinctions",
            "",
        ]
    )

    for distinction in report[
        "preserved_distinctions"
    ]:
        lines.append(
            f"- `{distinction}`"
        )

    lines.extend(
        [
            "",
            (
                "> M7.5 accepts the migrated canonical evidence architecture. "
                "It does not delete, rewrite, reinterpret, or replace "
                "historical Paper 1 scientific artifacts."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_reports(
    report: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        out_dir
        / "paper01_final_migration_acceptance.json"
    )
    md_path = (
        out_dir
        / "paper01_final_migration_acceptance.md"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        _render_markdown(report),
        encoding="utf-8",
    )

    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "studies/paper01_benchmark/"
            "audits/m7_5"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        args.repo_root
        .expanduser()
        .resolve()
    )
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    report = run_audit(repo_root)
    json_path, md_path = write_reports(
        report,
        out_dir,
    )

    print(
        f"[audit] status={report['status']}"
    )
    print(
        f"[audit] acceptance={report['acceptance']}"
    )
    print(
        "[audit] checks="
        f"{report['summary']['checks_passed']}/"
        f"{report['summary']['checks_total']}"
    )
    print(
        "[audit] governed reports="
        f"{report['summary']['governed_reports']} "
        "views="
        f"{report['summary']['dataset_seed_views']}/6 "
        "experiments="
        f"{report['summary']['canonical_experiments']}/42"
    )
    print(f"[audit] wrote {json_path}")
    print(f"[audit] wrote {md_path}")

    return (
        0
        if report["status"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
