#!/usr/bin/env python3
"""
M7.3 — Makefile Safety and Authority Review.

Read-only source/governance audit.

The Makefile is a mixed orchestration surface. This audit classifies the
Paper-1/Phase-1 targets without modifying them.

Permanent distinctions:
    CANONICAL CONSUMER != HISTORICAL PRODUCER
    HISTORICAL WORKFLOW != CANONICAL SCIENTIFIC AUTHORITY
    DESTRUCTIVE MAINTENANCE != MIGRATION
    TARGET EXISTS != TARGET IS SAFE FOR CANONICAL USE

Classification vocabulary:
- canonical_safe:
    explicit canonical Paper 1 identity and canonical consumer path.
- historical_preserved:
    historical producer/mutator intentionally preserved.
- mixed_authority:
    target can lead into legacy/default authority despite migrated components.
- destructive:
    target removes artifacts/files and requires explicit operator intent.
- generic_nonpaper1:
    not part of Paper 1 authority migration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any


POLICY_REL = Path(
    "studies/paper01_benchmark/audits/m6_5_2/"
    "paper01_consumer_migration_policy.json"
)

TARGET_CLASSIFICATION = {
    "phase1_freeze": "historical_preserved",
    "phase1_scores": "canonical_safe",
    "phase1_check": "canonical_safe",
    "phase1_backfill": "historical_preserved",
    "phase1_gate": "canonical_safe",
    "normalize-summaries": "historical_preserved",
    "paper1-jsonl": "mixed_authority",
    "table": "mixed_authority",
    "scores-csv": "mixed_authority",
    "report": "mixed_authority",
    "figs-core": "mixed_authority",
    "figs-diversity": "mixed_authority",
    "figs-imbalance": "mixed_authority",
    "figs-qual": "mixed_authority",
    "figs-hparams": "mixed_authority",
    "figs-all": "mixed_authority",
    "paper1_prepare": "historical_preserved",
    "paper1_build": "mixed_authority",
    "paper1": "mixed_authority",
    "clean-summaries": "destructive",
    "clean-synth": "destructive",
}

CANONICAL_TARGETS = {
    "phase1_scores",
    "phase1_check",
    "phase1_gate",
}

HISTORICAL_TARGETS = {
    "phase1_freeze",
    "phase1_backfill",
    "normalize-summaries",
    "paper1_prepare",
}

MIXED_TARGETS = {
    "paper1-jsonl",
    "table",
    "scores-csv",
    "report",
    "figs-core",
    "figs-diversity",
    "figs-imbalance",
    "figs-qual",
    "figs-hparams",
    "figs-all",
    "paper1_build",
    "paper1",
}

DESTRUCTIVE_TARGETS = {
    "clean-summaries",
    "clean-synth",
}

CANONICAL_MARKERS = {
    "phase1_scores": (
        'test -n "$(PHASE1_DATASET)"',
        'test -n "$(PHASE1_SEED)"',
        'TRUSTFORGE_ARTIFACTS_ROOT',
        'PYTHONPATH="$(CURDIR)/src"',
        'tools/build_phase1_scores.py',
    ),
    "phase1_check": (
        'test -n "$(PHASE1_DATASET)"',
        'test -n "$(PHASE1_SEED)"',
        'PYTHONPATH="$(CURDIR)/src"',
        'tools/check_phase1_integrity.py',
    ),
    "phase1_gate": (
        'test -n "$(PHASE1_DATASET)"',
        'test -n "$(PHASE1_SEED)"',
        'TRUSTFORGE_ARTIFACTS_ROOT',
        'tools/check_phase1_integrity.py',
        'tools/build_phase1_scores.py',
    ),
}

HISTORICAL_MARKERS = {
    "phase1_freeze": (
        "tools/freeze_phase1_snapshots.py",
        "PHASE1_SUMMARY_NAME",
        "PHASE1_MANIFEST_NAME",
    ),
    "phase1_backfill": (
        "scripts/backfill_kid_and_downstream.py",
        "PHASE1_SUMMARY_NAME",
        "PHASE1_MANIFEST_NAME",
    ),
    "normalize-summaries": (
        "scripts/normalize_summaries.py",
    ),
    "paper1_prepare": (
        "phase1_freeze",
        "historical snapshot prepared",
    ),
}

MIXED_MARKERS = {
    "paper1-jsonl": (
        "normalize-summaries",
        "tools/build_paper1_jsonl.py",
    ),
    "table": (
        "paper1-jsonl",
        "scripts/collect_scores.py",
    ),
    "scores-csv": (
        "paper1-jsonl",
        "scripts/jsonl_to_csv.py",
    ),
    "report": (
        "table",
        "scripts/phase1_report.py",
    ),
    "figs-core": (
        "paper1-jsonl",
        "scripts.plots.core.pareto_downstream_vs_similarity",
    ),
    "figs-diversity": (
        "paper1-jsonl",
        "scripts.plots.diversity.ms_ssim_hist",
    ),
    "figs-imbalance": (
        "paper1-jsonl",
        "scripts.plots.imbalance.class_counts_before_after",
    ),
    "figs-qual": (
        "paper1-jsonl",
    ),
    "figs-hparams": (
        "paper1-jsonl",
        "scripts.plots.hparams.parallel_coords",
    ),
    "figs-all": (
        "figs-core",
        "figs-diversity",
        "figs-imbalance",
        "figs-qual",
        "figs-hparams",
    ),
    "paper1_build": (
        "paper1-jsonl",
        "scores-csv",
        "figs-all",
        "report",
    ),
    "paper1": (
        "paper1_prepare",
        "paper1_build",
    ),
}

DESTRUCTIVE_MARKERS = {
    "clean-summaries": (
        "rm -f",
        "latest.json",
        "summary_*.json",
    ),
    "clean-synth": (
        "find artifacts",
        "-delete",
        "manifest",
    ),
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


class MakefileAuditError(RuntimeError):
    pass


def _load_policy(repo_root: Path) -> dict[str, Any]:
    path = repo_root / POLICY_REL
    if not path.is_file():
        raise MakefileAuditError(
            f"missing M6.5.2 policy: {path}"
        )
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _makefile_policy_record(
    policy: dict[str, Any],
) -> dict[str, Any]:
    matches = [
        record
        for record in policy["records"]
        if record.get("path") == "Makefile"
    ]
    if len(matches) != 1:
        raise MakefileAuditError(
            "expected exactly one Makefile policy record"
        )
    return matches[0]


def _target_blocks(
    source: str,
) -> dict[str, str]:
    lines = source.splitlines()
    starts: list[tuple[int, str]] = []

    for index, line in enumerate(lines):
        match = re.match(
            r"^([A-Za-z0-9_.-]+)\s*:"
            r"(?:\s*(.*))?$",
            line,
        )
        if match:
            starts.append(
                (index, match.group(1))
            )

    blocks: dict[str, str] = {}
    for pos, (index, name) in enumerate(starts):
        end = (
            starts[pos + 1][0]
            if pos + 1 < len(starts)
            else len(lines)
        )
        blocks[name] = "\n".join(
            lines[index:end]
        )

    return blocks


def _check_partition() -> CheckResult:
    expected = (
        CANONICAL_TARGETS
        | HISTORICAL_TARGETS
        | MIXED_TARGETS
        | DESTRUCTIVE_TARGETS
    )
    observed = set(
        TARGET_CLASSIFICATION
    )

    disjoint = (
        len(
            CANONICAL_TARGETS
            & HISTORICAL_TARGETS
        )
        == 0
        and len(
            CANONICAL_TARGETS
            & MIXED_TARGETS
        )
        == 0
        and len(
            CANONICAL_TARGETS
            & DESTRUCTIVE_TARGETS
        )
        == 0
        and len(
            HISTORICAL_TARGETS
            & MIXED_TARGETS
        )
        == 0
        and len(
            HISTORICAL_TARGETS
            & DESTRUCTIVE_TARGETS
        )
        == 0
        and len(
            MIXED_TARGETS
            & DESTRUCTIVE_TARGETS
        )
        == 0
    )

    return CheckResult(
        name="classification_partition",
        passed=(
            disjoint
            and expected == observed
        ),
        detail=(
            f"targets={len(observed)} "
            f"canonical={len(CANONICAL_TARGETS)} "
            f"historical={len(HISTORICAL_TARGETS)} "
            f"mixed={len(MIXED_TARGETS)} "
            f"destructive={len(DESTRUCTIVE_TARGETS)}"
        ),
    )


def _check_policy(
    policy: dict[str, Any],
) -> list[CheckResult]:
    record = _makefile_policy_record(
        policy
    )

    return [
        CheckResult(
            name="policy_priority_p2",
            passed=(
                record.get("priority") == "P2"
            ),
            detail=(
                f"priority={record.get('priority')}"
            ),
        ),
        CheckResult(
            name="policy_review_before_migration",
            passed=(
                record.get("treatment")
                == "REVIEW_BEFORE_MIGRATION"
            ),
            detail=(
                f"treatment={record.get('treatment')}"
            ),
        ),
        CheckResult(
            name="policy_no_historical_mutation",
            passed=(
                "historical_artifacts_modified=NO"
                in record.get("constraints", [])
            ),
            detail=(
                "historical artifacts must remain unmodified"
            ),
        ),
        CheckResult(
            name="policy_no_latest_authority",
            passed=(
                "latest_json_authoritative=NO"
                in record.get("constraints", [])
            ),
            detail=(
                "latest.json cannot be scientific authority"
            ),
        ),
    ]


def _check_markers(
    blocks: dict[str, str],
    mapping: dict[str, tuple[str, ...]],
    category: str,
) -> list[CheckResult]:
    checks: list[CheckResult] = []

    for target, markers in sorted(
        mapping.items()
    ):
        block = blocks.get(target)
        if block is None:
            checks.append(
                CheckResult(
                    name=(
                        f"{category}_target:{target}"
                    ),
                    passed=False,
                    detail="target missing",
                )
            )
            continue

        missing = [
            marker
            for marker in markers
            if marker not in block
        ]
        checks.append(
            CheckResult(
                name=(
                    f"{category}_target:{target}"
                ),
                passed=not missing,
                detail=(
                    "expected markers present"
                    if not missing
                    else f"missing={missing}"
                ),
            )
        )

    return checks


def _check_mixed_identity_gap(
    blocks: dict[str, str],
) -> list[CheckResult]:
    checks: list[CheckResult] = []

    # The root mixed producer is paper1-jsonl. It invokes a migrated dual-mode
    # producer but does not require PHASE1_DATASET/PHASE1_SEED, so default Make
    # invocation remains legacy/historical. Downstream mixed targets inherit
    # that authority shape through dependencies.
    block = blocks.get(
        "paper1-jsonl",
        "",
    )
    requires_dataset = (
        'test -n "$(PHASE1_DATASET)"'
        in block
    )
    requires_seed = (
        'test -n "$(PHASE1_SEED)"'
        in block
    )

    checks.append(
        CheckResult(
            name=(
                "paper1_jsonl_canonical_identity_not_forced"
            ),
            passed=(
                not requires_dataset
                and not requires_seed
            ),
            detail=(
                "paper1-jsonl remains dual/legacy by default; "
                "canonical identity is not forced"
            ),
        )
    )

    return checks


def _decision(
    checks: list[CheckResult],
) -> dict[str, Any]:
    passed = all(
        check.passed
        for check in checks
    )

    return {
        "audit_passed": passed,
        "makefile_modification_authorized": False,
        "recommendation": (
            "NO_CHANGE_DURING_M7_3_REVIEW"
            if passed
            else "REVIEW_FAILED"
        ),
        "reason": (
            "The Makefile safely exposes canonical P0 gates separately, "
            "while historical and destructive targets remain visibly distinct. "
            "However paper1-jsonl and its dependent report/figure/build chain "
            "remain mixed-authority by default. M7.3 records this boundary; it "
            "does not authorize editing historical orchestration. Any future "
            "canonical Make entrypoint should be additive and separately named "
            "rather than silently changing historical targets."
            if passed
            else (
                "The expected authority/safety structure was not confirmed."
            )
        ),
    }


def run_audit(
    repo_root: Path,
) -> dict[str, Any]:
    makefile = repo_root / "Makefile"
    if not makefile.is_file():
        raise MakefileAuditError(
            f"missing Makefile: {makefile}"
        )

    source = makefile.read_text(
        encoding="utf-8"
    )
    blocks = _target_blocks(source)
    policy = _load_policy(repo_root)

    checks: list[CheckResult] = []
    checks.append(_check_partition())
    checks.extend(_check_policy(policy))
    checks.extend(
        _check_markers(
            blocks,
            CANONICAL_MARKERS,
            "canonical",
        )
    )
    checks.extend(
        _check_markers(
            blocks,
            HISTORICAL_MARKERS,
            "historical",
        )
    )
    checks.extend(
        _check_markers(
            blocks,
            MIXED_MARKERS,
            "mixed",
        )
    )
    checks.extend(
        _check_markers(
            blocks,
            DESTRUCTIVE_MARKERS,
            "destructive",
        )
    )
    checks.extend(
        _check_mixed_identity_gap(blocks)
    )

    decision = _decision(checks)

    rows = [
        {
            "target": target,
            "classification": classification,
            "migration_action": {
                "canonical_safe": "preserve",
                "historical_preserved": "preserve_historical",
                "mixed_authority": "do_not_relabel_as_canonical",
                "destructive": "operator_only_no_migration",
            }[classification],
        }
        for target, classification
        in sorted(
            TARGET_CLASSIFICATION.items()
        )
    ]

    return {
        "audit": "M7.3",
        "status": (
            "PASS"
            if decision["audit_passed"]
            else "FAIL"
        ),
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
            "targets_reviewed": len(rows),
            "canonical_safe": len(
                CANONICAL_TARGETS
            ),
            "historical_preserved": len(
                HISTORICAL_TARGETS
            ),
            "mixed_authority": len(
                MIXED_TARGETS
            ),
            "destructive": len(
                DESTRUCTIVE_TARGETS
            ),
        },
        "checks": [
            asdict(check)
            for check in checks
        ],
        "targets": rows,
        "decision": decision,
        "authority_statement": (
            "The Makefile is an orchestration surface, not a source of "
            "scientific authority. Canonical Paper 1 authority remains in "
            "TrustForge linkage/export interfaces. Historical and destructive "
            "Make targets retain their existing semantics and must not be "
            "silently reinterpreted as canonical."
        ),
    }


def _render_markdown(
    report: dict[str, Any],
) -> str:
    summary = report["summary"]
    decision = report["decision"]

    lines = [
        "# M7.3 — Makefile Safety and Authority Review",
        "",
        f"**Status:** {report['status']}",
        "",
        "## Summary",
        "",
        (
            f"- Checks: "
            f"{summary['checks_passed']}/"
            f"{summary['checks_total']} passed"
        ),
        (
            f"- Targets reviewed: "
            f"{summary['targets_reviewed']}"
        ),
        (
            f"- Canonical-safe: "
            f"{summary['canonical_safe']}"
        ),
        (
            f"- Historical-preserved: "
            f"{summary['historical_preserved']}"
        ),
        (
            f"- Mixed-authority: "
            f"{summary['mixed_authority']}"
        ),
        (
            f"- Destructive: "
            f"{summary['destructive']}"
        ),
        "",
        "## Target classification",
        "",
        "| Target | Classification | Migration action |",
        "|---|---|---|",
    ]

    for row in report["targets"]:
        lines.append(
            f"| `{row['target']}` | "
            f"{row['classification']} | "
            f"{row['migration_action']} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                f"- Makefile modification authorized: "
                f"**{'YES' if decision['makefile_modification_authorized'] else 'NO'}**"
            ),
            (
                f"- Recommendation: "
                f"`{decision['recommendation']}`"
            ),
            "",
            decision["reason"],
            "",
            "## Authority statement",
            "",
            report["authority_statement"],
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
        / "paper01_makefile_safety_review.json"
    )
    md_path = (
        out_dir
        / "paper01_makefile_safety_review.md"
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
            "audits/m7_3"
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
        "[audit] checks="
        f"{report['summary']['checks_passed']}/"
        f"{report['summary']['checks_total']}"
    )
    print(
        "[audit] targets="
        f"{report['summary']['targets_reviewed']} "
        "canonical="
        f"{report['summary']['canonical_safe']} "
        "historical="
        f"{report['summary']['historical_preserved']} "
        "mixed="
        f"{report['summary']['mixed_authority']} "
        "destructive="
        f"{report['summary']['destructive']}"
    )
    print(
        "[audit] makefile modification authorized="
        f"{report['decision']['makefile_modification_authorized']}"
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
