#!/usr/bin/env python3
"""
M7.1 — Paper 1 P2 Consumer / Interface Regression Audit.

Read-only source audit.

Purpose:
- verify the exact 13 P2 surfaces recorded by M6.5.2;
- classify the 12 downstream consumers by interface shape;
- verify that none of those 12 directly select Paper 1 authority from
  timestamped summaries / latest aliases;
- identify the two consumers that remain filesystem-coupled through
  manifest_path;
- record the Makefile as a dedicated orchestration/safety-review surface.

This audit does not modify any P2 consumer or historical artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


POLICY_REL = Path(
    "studies/paper01_benchmark/audits/m6_5_2/"
    "paper01_consumer_migration_policy.json"
)

P2_JSONL_CONSUMERS = (
    "scripts/jsonl_to_csv.py",
    "scripts/plots/_common.py",
    "scripts/plots/core/calibration_curves.py",
    "scripts/plots/core/pareto_downstream_vs_similarity.py",
    "scripts/plots/core/per_class_delta_f1.py",
    "scripts/plots/diversity/ms_ssim_hist.py",
    "scripts/plots/diversity/nn_distance_distrib.py",
    "scripts/plots/hparams/ablation_bars.py",
    "scripts/plots/hparams/parallel_coords.py",
    "scripts/plots/imbalance/class_counts_before_after.py",
    "scripts/plots/imbalance/simple_stats_sanity.py",
    "scripts/plots/qual/class_triptychs.py",
)

FILESYSTEM_COUPLED = {
    "scripts/plots/imbalance/simple_stats_sanity.py",
    "scripts/plots/qual/class_triptychs.py",
}

MAKEFILE = "Makefile"

FORBIDDEN_AUTHORITY_MARKERS = (
    "latest.json",
    "paper1.json",
    "summary_*.json",
)

INTERFACE_MARKERS = {
    "scripts/jsonl_to_csv.py": (
        "phase1_summaries.jsonl",
        "run_id",
        "model",
    ),
    "scripts/plots/_common.py": (
        "DEFAULT_JSONL",
        "read_jsonl",
        "pd.json_normalize",
    ),
    "scripts/plots/core/calibration_curves.py": (
        "read_jsonl",
        "utility_real_plus_synth.ece",
    ),
    "scripts/plots/core/pareto_downstream_vs_similarity.py": (
        "read_jsonl",
        "metrics.downstream.macro_f1",
    ),
    "scripts/plots/core/per_class_delta_f1.py": (
        "read_jsonl",
        "metrics.real_only.per_class_f1.",
        "metrics.real_plus_synth.per_class_f1.",
    ),
    "scripts/plots/diversity/ms_ssim_hist.py": (
        "read_jsonl",
        "metrics.ms_ssim",
    ),
    "scripts/plots/diversity/nn_distance_distrib.py": (
        "read_jsonl",
        "metrics.nn_dists",
    ),
    "scripts/plots/hparams/ablation_bars.py": (
        "read_jsonl",
        "--metric",
        "--ablate-by",
    ),
    "scripts/plots/hparams/parallel_coords.py": (
        "read_jsonl",
        "--cols",
    ),
    "scripts/plots/imbalance/class_counts_before_after.py": (
        "read_jsonl",
        "counts.per_class.real.",
        "counts.per_class.synth.",
    ),
    "scripts/plots/imbalance/simple_stats_sanity.py": (
        "read_jsonl",
        "manifest_path",
        "_fallback_synth_glob",
    ),
    "scripts/plots/qual/class_triptychs.py": (
        "read_jsonl",
        "manifest_path",
        "_fallback_synth_by_class",
    ),
}

MAKEFILE_MARKERS = (
    "paper1-jsonl",
    "paper1_build",
    "clean-summaries",
    "clean-synth",
    "PHASE1_DATASET",
    "PHASE1_SEED",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


class P2AuditError(RuntimeError):
    pass


def _load_policy(repo_root: Path) -> dict[str, Any]:
    path = repo_root / POLICY_REL
    if not path.is_file():
        raise P2AuditError(f"missing policy: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _p2_records(policy: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            record
            for record in policy["records"]
            if record.get("priority") == "P2"
        ],
        key=lambda record: record["path"],
    )


def _check_policy(policy: dict[str, Any]) -> CheckResult:
    observed = {
        record["path"]
        for record in _p2_records(policy)
    }
    expected = set(P2_JSONL_CONSUMERS) | {MAKEFILE}
    return CheckResult(
        name="p2_policy_inventory",
        passed=observed == expected,
        detail=(
            f"observed={len(observed)} expected={len(expected)} "
            f"missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        ),
    )


def _check_consumer(
    repo_root: Path,
    relative: str,
) -> list[CheckResult]:
    path = repo_root / relative
    if not path.is_file():
        return [
            CheckResult(
                name=f"consumer_exists:{relative}",
                passed=False,
                detail="file missing",
            )
        ]

    source = path.read_text(encoding="utf-8")

    checks = [
        CheckResult(
            name=f"interface_markers:{relative}",
            passed=all(
                marker in source
                for marker in INTERFACE_MARKERS[relative]
            ),
            detail=(
                "expected JSONL/interface markers present"
            ),
        )
    ]

    # P2 policy says these consumers should consume a canonical-derived
    # consolidated contract, not choose historical authority themselves.
    present_forbidden = [
        marker
        for marker in FORBIDDEN_AUTHORITY_MARKERS
        if marker in source
    ]
    checks.append(
        CheckResult(
            name=f"no_direct_summary_authority:{relative}",
            passed=not present_forbidden,
            detail=(
                "no direct latest/paper1/timestamped-summary authority markers"
                if not present_forbidden
                else f"found={present_forbidden}"
            ),
        )
    )

    return checks


def _check_makefile(repo_root: Path) -> list[CheckResult]:
    path = repo_root / MAKEFILE
    if not path.is_file():
        return [
            CheckResult(
                name="makefile_exists",
                passed=False,
                detail="Makefile missing",
            )
        ]

    source = path.read_text(encoding="utf-8")
    missing = [
        marker
        for marker in MAKEFILE_MARKERS
        if marker not in source
    ]

    # This is not a migration pass/fail judgment. Presence of historical
    # paper1_build and destructive clean targets is precisely why M7.3 exists.
    return [
        CheckResult(
            name="makefile_review_surface_present",
            passed=not missing,
            detail=(
                "mixed canonical/historical orchestration markers present; "
                "dedicated M7.3 review required"
                if not missing
                else f"missing expected markers={missing}"
            ),
        )
    ]


def run_audit(repo_root: Path) -> dict[str, Any]:
    policy = _load_policy(repo_root)
    checks: list[CheckResult] = []

    checks.append(_check_policy(policy))

    for relative in P2_JSONL_CONSUMERS:
        checks.extend(
            _check_consumer(
                repo_root,
                relative,
            )
        )

    checks.extend(_check_makefile(repo_root))

    passed = all(check.passed for check in checks)

    consumers = []
    for relative in P2_JSONL_CONSUMERS:
        consumers.append(
            {
                "path": relative,
                "interface": (
                    "jsonl_plus_manifest_filesystem"
                    if relative in FILESYSTEM_COUPLED
                    else "jsonl_only"
                ),
                "authority_role": (
                    "consumer_only"
                ),
                "m7_2_runtime_required": (
                    relative in FILESYSTEM_COUPLED
                ),
            }
        )

    return {
        "audit": "M7.1",
        "status": "PASS" if passed else "FAIL",
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
            "p2_total": 13,
            "jsonl_consumers": 12,
            "jsonl_only_consumers": 10,
            "filesystem_coupled_consumers": 2,
            "makefile_review_surfaces": 1,
        },
        "checks": [
            asdict(check)
            for check in checks
        ],
        "consumers": consumers,
        "makefile": {
            "path": MAKEFILE,
            "classification": "review_before_migration",
            "reason": (
                "Mixed historical and canonical orchestration, including "
                "paper1_build plus destructive clean targets. M7.3 must "
                "review safety and authority handoff before any modification."
            ),
        },
        "handoff": {
            "m7_2": (
                "Verify canonical-derived JSONL compatibility across the "
                "12 P2 consumers, with explicit runtime checks for the two "
                "manifest/filesystem-coupled visualizers."
            ),
            "m7_3": (
                "Review Makefile orchestration separately; do not modify it "
                "automatically."
            ),
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M7.1 — Paper 1 P2 Consumer / Interface Regression Audit",
        "",
        f"**Status:** {report['status']}",
        "",
        "## Summary",
        "",
        f"- P2 surfaces: {report['summary']['p2_total']}",
        f"- JSONL consumers: {report['summary']['jsonl_consumers']}",
        f"- JSONL-only consumers: {report['summary']['jsonl_only_consumers']}",
        (
            "- Filesystem-coupled JSONL consumers: "
            f"{report['summary']['filesystem_coupled_consumers']}"
        ),
        "- Makefile safety-review surfaces: 1",
        (
            f"- Checks: {report['summary']['checks_passed']}/"
            f"{report['summary']['checks_total']} passed"
        ),
        "",
        "## Consumer matrix",
        "",
        "| Consumer | Interface | Authority role | M7.2 runtime required |",
        "|---|---|---|---|",
    ]

    for consumer in report["consumers"]:
        lines.append(
            f"| `{consumer['path']}` | "
            f"{consumer['interface']} | "
            f"{consumer['authority_role']} | "
            f"{'yes' if consumer['m7_2_runtime_required'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Makefile handoff",
            "",
            report["makefile"]["reason"],
            "",
            "## M7 handoff",
            "",
            f"- **M7.2:** {report['handoff']['m7_2']}",
            f"- **M7.3:** {report['handoff']['m7_3']}",
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
        / "paper01_p2_interface_audit.json"
    )
    md_path = (
        out_dir
        / "paper01_p2_interface_audit.md"
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
            "audits/m7_1"
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
        "[audit] P2="
        f"{report['summary']['p2_total']} "
        "jsonl_only="
        f"{report['summary']['jsonl_only_consumers']} "
        "filesystem_coupled="
        f"{report['summary']['filesystem_coupled_consumers']}"
    )
    print(f"[audit] wrote {json_path}")
    print(f"[audit] wrote {md_path}")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
