#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess

POLICY_REL = Path("studies/paper01_benchmark/audits/m6_5_2/paper01_consumer_migration_policy.json")
MIGRATION_BASELINE_COMMIT = "a450c37"
PROTECTED_PATHS = ("eval/runner.py", "tools/freeze_phase1_snapshots.py")

EXPECTED_P0 = {
    "tools/build_phase1_scores.py",
    "tools/check_phase1_integrity.py",
}
EXPECTED_P1 = {
    "scripts/build_jsonl.sh",
    "scripts/metrics/aggregate.py",
    "scripts/metrics/print_cfid_table.py",
    "scripts/phase1_html.py",
    "scripts/phase1_report.py",
    "scripts/summaries_to_jsonl.py",
    "scripts/tuning_dashboard.py",
    "scripts/utils/backfill_counts.py",
    "tools/aggregate_phase1.py",
    "tools/build_paper1_jsonl.py",
}
EXPECTED_P2 = {
    "Makefile",
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
}

SOURCE_MARKERS = {
    "tools/build_phase1_scores.py": ("load_paper01_consumer_view",),
    "tools/check_phase1_integrity.py": ("load_paper01_consumer_view",),
    "scripts/build_jsonl.sh": ("PHASE1_DATASET", "PHASE1_SEED", "summaries_to_jsonl.py"),
    "scripts/summaries_to_jsonl.py": ("load_paper01_export_view", "paper01_export_view_to_legacy_jsonl"),
    "scripts/utils/backfill_counts.py": ("--canonical-input", "validate_canonical_input"),
    "tools/aggregate_phase1.py": ("--canonical", "load_paper01_export_view", "allow_filesystem_fallback=False"),
    "tools/build_paper1_jsonl.py": ("load_paper01_export_view", "paper01_export_view_to_legacy_jsonl"),
    "scripts/metrics/aggregate.py": ("--canonical", "load_paper01_export_view", "duplicate canonical model"),
    "scripts/metrics/print_cfid_table.py": ("--canonical", "load_paper01_export_view"),
    "scripts/phase1_html.py": ("--canonical", "load_paper01_export_view"),
    "scripts/phase1_report.py": ("--canonical", "load_paper01_export_view"),
    "scripts/tuning_dashboard.py": ("--canonical-paper01", "load_paper01_export_view", "OPERATIONAL COMPLETION STATUS != ACCEPTED SCIENTIFIC EVIDENCE"),
}

REQUIRED_TEST_FILES = (
    "tests/trustforge/test_paper01_p0_consumers.py",
    "tests/trustforge/test_paper01_jsonl_producers.py",
    "tests/trustforge/test_paper01_backfill_counts.py",
    "tests/trustforge/test_paper01_aggregate_phase1.py",
    "tests/trustforge/test_paper01_direct_reporters.py",
    "tests/trustforge/test_paper01_metrics_aggregate.py",
    "tests/trustforge/test_paper01_tuning_dashboard.py",
)

class ClosureAuditError(RuntimeError):
    pass

@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

def _git(repo_root: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise ClosureAuditError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()

def _load_policy(repo_root: Path) -> dict:
    path = repo_root / POLICY_REL
    if not path.is_file():
        raise ClosureAuditError(f"missing policy: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def _paths_for_priority(policy: dict, priority: str) -> set[str]:
    return {str(r["path"]) for r in policy["records"] if r.get("priority") == priority}

def _check_policy_inventory(policy: dict) -> list[CheckResult]:
    out = []
    for priority, expected in (("P0", EXPECTED_P0), ("P1", EXPECTED_P1), ("P2", EXPECTED_P2)):
        observed = _paths_for_priority(policy, priority)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        out.append(CheckResult(
            f"policy_{priority.lower()}_inventory",
            observed == expected,
            f"{priority}: observed={len(observed)} expected={len(expected)} missing={missing} extra={extra}",
        ))
    return out

def _check_source_markers(repo_root: Path) -> list[CheckResult]:
    out = []
    for rel, markers in sorted(SOURCE_MARKERS.items()):
        path = repo_root / rel
        if not path.is_file():
            out.append(CheckResult(f"source_markers:{rel}", False, "file missing"))
            continue
        text = path.read_text(encoding="utf-8")
        missing = [m for m in markers if m not in text]
        out.append(CheckResult(
            f"source_markers:{rel}",
            not missing,
            "all canonical markers present" if not missing else f"missing markers={missing}",
        ))
    return out

def _check_tests_exist(repo_root: Path) -> list[CheckResult]:
    return [
        CheckResult(f"regression_test:{rel}", (repo_root / rel).is_file(), "present" if (repo_root / rel).is_file() else "missing")
        for rel in REQUIRED_TEST_FILES
    ]

def _changed_since_baseline(repo_root: Path) -> set[str]:
    out = _git(repo_root, "diff", "--name-only", f"{MIGRATION_BASELINE_COMMIT}..HEAD")
    return {line.strip() for line in out.splitlines() if line.strip()}

def _check_protected_paths(repo_root: Path) -> list[CheckResult]:
    changed = _changed_since_baseline(repo_root)
    return [
        CheckResult(
            f"protected_path:{rel}",
            rel not in changed,
            f"unchanged since {MIGRATION_BASELINE_COMMIT}" if rel not in changed else f"CHANGED since {MIGRATION_BASELINE_COMMIT}",
        )
        for rel in PROTECTED_PATHS
    ]

def _check_repository_identity(repo_root: Path) -> list[CheckResult]:
    branch = _git(repo_root, "branch", "--show-current")
    head = _git(repo_root, "rev-parse", "--short", "HEAD")
    branch_display = branch if branch else "<detached>"
    return [
        CheckResult(
            "repository_branch",
            True,
            (
                f"branch={branch_display}; informational only; "
                "historical migration branch identity is not required "
                "for Paper 1 closure"
            ),
        ),
        CheckResult("repository_head_resolved", bool(head), f"HEAD={head}"),
    ]

def run_audit(repo_root: Path) -> dict:
    policy = _load_policy(repo_root)
    checks = []
    checks.extend(_check_repository_identity(repo_root))
    checks.extend(_check_policy_inventory(policy))
    checks.extend(_check_source_markers(repo_root))
    checks.extend(_check_tests_exist(repo_root))
    checks.extend(_check_protected_paths(repo_root))
    passed = all(c.passed for c in checks)
    p2 = sorted(
        [r for r in policy["records"] if r.get("priority") == "P2"],
        key=lambda r: r["path"],
    )
    return {
        "audit": "M6.5.5",
        "status": "PASS" if passed else "FAIL",
        "migration_baseline_commit": MIGRATION_BASELINE_COMMIT,
        "checks": [asdict(c) for c in checks],
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(c.passed for c in checks),
            "checks_failed": sum(not c.passed for c in checks),
            "p0_count": len(EXPECTED_P0),
            "p1_count": len(EXPECTED_P1),
            "p2_deferred_count": len(EXPECTED_P2),
        },
        "deferred_p2": [
            {
                "path": r["path"],
                "treatment": r["treatment"],
                "classification": r["source_classification"],
                "rationale": r["rationale"],
            }
            for r in p2
        ],
        "closure_statement": (
            "P0 and P1 migration surfaces are accounted for; protected historical implementation paths remain unchanged since the M6.5.2 policy checkpoint; P2 work remains explicitly deferred."
            if passed else
            "M6.5 closure cannot be asserted until failed checks are resolved."
        ),
    }

def _render_markdown(report: dict) -> str:
    lines = [
        "# M6.5.5 — Paper 1 Migration Closure Audit",
        "",
        f"**Status:** {report['status']}",
        "",
        f"**Migration policy checkpoint:** `{report['migration_baseline_commit']}`",
        "",
        "## Summary",
        "",
        f"- Checks: {report['summary']['checks_passed']}/{report['summary']['checks_total']} passed",
        f"- P0 migrated targets accounted for: {report['summary']['p0_count']}",
        f"- P1 migrated targets accounted for: {report['summary']['p1_count']}",
        f"- P2 targets explicitly deferred: {report['summary']['p2_deferred_count']}",
        "",
        "## Closure checks",
        "",
    ]
    for c in report["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        lines.append(f"- **{mark}** `{c['name']}` — {c['detail']}")
    lines += ["", "## Deferred P2 handoff", ""]
    for r in report["deferred_p2"]:
        lines.append(f"- `{r['path']}` — **{r['treatment']}** — {r['rationale']}")
    lines += [
        "",
        "## Closure statement",
        "",
        report["closure_statement"],
        "",
        "> This audit does not require legacy aliases, globbing, mtime logic, or historical operational machinery to disappear repository-wide. Historical compatibility paths remain preserved by design. The closure claim applies to governed P0/P1 canonical migration surfaces.",
        "",
    ]
    return "\n".join(lines)

def write_reports(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "paper01_migration_closure_audit.json"
    mp = out_dir / "paper01_migration_closure_audit.md"
    jp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mp.write_text(_render_markdown(report), encoding="utf-8")
    return jp, mp

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--out-dir", type=Path, default=Path("studies/paper01_benchmark/audits/m6_5_5"))
    return p.parse_args()

def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    report = run_audit(repo_root)
    jp, mp = write_reports(report, out_dir)
    print(f"[audit] status={report['status']}")
    print(f"[audit] checks={report['summary']['checks_passed']}/{report['summary']['checks_total']}")
    print(f"[audit] deferred P2={report['summary']['p2_deferred_count']}")
    print(f"[audit] wrote {jp}")
    print(f"[audit] wrote {mp}")
    return 0 if report["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
