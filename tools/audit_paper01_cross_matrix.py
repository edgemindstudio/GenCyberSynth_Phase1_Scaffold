#!/usr/bin/env python3
"""
M7.4 — Paper 1 cross-dataset / cross-seed regression matrix.

Read-only orchestration audit.

This audit does not reimplement M7.2A or M7.2B. It composes their governed
checks across all six Paper 1 dataset/seed views:

    ustc_tfc2016      × seeds 42,43,44
    cicmaldroid2020   × seeds 42,43,44

Each view must pass:
- M7.2A canonical-derived JSONL structural compatibility;
- M7.2B canonical manifest/filesystem runtime compatibility.

Six views × seven families = 42 canonical Paper 1 experiments.

No experiment is rerun. No historical artifact, manifest, image, summary, or
result table is modified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable


DATASETS = (
    "ustc_tfc2016",
    "cicmaldroid2020",
)
SEEDS = (42, 43, 44)
FAMILIES_PER_VIEW = 7

JSONL_AUDIT_REL = Path(
    "tools/audit_paper01_jsonl_compat.py"
)
RUNTIME_AUDIT_REL = Path(
    "tools/audit_paper01_filesystem_runtime.py"
)


class CrossMatrixAuditError(RuntimeError):
    """Raised when M7.4 cannot construct the governed regression matrix."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _load_module(
    path: Path,
    name: str,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    if spec is None or spec.loader is None:
        raise CrossMatrixAuditError(
            f"cannot import audit module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_subaudits(
    repo_root: Path,
):
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    jsonl = _load_module(
        repo_root / JSONL_AUDIT_REL,
        "trustforge_m74_jsonl_compat",
    )
    runtime = _load_module(
        repo_root / RUNTIME_AUDIT_REL,
        "trustforge_m74_filesystem_runtime",
    )
    return jsonl, runtime


def expected_views() -> tuple[tuple[str, int], ...]:
    return tuple(
        (dataset, seed)
        for dataset in DATASETS
        for seed in SEEDS
    )


def _run_view(
    repo_root: Path,
    *,
    dataset: str,
    seed: int,
    jsonl_runner: Callable[..., dict[str, Any]],
    runtime_runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    jsonl_report = jsonl_runner(
        repo_root,
        dataset=dataset,
        seed=seed,
    )
    runtime_report = runtime_runner(
        repo_root,
        dataset=dataset,
        seed=seed,
    )

    jsonl_pass = (
        jsonl_report.get("status") == "PASS"
    )
    runtime_pass = (
        runtime_report.get("status") == "PASS"
    )
    rows = int(
        jsonl_report.get(
            "summary",
            {},
        ).get("rows", 0)
    )
    models = int(
        runtime_report.get(
            "summary",
            {},
        ).get("models", 0)
    )

    return {
        "dataset": dataset,
        "seed": seed,
        "status": (
            "PASS"
            if (
                jsonl_pass
                and runtime_pass
                and rows == FAMILIES_PER_VIEW
                and models == FAMILIES_PER_VIEW
            )
            else "FAIL"
        ),
        "experiments": rows,
        "jsonl": {
            "status": jsonl_report.get(
                "status"
            ),
            "checks_passed": (
                jsonl_report.get(
                    "summary",
                    {},
                ).get(
                    "checks_passed",
                    0,
                )
            ),
            "checks_total": (
                jsonl_report.get(
                    "summary",
                    {},
                ).get(
                    "checks_total",
                    0,
                )
            ),
            "rows": rows,
            "capabilities": (
                jsonl_report.get(
                    "capabilities",
                    {},
                )
            ),
        },
        "runtime": {
            "status": runtime_report.get(
                "status"
            ),
            "checks_passed": (
                runtime_report.get(
                    "summary",
                    {},
                ).get(
                    "checks_passed",
                    0,
                )
            ),
            "checks_total": (
                runtime_report.get(
                    "summary",
                    {},
                ).get(
                    "checks_total",
                    0,
                )
            ),
            "models": models,
            "model_evidence": (
                runtime_report.get(
                    "models",
                    [],
                )
            ),
        },
    }


def aggregate_views(
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = set(expected_views())
    observed = {
        (
            str(view.get("dataset")),
            int(view.get("seed")),
        )
        for view in views
    }

    checks = [
        CheckResult(
            name="exact_six_views",
            passed=(
                len(views) == 6
                and observed == expected
            ),
            detail=(
                f"views={len(views)} "
                f"missing={sorted(expected - observed)} "
                f"extra={sorted(observed - expected)}"
            ),
        ),
        CheckResult(
            name="all_views_pass",
            passed=all(
                view.get("status") == "PASS"
                for view in views
            ),
            detail=(
                f"passed={sum(view.get('status') == 'PASS' for view in views)}/"
                f"{len(views)}"
            ),
        ),
        CheckResult(
            name="exact_42_experiments",
            passed=sum(
                int(view.get("experiments", 0))
                for view in views
            ) == 42,
            detail=(
                "experiments="
                f"{sum(int(view.get('experiments', 0)) for view in views)}"
            ),
        ),
        CheckResult(
            name="all_jsonl_subaudits_pass",
            passed=all(
                view.get(
                    "jsonl",
                    {},
                ).get("status")
                == "PASS"
                for view in views
            ),
            detail=(
                f"passed={sum(view.get('jsonl', {}).get('status') == 'PASS' for view in views)}/"
                f"{len(views)}"
            ),
        ),
        CheckResult(
            name="all_runtime_subaudits_pass",
            passed=all(
                view.get(
                    "runtime",
                    {},
                ).get("status")
                == "PASS"
                for view in views
            ),
            detail=(
                f"passed={sum(view.get('runtime', {}).get('status') == 'PASS' for view in views)}/"
                f"{len(views)}"
            ),
        ),
        CheckResult(
            name="seven_rows_and_models_per_view",
            passed=all(
                (
                    view.get(
                        "jsonl",
                        {},
                    ).get("rows")
                    == FAMILIES_PER_VIEW
                    and view.get(
                        "runtime",
                        {},
                    ).get("models")
                    == FAMILIES_PER_VIEW
                )
                for view in views
            ),
            detail="expected 7 JSONL rows and 7 runtime models per view",
        ),
    ]

    return {
        "checks": checks,
        "passed": all(
            check.passed
            for check in checks
        ),
    }


def run_audit(
    repo_root: Path,
) -> dict[str, Any]:
    jsonl_module, runtime_module = (
        _load_subaudits(repo_root)
    )

    views: list[dict[str, Any]] = []
    for dataset, seed in expected_views():
        views.append(
            _run_view(
                repo_root,
                dataset=dataset,
                seed=seed,
                jsonl_runner=(
                    jsonl_module.run_audit
                ),
                runtime_runner=(
                    runtime_module.run_audit
                ),
            )
        )

    aggregate = aggregate_views(views)
    checks = aggregate["checks"]

    return {
        "audit": "M7.4",
        "status": (
            "PASS"
            if aggregate["passed"]
            else "FAIL"
        ),
        "summary": {
            "views": len(views),
            "views_passed": sum(
                view["status"] == "PASS"
                for view in views
            ),
            "experiments": sum(
                int(view["experiments"])
                for view in views
            ),
            "checks_total": len(checks),
            "checks_passed": sum(
                check.passed
                for check in checks
            ),
            "checks_failed": sum(
                not check.passed
                for check in checks
            ),
            "jsonl_checks_total": sum(
                int(
                    view["jsonl"][
                        "checks_total"
                    ]
                )
                for view in views
            ),
            "jsonl_checks_passed": sum(
                int(
                    view["jsonl"][
                        "checks_passed"
                    ]
                )
                for view in views
            ),
            "runtime_checks_total": sum(
                int(
                    view["runtime"][
                        "checks_total"
                    ]
                )
                for view in views
            ),
            "runtime_checks_passed": sum(
                int(
                    view["runtime"][
                        "checks_passed"
                    ]
                )
                for view in views
            ),
        },
        "checks": [
            asdict(check)
            for check in checks
        ],
        "views": views,
        "authority_statement": (
            "All 42 Paper 1 experiments are exercised through explicit "
            "dataset+seed canonical export authority and canonical-derived "
            "compatibility/runtime interfaces. Historical experiments are "
            "not rerun and filesystem fallback is not scientific authority."
        ),
    }


def _capability_summary(
    view: dict[str, Any],
) -> str:
    capabilities = (
        view.get("jsonl", {})
        .get("capabilities", {})
    )
    names = (
        "counts",
        "generative_similarity",
        "downstream_utility",
        "manifest_path",
    )
    values = []
    for name in names:
        info = capabilities.get(name, {})
        values.append(
            f"{name}="
            f"{info.get('available_rows', 0)}/7"
        )
    return "; ".join(values)


def _render_markdown(
    report: dict[str, Any],
) -> str:
    summary = report["summary"]

    lines = [
        "# M7.4 — Paper 1 Cross-dataset / Cross-seed Regression Matrix",
        "",
        f"**Status:** {report['status']}",
        "",
        "## Summary",
        "",
        (
            f"- Views: "
            f"{summary['views_passed']}/"
            f"{summary['views']} passed"
        ),
        (
            f"- Canonical experiments: "
            f"{summary['experiments']}/42"
        ),
        (
            f"- Matrix checks: "
            f"{summary['checks_passed']}/"
            f"{summary['checks_total']} passed"
        ),
        (
            f"- JSONL structural checks: "
            f"{summary['jsonl_checks_passed']}/"
            f"{summary['jsonl_checks_total']} passed"
        ),
        (
            f"- Runtime filesystem checks: "
            f"{summary['runtime_checks_passed']}/"
            f"{summary['runtime_checks_total']} passed"
        ),
        "",
        "## Matrix",
        "",
        (
            "| Dataset | Seed | Experiments | JSONL | Runtime | "
            "Core capabilities |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]

    for view in report["views"]:
        lines.append(
            f"| `{view['dataset']}` | "
            f"{view['seed']} | "
            f"{view['experiments']} | "
            f"{view['jsonl']['status']} | "
            f"{view['runtime']['status']} | "
            f"{_capability_summary(view)} |"
        )

    lines.extend(
        [
            "",
            "## Authority statement",
            "",
            report["authority_statement"],
            "",
            (
                "> M7.4 validates the already-adjudicated evidence chain. "
                "It does not rerun generators/evaluators, rewrite summaries, "
                "or promote filesystem discovery into scientific authority."
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
        / "paper01_cross_dataset_seed_matrix.json"
    )
    md_path = (
        out_dir
        / "paper01_cross_dataset_seed_matrix.md"
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
            "audits/m7_4"
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
        "[audit] views="
        f"{report['summary']['views_passed']}/"
        f"{report['summary']['views']} "
        "experiments="
        f"{report['summary']['experiments']}/42"
    )
    print(
        "[audit] jsonl checks="
        f"{report['summary']['jsonl_checks_passed']}/"
        f"{report['summary']['jsonl_checks_total']} "
        "runtime checks="
        f"{report['summary']['runtime_checks_passed']}/"
        f"{report['summary']['runtime_checks_total']}"
    )

    for view in report["views"]:
        print(
            "[audit] "
            f"{view['dataset']} seed={view['seed']}: "
            f"status={view['status']} "
            f"jsonl={view['jsonl']['status']} "
            f"runtime={view['runtime']['status']} "
            f"experiments={view['experiments']}"
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
