from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "tools/audit_paper01_cross_matrix.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = _load(
    SCRIPT_PATH,
    "trustforge_test_m74_cross_matrix",
)


def _fake_view(
    dataset: str,
    seed: int,
    *,
    status: str = "PASS",
) -> dict:
    return {
        "dataset": dataset,
        "seed": seed,
        "status": status,
        "experiments": 7,
        "jsonl": {
            "status": (
                "PASS"
                if status == "PASS"
                else "FAIL"
            ),
            "checks_passed": 37,
            "checks_total": 37,
            "rows": 7,
            "capabilities": {},
        },
        "runtime": {
            "status": (
                "PASS"
                if status == "PASS"
                else "FAIL"
            ),
            "checks_passed": 58,
            "checks_total": 58,
            "models": 7,
            "model_evidence": [],
        },
    }


def test_expected_views_are_exact_six() -> None:
    assert audit.expected_views() == (
        ("ustc_tfc2016", 42),
        ("ustc_tfc2016", 43),
        ("ustc_tfc2016", 44),
        ("cicmaldroid2020", 42),
        ("cicmaldroid2020", 43),
        ("cicmaldroid2020", 44),
    )


def test_aggregate_accepts_complete_passing_matrix() -> None:
    views = [
        _fake_view(dataset, seed)
        for dataset, seed
        in audit.expected_views()
    ]

    result = audit.aggregate_views(
        views
    )

    assert result["passed"]
    assert all(
        check.passed
        for check in result["checks"]
    )


def test_aggregate_rejects_missing_view() -> None:
    views = [
        _fake_view(dataset, seed)
        for dataset, seed
        in audit.expected_views()[:-1]
    ]

    result = audit.aggregate_views(
        views
    )

    assert not result["passed"]


def test_aggregate_rejects_failed_subaudit() -> None:
    views = [
        _fake_view(dataset, seed)
        for dataset, seed
        in audit.expected_views()
    ]
    views[0] = _fake_view(
        "ustc_tfc2016",
        42,
        status="FAIL",
    )

    result = audit.aggregate_views(
        views
    )

    assert not result["passed"]


def test_aggregate_requires_42_experiments() -> None:
    views = [
        _fake_view(dataset, seed)
        for dataset, seed
        in audit.expected_views()
    ]
    views[0]["experiments"] = 6

    result = audit.aggregate_views(
        views
    )

    assert not result["passed"]


def test_run_view_composes_existing_audits() -> None:
    def jsonl_runner(
        repo_root,
        *,
        dataset,
        seed,
    ):
        return {
            "status": "PASS",
            "summary": {
                "rows": 7,
                "checks_passed": 37,
                "checks_total": 37,
            },
            "capabilities": {},
        }

    def runtime_runner(
        repo_root,
        *,
        dataset,
        seed,
    ):
        return {
            "status": "PASS",
            "summary": {
                "models": 7,
                "checks_passed": 58,
                "checks_total": 58,
            },
            "models": [],
        }

    view = audit._run_view(
        REPO_ROOT,
        dataset="ustc_tfc2016",
        seed=42,
        jsonl_runner=jsonl_runner,
        runtime_runner=runtime_runner,
    )

    assert view["status"] == "PASS"
    assert view["experiments"] == 7


def test_reports_write_to_requested_directory(
    tmp_path: Path,
) -> None:
    views = [
        _fake_view(dataset, seed)
        for dataset, seed
        in audit.expected_views()
    ]
    aggregate = audit.aggregate_views(
        views
    )
    report = {
        "audit": "M7.4",
        "status": "PASS",
        "summary": {
            "views": 6,
            "views_passed": 6,
            "experiments": 42,
            "checks_total": 6,
            "checks_passed": 6,
            "checks_failed": 0,
            "jsonl_checks_total": 222,
            "jsonl_checks_passed": 222,
            "runtime_checks_total": 348,
            "runtime_checks_passed": 348,
        },
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "detail": check.detail,
            }
            for check in aggregate["checks"]
        ],
        "views": views,
        "authority_statement": "test",
    }

    json_path, md_path = (
        audit.write_reports(
            report,
            tmp_path,
        )
    )

    assert json_path.parent == tmp_path
    assert md_path.parent == tmp_path
    assert json_path.is_file()
    assert md_path.is_file()


def test_source_is_read_only_orchestration() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "audit_paper01_jsonl_compat.py" in source
    assert (
        "audit_paper01_filesystem_runtime.py"
        in source
    )
    assert "unlink(" not in source
    assert "rmtree(" not in source
    assert "remove(" not in source
