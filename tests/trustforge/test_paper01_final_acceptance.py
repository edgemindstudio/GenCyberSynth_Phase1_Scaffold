from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "tools/audit_paper01_final_acceptance.py"
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
    "trustforge_test_m75_final_acceptance",
)


def test_required_governed_reports_are_exact() -> None:
    assert set(audit.REPORTS) == {
        "m6_5_5",
        "m7_1",
        "m7_2a_ustc",
        "m7_2a_cic",
        "m7_2b_ustc",
        "m7_2b_cic",
        "m7_3",
        "m7_4",
    }


def test_all_governed_reports_exist() -> None:
    for relative in audit.REPORTS.values():
        assert (
            REPO_ROOT / relative
        ).is_file()


def test_all_governed_reports_are_pass() -> None:
    for key in audit.REPORTS:
        report = audit._load_report(
            REPO_ROOT,
            key,
        )
        assert report["status"] == "PASS"


def test_real_repository_is_accepted() -> None:
    report = audit.run_audit(
        REPO_ROOT
    )

    assert report["status"] == "PASS"
    assert report["acceptance"] == "ACCEPTED"
    assert report["summary"][
        "checks_failed"
    ] == 0
    assert report["summary"][
        "canonical_experiments"
    ] == 42
    assert report["summary"][
        "dataset_seed_views"
    ] == 6


def test_acceptance_preserves_makefile_decision() -> None:
    report = audit.run_audit(
        REPO_ROOT
    )

    names = {
        check["name"]: check
        for check in report["checks"]
    }

    assert names[
        "m7_3_no_makefile_change_authorized"
    ]["passed"]
    assert names[
        "m7_3_recommendation"
    ]["passed"]


def test_acceptance_requires_full_matrix() -> None:
    report = audit.run_audit(
        REPO_ROOT
    )

    names = {
        check["name"]: check
        for check in report["checks"]
    }

    for key in (
        "m7_4_views",
        "m7_4_experiments",
        "m7_4_matrix_checks",
        "m7_4_jsonl_checks",
        "m7_4_runtime_checks",
    ):
        assert names[key]["passed"]


def test_acceptance_statement_preserves_history() -> None:
    report = audit.run_audit(
        REPO_ROOT
    )

    statement = report[
        "acceptance_statement"
    ].lower()

    assert "historical execution artifacts" in statement
    assert "no makefile reinterpretation" in statement
    assert "no" in statement


def test_reports_write_to_requested_directory(
    tmp_path: Path,
) -> None:
    report = audit.run_audit(
        REPO_ROOT
    )

    json_path, md_path = (
        audit.write_reports(
            report,
            tmp_path,
        )
    )

    assert json_path.parent == tmp_path
    assert md_path.parent == tmp_path

    payload = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )
    assert payload[
        "acceptance"
    ] == "ACCEPTED"

    markdown = md_path.read_text(
        encoding="utf-8"
    )
    assert (
        "**Acceptance:** ACCEPTED"
        in markdown
    )


def test_audit_is_read_only_acceptance() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "unlink(" not in source
    assert "rmtree(" not in source
    assert "remove(" not in source
    assert "Makefile" in source
    assert (
        "does not rerun experiments"
        in source
    )
