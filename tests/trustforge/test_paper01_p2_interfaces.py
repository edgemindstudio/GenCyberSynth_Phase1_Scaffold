from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "tools/audit_paper01_p2_interfaces.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    import sys
    sys.modules[name] = module

    spec.loader.exec_module(module)
    return module


audit = _load(
    SCRIPT_PATH,
    "trustforge_test_m71_p2_interface_audit",
)


def test_expected_p2_partition() -> None:
    assert len(audit.P2_JSONL_CONSUMERS) == 12
    assert len(audit.FILESYSTEM_COUPLED) == 2
    assert audit.FILESYSTEM_COUPLED <= set(
        audit.P2_JSONL_CONSUMERS
    )


def test_policy_inventory_matches_m652() -> None:
    policy = audit._load_policy(REPO_ROOT)
    result = audit._check_policy(policy)

    assert result.passed


def test_all_jsonl_consumers_pass_static_contract_checks() -> None:
    failed = []

    for relative in audit.P2_JSONL_CONSUMERS:
        failed.extend(
            check
            for check in audit._check_consumer(
                REPO_ROOT,
                relative,
            )
            if not check.passed
        )

    assert failed == []


def test_filesystem_coupled_consumers_are_exact() -> None:
    assert audit.FILESYSTEM_COUPLED == {
        "scripts/plots/imbalance/simple_stats_sanity.py",
        "scripts/plots/qual/class_triptychs.py",
    }


def test_makefile_is_separate_review_surface() -> None:
    checks = audit._check_makefile(
        REPO_ROOT
    )

    assert len(checks) == 1
    assert checks[0].passed
    assert "M7.3" in checks[0].detail


def test_run_audit_passes_repository() -> None:
    report = audit.run_audit(
        REPO_ROOT
    )

    assert report["status"] == "PASS"
    assert report["summary"]["p2_total"] == 13
    assert report["summary"]["jsonl_consumers"] == 12
    assert (
        report["summary"][
            "jsonl_only_consumers"
        ]
        == 10
    )
    assert (
        report["summary"][
            "filesystem_coupled_consumers"
        ]
        == 2
    )


def test_consumer_matrix_marks_runtime_handoff() -> None:
    report = audit.run_audit(
        REPO_ROOT
    )

    runtime_paths = {
        row["path"]
        for row in report["consumers"]
        if row["m7_2_runtime_required"]
    }

    assert runtime_paths == audit.FILESYSTEM_COUPLED


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
    assert payload["status"] == "PASS"

    markdown = md_path.read_text(
        encoding="utf-8"
    )
    assert "Consumer matrix" in markdown
    assert "M7.3" in markdown


def test_audit_is_read_only_source_analysis() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "/home/bruno.fonkeng/gencys" not in source
    assert "unlink(" not in source
    assert "rmtree(" not in source
    assert "remove(" not in source
