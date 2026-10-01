from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "tools/audit_paper01_jsonl_compat.py"
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
    "trustforge_test_m72a_jsonl_compat",
)


def _rows():
    rows = []

    for family in sorted(
        audit.PAPER01_FAMILIES
    ):
        rows.append(
            {
                "model": family,
                "seed": 42,
                "budget_per_class": 2000,
                "run_id": f"{family}_s42",
                "timestamp": (
                    "2026-04-04T16:11:24+00:00"
                ),
                "source_path": (
                    f"/history/{family}/"
                    "summary_20260404_161124.json"
                ),
                "num_real": 100,
                "num_fake": 18000,
                "kid": 0.1,
                "ms_ssim": 0.5,
                "macro_f1": 0.8,
                "manifest_path": (
                    f"/history/{family}/manifest.json"
                ),
            }
        )

    return rows


def test_consumer_inventory_is_twelve() -> None:
    assert len(audit.P2_CONSUMERS) == 12
    assert set(
        audit.CONSUMER_CAPABILITIES
    ) == set(audit.P2_CONSUMERS)


def test_validate_rows_accepts_seven_canonical_rows() -> None:
    checks = audit.validate_rows(
        _rows(),
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert all(
        check.passed
        for check in checks
    )


def test_validate_rows_rejects_alias_source() -> None:
    rows = _rows()
    rows[0]["source_path"] = (
        "/history/latest.json"
    )

    checks = audit.validate_rows(
        rows,
        dataset="ustc_tfc2016",
        seed=42,
    )

    failed = [
        check
        for check in checks
        if not check.passed
    ]

    assert any(
        check.name.startswith(
            "canonical_source:"
        )
        for check in failed
    )


def test_validate_rows_rejects_seed_mismatch() -> None:
    rows = _rows()
    rows[0]["seed"] = 43

    checks = audit.validate_rows(
        rows,
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert any(
        (
            check.name.startswith(
                "seed_identity:"
            )
            and not check.passed
        )
        for check in checks
    )


def test_capability_report_is_non_authoritative() -> None:
    rows = _rows()
    for row in rows:
        row.pop("manifest_path")

    report = audit.capability_report(
        rows
    )

    assert (
        report["manifest_path"][
            "available_rows"
        ]
        == 0
    )
    assert report[
        "generative_similarity"
    ]["all_rows"]


def test_consumer_report_keeps_structure_pass_when_optional_missing() -> None:
    rows = _rows()
    for row in rows:
        row.pop("manifest_path")

    capabilities = (
        audit.capability_report(rows)
    )
    consumers = audit.consumer_report(
        capabilities
    )

    assert all(
        row["structural_contract"]
        == "PASS"
        for row in consumers
    )


def test_real_ustc_seed42_audit_passes() -> None:
    report = audit.run_audit(
        REPO_ROOT,
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert report["status"] == "PASS"
    assert report["summary"]["rows"] == 7
    assert report["summary"][
        "p2_consumers"
    ] == 12
    assert report["summary"][
        "checks_failed"
    ] == 0


def test_real_cic_seed42_audit_passes() -> None:
    report = audit.run_audit(
        REPO_ROOT,
        dataset="cicmaldroid2020",
        seed=42,
    )

    assert report["status"] == "PASS"
    assert report["summary"]["rows"] == 7
    assert report["summary"][
        "checks_failed"
    ] == 0


def test_reports_write_to_requested_directory(
    tmp_path: Path,
) -> None:
    report = {
        "audit": "M7.2A",
        "dataset": "ustc_tfc2016",
        "seed": 42,
        "status": "PASS",
        "summary": {
            "checks_total": 1,
            "checks_passed": 1,
            "checks_failed": 0,
            "rows": 7,
            "p2_consumers": 12,
        },
        "checks": [],
        "capabilities": {
            "counts": {
                "available_rows": 7,
                "total_rows": 7,
                "all_rows": True,
                "any_rows": True,
                "per_family": {},
            }
        },
        "consumers": [],
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


def test_source_never_uses_latest_as_authority() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "load_paper01_export_view" in source
    assert (
        "paper01_export_view_to_legacy_jsonl"
        in source
    )
    assert "Optional metrics" in source
