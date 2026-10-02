from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "tools/audit_paper01_migration_closure.py"

def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

audit = _load(SCRIPT_PATH, "trustforge_test_m655_closure_audit")

def test_expected_priority_counts():
    assert len(audit.EXPECTED_P0) == 2
    assert len(audit.EXPECTED_P1) == 10
    assert len(audit.EXPECTED_P2) == 13

def test_policy_inventory_matches_recorded_m652_policy():
    checks = audit._check_policy_inventory(audit._load_policy(REPO_ROOT))
    assert len(checks) == 3
    assert all(c.passed for c in checks)

def test_all_migrated_p0_p1_surfaces_have_markers():
    checks = audit._check_source_markers(REPO_ROOT)
    assert len(checks) == 12
    assert [c for c in checks if not c.passed] == []

def test_required_regression_tests_exist():
    checks = audit._check_tests_exist(REPO_ROOT)
    assert len(checks) == 7
    assert all(c.passed for c in checks)

def test_protected_paths_unchanged_since_policy_checkpoint():
    checks = audit._check_protected_paths(REPO_ROOT)
    assert len(checks) == 2
    assert [c for c in checks if not c.passed] == []

def test_repository_identity_does_not_require_historical_branch(monkeypatch):
    def fake_git(repo_root: Path, *args: str) -> str:
        del repo_root

        if args == ("branch", "--show-current"):
            return "main"

        if args == ("rev-parse", "--short", "HEAD"):
            return "deadbee"

        raise AssertionError(f"unexpected git invocation: {args}")

    monkeypatch.setattr(audit, "_git", fake_git)

    checks = audit._check_repository_identity(REPO_ROOT)

    assert len(checks) == 2
    assert checks[0].name == "repository_branch"
    assert checks[0].passed
    assert "branch=main" in checks[0].detail
    assert "informational only" in checks[0].detail
    assert checks[1].name == "repository_head_resolved"
    assert checks[1].passed
    assert checks[1].detail == "HEAD=deadbee"


def test_run_audit_passes_on_repository():
    report = audit.run_audit(REPO_ROOT)
    assert report["status"] == "PASS"
    assert report["summary"]["p0_count"] == 2
    assert report["summary"]["p1_count"] == 10
    assert report["summary"]["p2_deferred_count"] == 13
    assert report["summary"]["checks_failed"] == 0

def test_deferred_p2_inventory_is_explicit():
    report = audit.run_audit(REPO_ROOT)
    assert {r["path"] for r in report["deferred_p2"]} == audit.EXPECTED_P2

def test_reports_write_only_to_requested_directory(tmp_path):
    report = {
        "audit": "M6.5.5",
        "status": "PASS",
        "migration_baseline_commit": "a450c37",
        "checks": [],
        "summary": {
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "p0_count": 2,
            "p1_count": 10,
            "p2_deferred_count": 13,
        },
        "deferred_p2": [],
        "closure_statement": "test",
    }
    jp, mp = audit.write_reports(report, tmp_path)
    assert jp.parent == tmp_path and mp.parent == tmp_path
    assert json.loads(jp.read_text(encoding="utf-8"))["status"] == "PASS"
    assert "Closure statement" in mp.read_text(encoding="utf-8")

def test_audit_does_not_reference_historical_data_roots():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "/home/bruno.fonkeng/gencys" not in source
    assert "TRUSTFORGE_DATA_ROOT" not in source
    assert "TRUSTFORGE_ARTIFACTS_ROOT" not in source

def test_closure_scope_explicitly_preserves_legacy_paths():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "does not require legacy aliases" in source
    assert "Historical compatibility paths remain preserved by design" in source
