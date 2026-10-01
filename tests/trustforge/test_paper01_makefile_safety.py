from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "tools/audit_paper01_makefile_safety.py"
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
    "trustforge_test_m73_makefile_audit",
)


def test_classification_partition() -> None:
    result = audit._check_partition()

    assert result.passed
    assert len(
        audit.TARGET_CLASSIFICATION
    ) == 21
    assert len(
        audit.CANONICAL_TARGETS
    ) == 3
    assert len(
        audit.HISTORICAL_TARGETS
    ) == 4
    assert len(
        audit.MIXED_TARGETS
    ) == 12
    assert len(
        audit.DESTRUCTIVE_TARGETS
    ) == 2


def test_makefile_policy_is_review_before_migration() -> None:
    policy = audit._load_policy(
        REPO_ROOT
    )
    checks = audit._check_policy(
        policy
    )

    assert all(
        check.passed
        for check in checks
    )


def test_makefile_target_blocks_are_detected() -> None:
    source = (
        REPO_ROOT / "Makefile"
    ).read_text(encoding="utf-8")
    blocks = audit._target_blocks(
        source
    )

    for target in (
        audit.CANONICAL_TARGETS
        | audit.HISTORICAL_TARGETS
        | audit.MIXED_TARGETS
        | audit.DESTRUCTIVE_TARGETS
    ):
        assert target in blocks


def test_canonical_targets_have_required_identity_markers() -> None:
    source = (
        REPO_ROOT / "Makefile"
    ).read_text(encoding="utf-8")
    blocks = audit._target_blocks(
        source
    )

    checks = audit._check_markers(
        blocks,
        audit.CANONICAL_MARKERS,
        "canonical",
    )

    assert all(
        check.passed
        for check in checks
    )


def test_historical_targets_remain_explicit() -> None:
    source = (
        REPO_ROOT / "Makefile"
    ).read_text(encoding="utf-8")
    blocks = audit._target_blocks(
        source
    )

    checks = audit._check_markers(
        blocks,
        audit.HISTORICAL_MARKERS,
        "historical",
    )

    assert all(
        check.passed
        for check in checks
    )


def test_mixed_targets_are_explicitly_detected() -> None:
    source = (
        REPO_ROOT / "Makefile"
    ).read_text(encoding="utf-8")
    blocks = audit._target_blocks(
        source
    )

    checks = audit._check_markers(
        blocks,
        audit.MIXED_MARKERS,
        "mixed",
    )

    assert all(
        check.passed
        for check in checks
    )


def test_destructive_targets_are_detected() -> None:
    source = (
        REPO_ROOT / "Makefile"
    ).read_text(encoding="utf-8")
    blocks = audit._target_blocks(
        source
    )

    checks = audit._check_markers(
        blocks,
        audit.DESTRUCTIVE_MARKERS,
        "destructive",
    )

    assert all(
        check.passed
        for check in checks
    )


def test_paper1_jsonl_does_not_force_canonical_identity() -> None:
    source = (
        REPO_ROOT / "Makefile"
    ).read_text(encoding="utf-8")
    blocks = audit._target_blocks(
        source
    )

    checks = (
        audit._check_mixed_identity_gap(
            blocks
        )
    )

    assert len(checks) == 1
    assert checks[0].passed


def test_real_repository_audit_passes_without_authorizing_change() -> None:
    report = audit.run_audit(
        REPO_ROOT
    )

    assert report["status"] == "PASS"
    assert report["summary"][
        "targets_reviewed"
    ] == 21
    assert report["decision"][
        "makefile_modification_authorized"
    ] is False
    assert report["decision"][
        "recommendation"
    ] == "NO_CHANGE_DURING_M7_3_REVIEW"


def test_reports_write_only_to_requested_directory(
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
    assert (
        "Makefile modification authorized"
        in markdown
    )


def test_audit_does_not_modify_makefile() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "write_text(" in source  # reports only
    assert 'repo_root / "Makefile"' in source
    assert ".read_text(" in source
    assert 'makefile.write_text(' not in source
    assert "unlink(" not in source
    assert "rmtree(" not in source
