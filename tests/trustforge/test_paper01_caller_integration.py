from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"
WRAPPER = REPO_ROOT / "tools/phase1_gate.sh"


def _makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _wrapper() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def _target_block(source: str, target: str, next_marker: str) -> str:
    pattern = re.compile(
        rf"(?ms)^{re.escape(target)}:(.*?)(?=^{re.escape(next_marker)}:)"
    )
    match = pattern.search(source)
    assert match is not None, f"missing target block: {target}"
    return match.group(0)


def test_makefile_declares_explicit_canonical_identity_variables() -> None:
    source = _makefile()

    assert "PHASE1_DATASET" in source
    assert "PHASE1_SEED" in source
    assert "TRUSTFORGE_ARTIFACTS_ROOT" in source


def test_phase1_freeze_remains_historical_snapshot_operation() -> None:
    source = _makefile()
    block = _target_block(source, "phase1_freeze", "phase1_scores")

    assert "tools/freeze_phase1_snapshots.py" in block
    assert "PHASE1_SUMMARY_NAME" in block
    assert "PHASE1_MANIFEST_NAME" in block
    assert "PHASE1_ALLOWED_BUDGETS" in block
    assert "PHASE1_DATASET" not in block
    assert "PHASE1_SEED" not in block


def test_phase1_scores_passes_canonical_identity_and_artifact_root() -> None:
    source = _makefile()
    block = _target_block(source, "phase1_scores", "phase1_check")

    assert "tools/build_phase1_scores.py" in block
    assert "PHASE1_DATASET" in block
    assert "PHASE1_SEED" in block
    assert "TRUSTFORGE_ARTIFACTS_ROOT" in block
    assert "PHASE1_SUMMARY_NAME" not in block


def test_phase1_check_passes_canonical_identity_only() -> None:
    source = _makefile()
    block = _target_block(source, "phase1_check", "phase1_backfill")

    assert "tools/check_phase1_integrity.py" in block
    assert "PHASE1_DATASET" in block
    assert "PHASE1_SEED" in block
    assert "PHASE1_SUMMARY_NAME" not in block
    assert "PHASE1_ALLOWED_BUDGETS" not in block


def test_phase1_backfill_remains_historical_operation() -> None:
    source = _makefile()
    block = _target_block(source, "phase1_backfill", "phase1_gate")

    assert "scripts/backfill_kid_and_downstream.py" in block
    assert "PHASE1_SUMMARY_NAME" in block
    assert "PHASE1_MANIFEST_NAME" in block


def test_phase1_gate_no_longer_invokes_freeze() -> None:
    source = _makefile()
    block = _target_block(source, "phase1_gate", "normalize-summaries")

    assert "phase1_freeze" not in block
    assert "tools/check_phase1_integrity.py" in block
    assert "tools/build_phase1_scores.py" in block


def test_phase1_gate_runs_integrity_before_score_export() -> None:
    source = _makefile()
    block = _target_block(source, "phase1_gate", "normalize-summaries")

    check_index = block.index("tools/check_phase1_integrity.py")
    build_index = block.index("tools/build_phase1_scores.py")

    assert check_index < build_index


def test_canonical_make_targets_use_src_pythonpath() -> None:
    source = _makefile()

    score_block = _target_block(source, "phase1_scores", "phase1_check")
    check_block = _target_block(source, "phase1_check", "phase1_backfill")
    gate_block = _target_block(source, "phase1_gate", "normalize-summaries")

    required = 'PYTHONPATH="$(CURDIR)/src"'

    assert required in score_block
    assert required in check_block
    assert gate_block.count(required) == 2


def test_paper1_prepare_remains_historical_snapshot_preparation() -> None:
    source = _makefile()
    block = _target_block(source, "paper1_prepare", "paper1_build")

    assert "phase1_freeze" in block
    assert "phase1_gate" not in block


def test_wrapper_requires_explicit_scientific_identity() -> None:
    source = _wrapper()

    assert '${PHASE1_DATASET:?' in source
    assert '${PHASE1_SEED:?' in source
    assert '${TRUSTFORGE_ARTIFACTS_ROOT:?' in source


def test_wrapper_runs_integrity_before_score_export() -> None:
    source = _wrapper()

    check_index = source.index("tools/check_phase1_integrity.py")
    build_index = source.index("tools/build_phase1_scores.py")

    assert check_index < build_index


def test_wrapper_does_not_invoke_historical_freeze() -> None:
    source = _wrapper()

    assert "freeze_phase1_snapshots.py" not in source
    assert "PHASE1_SUMMARY_NAME" not in source
    assert "paper1.json" not in source
    assert "latest.json" not in source
