from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/generate_paper01_consumer_migration_policy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("m652_policy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


policy = _load_module()


def _source_record(
    path: str,
    classification: str,
    categories: list[str],
    accesses: list[str] | None = None,
) -> dict:
    return {
        "path": path,
        "classification": classification,
        "migration_policy": "source-policy",
        "categories": categories,
        "accesses": accesses or ["reference"],
        "evidence": [],
    }


def test_locked_path_adjudication_count() -> None:
    assert len(policy.PATH_TREATMENT) == 31
    assert policy.EXPECTED_PATH_TREATMENT_COUNT == 31


@pytest.mark.parametrize(
    ("classification", "path", "expected"),
    [
        (
            "canonical_consumer",
            "src/trustforge/paper01_execution_evidence.py",
            "ALREADY_CANONICAL",
        ),
        (
            "configuration_dependency",
            "configs/paper1_final_rerun.yaml",
            "CONFIGURATION_ONLY",
        ),
        (
            "governance_auditor",
            "scripts/inventory_paper01_execution_evidence.py",
            "GOVERNANCE_TOOL",
        ),
        (
            "historical_pipeline",
            "eval/runner.py",
            "PRESERVE_HISTORICAL",
        ),
    ],
)
def test_classification_level_treatments(
    classification: str,
    path: str,
    expected: str,
) -> None:
    rec = _source_record(path, classification, ["historical_paper1_root"])
    assert policy._decision_for(rec)["treatment"] == expected


def test_eval_runner_is_the_only_accepted_historical_pipeline() -> None:
    rec = _source_record(
        "other/runner.py",
        "historical_pipeline",
        ["timestamped_summary"],
    )
    with pytest.raises(ValueError, match="unexpected historical_pipeline"):
        policy._decision_for(rec)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("scripts/phase1_report.py", "MIGRATE_TO_CANONICAL_INTERFACE"),
        ("scripts/phase1_html.py", "MIGRATE_TO_CANONICAL_INTERFACE"),
        ("tools/build_phase1_scores.py", "MIGRATE_TO_CANONICAL_INTERFACE"),
        ("scripts/summaries_to_jsonl.py", "WRAP_WITH_COMPATIBILITY_LAYER"),
        ("tools/build_paper1_jsonl.py", "WRAP_WITH_COMPATIBILITY_LAYER"),
        ("scripts/backfill_kid_and_downstream.py", "PRESERVE_HISTORICAL"),
        ("tools/freeze_phase1_snapshots.py", "PRESERVE_HISTORICAL"),
        ("Makefile", "REVIEW_BEFORE_MIGRATION"),
    ],
)
def test_mutator_producer_path_decisions(path: str, expected: str) -> None:
    rec = _source_record(
        path,
        "historical_evidence_mutator_or_producer",
        ["timestamped_summary"],
        ["read", "write"],
    )
    assert policy._decision_for(rec)["treatment"] == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("scripts/build_jsonl.sh", "WRAP_WITH_COMPATIBILITY_LAYER"),
        ("scripts/metrics/aggregate.py", "MIGRATE_TO_CANONICAL_INTERFACE"),
        (
            "scripts/metrics/print_cfid_table.py",
            "MIGRATE_TO_CANONICAL_INTERFACE",
        ),
        ("scripts/tuning_dashboard.py", "MIGRATE_TO_CANONICAL_INTERFACE"),
        ("tools/check_phase1_integrity.py", "MIGRATE_TO_CANONICAL_INTERFACE"),
        ("scripts/jsonl_to_csv.py", "USE_CANONICAL_DERIVED_EXPORT"),
        ("scripts/plots/_common.py", "USE_CANONICAL_DERIVED_EXPORT"),
        (
            "scripts/plots/qual/class_triptychs.py",
            "USE_CANONICAL_DERIVED_EXPORT",
        ),
    ],
)
def test_legacy_consumer_path_decisions(path: str, expected: str) -> None:
    rec = _source_record(
        path,
        "legacy_or_direct_consumer",
        ["consolidated_jsonl"],
    )
    assert policy._decision_for(rec)["treatment"] == expected


def test_latest_alias_direct_migration_is_p0() -> None:
    rec = _source_record(
        "tools/check_phase1_integrity.py",
        "legacy_or_direct_consumer",
        ["latest_alias"],
    )
    decision = policy._decision_for(rec)
    assert decision["treatment"] == "MIGRATE_TO_CANONICAL_INTERFACE"
    assert decision["priority"] == "P0"
    assert decision["target_interface"] == policy.CANONICAL_LOADER


def test_non_latest_direct_migration_is_p1() -> None:
    rec = _source_record(
        "scripts/metrics/aggregate.py",
        "legacy_or_direct_consumer",
        ["timestamped_summary"],
    )
    decision = policy._decision_for(rec)
    assert decision["priority"] == "P1"


def test_derived_export_consumers_do_not_select_authority_directly() -> None:
    rec = _source_record(
        "scripts/plots/_common.py",
        "legacy_or_direct_consumer",
        ["consolidated_jsonl"],
    )
    decision = policy._decision_for(rec)
    assert decision["target_interface"] == policy.PROPOSED_EXPORT
    assert decision["priority"] == "P2"


def test_unknown_legacy_consumer_requires_explicit_adjudication() -> None:
    rec = _source_record(
        "scripts/new_unknown_consumer.py",
        "legacy_or_direct_consumer",
        ["timestamped_summary"],
    )
    with pytest.raises(ValueError, match="unadjudicated executable consumer"):
        policy._decision_for(rec)


def test_unknown_mutator_requires_explicit_adjudication() -> None:
    rec = _source_record(
        "scripts/new_unknown_mutator.py",
        "historical_evidence_mutator_or_producer",
        ["timestamped_summary"],
        ["write"],
    )
    with pytest.raises(ValueError, match="unadjudicated executable consumer"):
        policy._decision_for(rec)


def test_constraints_keep_historical_artifacts_protected() -> None:
    rec = _source_record(
        "scripts/backfill_ms_ssim_per_class.py",
        "historical_evidence_mutator_or_producer",
        ["manifest_evidence", "timestamped_summary"],
        ["read", "write"],
    )
    constraints = policy._decision_for(rec)["constraints"]
    assert "historical_artifacts_modified=NO" in constraints
    assert "eval_runner_modified=NO" in constraints
    assert "historical_behavior_preserved=YES" in constraints


def test_csv_writer_is_explicitly_lf_only(tmp_path: Path) -> None:
    out = tmp_path / "policy.csv"
    sample = {
        "records": [
            {
                "path": "x.py",
                "source_classification": "legacy_or_direct_consumer",
                "source_migration_policy": "candidate",
                "source_categories": ["timestamped_summary"],
                "source_accesses": ["read"],
                "treatment": "MIGRATE_TO_CANONICAL_INTERFACE",
                "priority": "P1",
                "target_interface": policy.CANONICAL_LOADER,
                "rationale": "x",
                "constraints": ["historical_artifacts_modified=NO"],
            }
        ]
    }
    policy._write_csv(out, sample)
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 2


def test_output_path_is_canonical(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    payload = {
        "records": [],
        "audit_version": policy.AUDIT_VERSION,
        "study_id": policy.STUDY_ID,
        "repository_head": "x",
        "source_audit": {"audit_version": policy.SOURCE_AUDIT_VERSION},
        "policy_principles": [],
        "record_count": 0,
        "summary": {"treatments": {}, "priorities": {}},
    }
    out = policy.write_policy(repo, payload)
    assert out == (repo / policy.OUTPUT_DIR_REL).resolve()


def test_live_repository_policy_shape() -> None:
    result = policy.build_policy(REPO_ROOT)

    assert result["audit_version"] == "M6.5.2-v1.0"
    assert result["source_audit"]["audit_version"] == "M6.5.1-v1.0"
    assert result["source_audit"]["record_count"] == 71
    assert result["record_count"] == 59

    assert result["summary"]["treatments"] == {
        "ALREADY_CANONICAL": 4,
        "CONFIGURATION_ONLY": 22,
        "GOVERNANCE_TOOL": 1,
        "MIGRATE_TO_CANONICAL_INTERFACE": 7,
        "PRESERVE_HISTORICAL": 7,
        "REVIEW_BEFORE_MIGRATION": 1,
        "USE_CANONICAL_DERIVED_EXPORT": 12,
        "WRAP_WITH_COMPATIBILITY_LAYER": 5,
    }


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    # Exercise the command only against the live repository because it requires
    # the committed M6.5.1 inventory and Git HEAD. The dry-run assertion is that
    # it does not create the M6.5.2 directory when absent.
    out_dir = REPO_ROOT / policy.OUTPUT_DIR_REL
    existed_before = out_dir.exists()

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(REPO_ROOT),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Dry run: no policy files written." in proc.stdout
    if not existed_before:
        assert not out_dir.exists()
