"""Tests for M6.5.1 Paper 1 downstream consumer inventory."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "inventory_paper01_downstream_consumers.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "inventory_paper01_downstream_consumers",
        SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load M6.5.1 inventory module."
        )

    module = importlib.util.module_from_spec(spec)

    # Python 3.10 dataclasses resolve annotation metadata through
    # sys.modules[cls.__module__] while the module body is executing.
    # Register the dynamically loaded module before exec_module().
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)
    return module


inventory = _load_module()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    subprocess.run(
        ["git", "init", "-q"],
        cwd=repo,
        check=True,
    )

    return repo


def _track(repo: Path, relative: str, text: str) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    subprocess.run(
        ["git", "add", relative],
        cwd=repo,
        check=True,
    )

    return path


def test_scan_detects_canonical_interface(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    path = _track(
        repo,
        "scripts/consumer.py",
        (
            "from trustforge.paper01_execution_evidence "
            "import load_paper01_linkage_study\n"
            "study = load_paper01_linkage_study('.')\n"
        ),
    )

    record = inventory.scan_file(repo, path)

    assert record is not None
    assert "canonical_linkage_interface" in record.categories
    assert record.classification == "canonical_consumer"
    assert record.migration_policy == "already_canonical"


def test_scan_detects_latest_alias(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    path = _track(
        repo,
        "scripts/report.py",
        (
            "p = Path('artifacts/gan/summaries/latest.json')\n"
            "record = json.load(open(p))\n"
        ),
    )

    record = inventory.scan_file(repo, path)

    assert record is not None
    assert "latest_alias" in record.categories
    assert record.classification == "legacy_or_direct_consumer"


def test_eval_runner_is_protected_historical_pipeline(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    path = _track(
        repo,
        "eval/runner.py",
        (
            "latest = summaries / 'latest.json'\n"
            "latest.write_text('{}')\n"
        ),
    )

    record = inventory.scan_file(repo, path)

    assert record is not None
    assert record.classification == "historical_pipeline"
    assert record.migration_policy == "protected_do_not_migrate_in_m6_5_1"


def test_summary_writer_requires_review_before_migration(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    path = _track(
        repo,
        "scripts/backfill.py",
        (
            "p = Path('artifacts/x/summaries/summary_*.json')\n"
            "p.write_text('{}')\n"
        ),
    )

    record = inventory.scan_file(repo, path)

    assert record is not None
    assert record.classification == "historical_evidence_mutator_or_producer"
    assert record.migration_policy == "review_before_any_migration"


def test_documentation_is_review_only(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    path = _track(
        repo,
        "README.md",
        "Use artifacts/summaries/phase1_summaries.jsonl\n",
    )

    record = inventory.scan_file(repo, path)

    assert record is not None
    assert record.classification == "documentation"
    assert record.migration_policy == "review_only"


def test_generic_manifest_only_file_is_not_paper1_consumer(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    path = _track(
        repo,
        "scripts/generic_manifest_tool.py",
        "manifest = Path('manifest.json')\n",
    )

    assert inventory.scan_file(repo, path) is None


def test_paper1_manifest_only_path_is_still_inventoryable(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    path = _track(
        repo,
        "scripts/paper01_manifest_review.py",
        "manifest = Path('manifest.json')\n",
    )

    record = inventory.scan_file(repo, path)

    assert record is not None
    assert "manifest_evidence" in record.categories


def test_canonical_evidence_data_products_are_excluded(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)

    excluded = _track(
        repo,
        "studies/paper01_benchmark/execution_evidence/example.yaml",
        "accepted_summary: summary_20260410_091359.json\n",
    )

    assert excluded not in inventory.tracked_files(repo)


def test_prior_paper1_audits_are_excluded(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)

    excluded = _track(
        repo,
        "studies/paper01_benchmark/audits/m6_4_1/inventory.md",
        "phase1_scores_dedup.csv\n",
    )

    assert excluded not in inventory.tracked_files(repo)


def test_paper_result_data_is_excluded(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    result = _track(
        repo,
        "papers/paper3_when_does_synth_help/results/raw/results.csv",
        "summary_file,metric\nsummary_20260501.json,0.5\n",
    )

    assert result not in inventory.tracked_files(repo)


def test_schema_examples_are_excluded(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    example = _track(
        repo,
        "schemas/examples/paper01/example.yaml",
        "result_table: phase1_scores_dedup.csv\n",
    )

    assert example not in inventory.tracked_files(repo)


def test_cross_paper_generic_summary_is_not_paper1_consumer(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)

    path = _track(
        repo,
        "papers/paper3_when_does_synth_help/scripts/collector.py",
        "files = glob('results/summary_*.json')\n",
    )

    assert inventory.scan_file(repo, path) is None


def test_cross_paper_explicit_paper1_root_is_retained(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)

    path = _track(
        repo,
        "papers/paper2_conditional_generation_done_right/scripts/baseline.py",
        "root = '~/gencys/artifacts_paper1_cicmaldroid'\n",
    )

    record = inventory.scan_file(repo, path)

    assert record is not None
    assert "historical_paper1_root" in record.categories


def test_yaml_dependency_is_not_code_migration_candidate(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)

    path = _track(
        repo,
        "configs/paper1_example.yaml",
        "artifacts: ~/gencys/artifacts_paper1_final\n",
    )

    record = inventory.scan_file(repo, path)

    assert record is not None
    assert record.classification == "configuration_dependency"
    assert (
        record.migration_policy
        == "review_dependency_no_automatic_migration"
    )


def test_execution_inventory_is_governance_auditor(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)

    path = _track(
        repo,
        "scripts/inventory_paper01_execution_evidence.py",
        "table = 'phase1_scores_dedup.csv'\nopen('audit.json', 'w').write('{}')\n",
    )

    record = inventory.scan_file(repo, path)

    assert record is not None
    assert record.classification == "governance_auditor"
    assert record.migration_policy == "preserve_governance_tool"


def test_untracked_file_is_not_in_inventory(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    untracked = repo / "scripts" / "untracked.py"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("x = 'latest.json'\n", encoding="utf-8")

    assert untracked not in inventory.tracked_files(repo)


def test_audit_output_directory_is_fixed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    allowed = inventory._ensure_output_dir(
        repo,
        inventory.DEFAULT_OUTPUT_DIR,
    )

    assert allowed == (repo / inventory.DEFAULT_OUTPUT_DIR).resolve()

    with pytest.raises(ValueError, match="must be exactly"):
        inventory._ensure_output_dir(
            repo,
            Path("studies/other"),
        )


def test_payload_has_no_wall_clock_field(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _track(
        repo,
        "scripts/consumer.py",
        "x = 'phase1_scores_dedup.csv'\n",
    )

    subprocess.run(
        [
            "git",
            "-c", "user.name=Test",
            "-c", "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )

    records = inventory.build_inventory(repo)
    payload = inventory.inventory_payload(repo, records)

    serialized = json.dumps(payload)

    assert "generated_at" not in serialized
    assert "timestamp" not in payload
    assert payload["audit_version"] == "M6.5.1-v1.0"


def test_current_repo_inventory_contains_canonical_consumers() -> None:
    records = inventory.build_inventory(REPO_ROOT)

    canonical = [
        record
        for record in records
        if record.classification == "canonical_consumer"
    ]

    assert canonical
    assert any(
        record.path == "scripts/trustforge_doctor.py"
        for record in canonical
    )


def test_current_repo_inventory_preserves_eval_runner() -> None:
    records = inventory.build_inventory(REPO_ROOT)

    by_path = {
        record.path: record
        for record in records
    }

    assert "eval/runner.py" in by_path
    assert (
        by_path["eval/runner.py"].migration_policy
        == "protected_do_not_migrate_in_m6_5_1"
    )
