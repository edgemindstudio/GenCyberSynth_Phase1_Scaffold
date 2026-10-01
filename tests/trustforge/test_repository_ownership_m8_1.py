from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "audit_repository_ownership_m8_1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m8_1_inventory_v2", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inv = load_module()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/trustforge/provenance/git.py", "SHARED_TRUSTFORGE_CORE"),
        ("src/trustforge/storage/paths.py", "SHARED_TRUSTFORGE_CORE"),
        ("schemas/experiment.schema.yaml", "SHARED_TRUSTFORGE_CORE"),
        ("schemas/manifest.schema.yaml", "SHARED_TRUSTFORGE_CORE"),
        ("schemas/study.schema.yaml", "SHARED_TRUSTFORGE_CORE"),
        ("docs/TRUSTFORGE_ARCHITECTURE.md", "SHARED_TRUSTFORGE_CORE"),
        ("src/trustforge/paper01_exports.py", "PAPER01_SPECIFIC"),
        ("studies/paper01_benchmark/experiment_index.yaml", "PAPER01_SPECIFIC"),
        ("schemas/examples/paper01/study.yaml", "PAPER01_SPECIFIC"),
        (
            "manifests/schemas/paper01_execution_evidence_linkage.schema.json",
            "PAPER01_SPECIFIC",
        ),
        ("docs/paper01_execution_evidence_linkage_schema.md", "PAPER01_SPECIFIC"),
        ("tools/aggregate_phase1.py", "PAPER01_SPECIFIC"),
        ("tools/build_phase1_scores.py", "PAPER01_SPECIFIC"),
        ("tools/check_phase1_integrity.py", "PAPER01_SPECIFIC"),
        ("scripts/phase1_report.py", "PAPER01_SPECIFIC"),
        (
            "papers/paper2_conditional_generation_done_right/scripts/x.py",
            "PAPER02_SPECIFIC",
        ),
        (
            "papers/paper3_when_does_synth_help/scripts/x.py",
            "PAPER03_SPECIFIC",
        ),
        (
            "papers/paper4_selective_synth_policies/scripts/x.py",
            "PAPER04_SPECIFIC",
        ),
        ("papers/paper03_when_does_synth_help/README.md", "SCAFFOLD_ONLY"),
        ("papers/paper05_shift_calibration/README.md", "SCAFFOLD_ONLY"),
        ("app/main.py", "MULTI_PAPER_SHARED_RUNTIME"),
        ("adapters/registry.py", "MULTI_PAPER_SHARED_RUNTIME"),
        ("eval/runner.py", "MULTI_PAPER_SHARED_RUNTIME"),
        ("gan/train.py", "MULTI_PAPER_SHARED_RUNTIME"),
        ("vae/sample.py", "MULTI_PAPER_SHARED_RUNTIME"),
        ("configs/config.yaml", "MULTI_PAPER_SHARED_RUNTIME"),
        ("Makefile", "MULTI_PAPER_SHARED_RUNTIME"),
        ("configs/paper1_final_rerun.yaml", "HISTORICAL_COMPATIBILITY"),
        ("configs/paper2_500.yaml", "HISTORICAL_COMPATIBILITY"),
        ("slurm/run_paper1.slurm", "HISTORICAL_COMPATIBILITY"),
        (".github/workflows/ci.yml", "OPERATIONAL_INFRASTRUCTURE"),
        ("requirements.txt", "OPERATIONAL_INFRASTRUCTURE"),
        ("model-template/README.md", "OPERATIONAL_INFRASTRUCTURE"),
        ("tests/test_smoke.py", "OPERATIONAL_INFRASTRUCTURE"),
        ("configs/config_500.yaml", "AMBIGUOUS_REQUIRES_REVIEW"),
    ],
)
def test_path_classification(path, expected):
    assert inv.classify_path(path) == expected


def test_all_classifications_are_declared():
    for path in [
        "src/trustforge/provenance/git.py",
        "app/main.py",
        "papers/paper3_when_does_synth_help/README.md",
        ".github/workflows/ci.yml",
        "weird/new/path.txt",
    ]:
        assert inv.classify_path(path) in inv.CLASSIFICATIONS


def test_paper03_scaffold_is_not_active_paper3():
    assert inv.classify_path(
        "papers/paper03_when_does_synth_help/README.md"
    ) == "SCAFFOLD_ONLY"
    assert inv.classify_path(
        "papers/paper3_when_does_synth_help/README.md"
    ) == "PAPER03_SPECIFIC"


def test_generic_schemas_are_not_paper01_examples():
    assert inv.classify_path(
        "schemas/experiment.schema.yaml"
    ) == "SHARED_TRUSTFORGE_CORE"
    assert inv.classify_path(
        "schemas/examples/paper01/experiment_ustc_gan_seed42.yaml"
    ) == "PAPER01_SPECIFIC"


def test_dependency_patterns():
    assert inv.APP_MAIN_RE.search("python -m app.main eval --model diffusion")
    assert inv.ROOT_IMPORT_RE.search("from common.data import load_dataset_npy")
    assert inv.TRUSTFORGE_IMPORT_RE.search("from trustforge.storage import paths")


def test_build_inventory_has_no_authority_assignments():
    data = inv.build_inventory(REPO_ROOT)
    boundary = data["authority_boundary"]
    assert boundary["future_owner_assignments_performed"] is False
    assert boundary["migration_actions_authorized"] is False
    assert boundary["historical_artifacts_modified"] is False
    assert boundary["authority_decision"] is None

    for row in data["files"]:
        assert row["future_owner"] is None
        assert row["migration_action"] is None
        assert row["authority_decision"] is None


def test_repository_contains_expected_accepted_paper1_anchor():
    data = inv.build_inventory(REPO_ROOT)
    paths = {row["path"] for row in data["files"]}
    assert "studies/paper01_benchmark/execution_evidence/index.yaml" in paths
    assert "src/trustforge/paper01_execution_evidence.py" in paths


def test_repository_contains_active_papers_2_3_4():
    data = inv.build_inventory(REPO_ROOT)
    classes = data["summary"]["classification_counts"]
    assert classes["PAPER02_SPECIFIC"] > 0
    assert classes["PAPER03_SPECIFIC"] > 0
    assert classes["PAPER04_SPECIFIC"] > 0


def test_operational_infrastructure_is_present():
    data = inv.build_inventory(REPO_ROOT)
    assert data["summary"]["classification_counts"]["OPERATIONAL_INFRASTRUCTURE"] > 0


def test_multi_paper_runtime_is_not_shared_core():
    assert inv.classify_path("app/main.py") == "MULTI_PAPER_SHARED_RUNTIME"
    assert inv.classify_path("eval/runner.py") == "MULTI_PAPER_SHARED_RUNTIME"
    assert inv.classify_path("app/main.py") != "SHARED_TRUSTFORGE_CORE"


def test_output_location_is_repository_local_audit():
    assert inv.OUTPUT_REL.as_posix() == "studies/repository_migration/audits/m8_1"


def test_markdown_preserves_guardrails():
    data = inv.build_inventory(REPO_ROOT)
    md = inv.render_markdown(data)
    assert "MULTI_PAPER_SHARED_RUNTIME" in md
    assert "does not imply `SHARED_TRUSTFORGE_CORE`" in md
    assert "OPERATIONAL_INFRASTRUCTURE" in md
    assert "not a scientific ownership assignment" in md
    assert "future_owner" in md
    assert "remain null" in md


def test_no_destructive_historical_operations_in_source():
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = [
        "shutil.rmtree(",
        ".unlink(",
        "os.remove(",
        "git clean",
        "rm -rf",
    ]
    for token in forbidden:
        assert token not in source
