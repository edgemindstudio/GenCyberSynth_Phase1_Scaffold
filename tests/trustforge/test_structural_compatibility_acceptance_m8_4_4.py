from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "accept_structural_compatibility_m8_4_4.py"
SCHEMA = (
    REPO_ROOT
    / "studies"
    / "repository_migration"
    / "audits"
    / "m8_4_4"
    / "structural_compatibility_acceptance.schema.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location("m8_4_4_acceptance", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m8 = load_module()


def data():
    return m8.acceptance_record()


def test_schema_loads():
    parsed = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    assert parsed["schema_type"] == "m8_4_4_structural_compatibility_acceptance"


def test_acceptance_validates():
    m8.validate(data(), m8.load_schema(REPO_ROOT))


def test_anchor_commits():
    d = data()["source_materialization"]
    assert d["m8_4_3_acceptance_commit"] == "f1db89a"
    assert d["prior_readiness_commit"] == "954ad5f"


def test_materialization_scope_exact():
    assert data()["materialization_scope"] == [
        "hpc/README.md",
        "studies/paper02_conditioning_audit/README.md",
        "studies/paper03_augmentation_regimes/README.md",
        "studies/paper04_selective_policies/README.md",
        "tests/trustforge/test_safe_scaffold_materialization_m8_4_3.py",
        "tools/materialize_safe_scaffolds_m8_4_3.py",
    ]


def test_historical_authority_roots_preserved():
    assert data()["historical_authority_roots"] == [
        "slurm",
        "papers/paper2_conditional_generation_done_right",
        "papers/paper3_when_does_synth_help",
        "papers/paper4_selective_synth_policies",
    ]


def test_no_runtime_or_authority_transfer():
    claims = data()["acceptance_claims"]
    assert claims["runtime_import_migration_observed"] is False
    assert claims["execution_path_migration_observed"] is False
    assert claims["historical_authority_transfer_observed"] is False


def test_historical_paths_unchanged():
    assert data()["acceptance_claims"]["historical_paths_unchanged"] is True


def test_suite_observation():
    obs = data()["observed_validation"]
    assert obs["trustforge_tests_passed"] == 658
    assert obs["trustforge_subtests_passed"] == 3
    assert obs["git_diff_check_clean"] is True


def test_no_downstream_authority_granted():
    assert all(value is False for value in data()["authority_boundary"].values())


def test_principles_include_reference_distinction():
    p = set(data()["principles"])
    assert "REFERENCE_IN_AUDIT != RUNTIME_DEPENDENCY" in p
    assert "REFERENCE_IN_TEST != EXECUTION_PATH" in p
    assert "STRUCTURAL_COMPATIBILITY_ACCEPTANCE != PAPER_MIGRATION_ACCEPTANCE" in p


def test_markdown_deterministic():
    first = m8.render_markdown(data())
    second = m8.render_markdown(data())
    assert first == second
    assert first.endswith("\n")
    assert not first.endswith("\n\n")


def test_no_destructive_operations():
    source = SCRIPT.read_text(encoding="utf-8")
    for token in [
        "git mv",
        "git clean",
        "git reset",
        "rm -rf",
        ".unlink(",
        "shutil.move",
        "shutil.copy",
    ]:
        assert token not in source
