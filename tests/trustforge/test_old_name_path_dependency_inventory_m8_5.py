from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "inventory_old_name_path_dependencies_m8_5.py"
SCHEMA = (
    REPO_ROOT
    / "studies"
    / "repository_migration"
    / "audits"
    / "m8_5"
    / "old_name_path_dependency_inventory.schema.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location("m8_5_inventory", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m8 = load_module()


def data():
    return m8.build_inventory()


def deps():
    return {x["dependency_id"]: x for x in data()["dependencies"]}


def test_schema_loads():
    parsed = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    assert parsed["schema_type"] == "m8_5_old_name_path_dependency_inventory"


def test_inventory_validates():
    m8.validate(data(), m8.load_schema(REPO_ROOT))


def test_m8_4_anchor():
    assert data()["source_acceptance"]["m8_4_acceptance_commit"] == "a11445a"


def test_inventory_is_read_only():
    assert all(value is False for value in data()["authority_boundary"].values())
    assert all(item["mutate_now"] is False for item in data()["dependencies"])


def test_remote_origin_is_rename_blocker():
    d = deps()["M8.5-001"]
    assert d["classification"] == "RENAME_BLOCKER"
    assert "GenCyberSynth_Phase1_Scaffold.git" in d["examples"][0]


def test_paper1_repository_identity_is_historical():
    assert deps()["M8.5-002"]["classification"] == "HISTORICAL_REFERENCE"
    assert deps()["M8.5-003"]["classification"] == "HISTORICAL_REFERENCE"


def test_paper2_embedded_absolute_repo_paths_are_historical():
    assert deps()["M8.5-004"]["classification"] == "HISTORICAL_REFERENCE"


def test_paper2_active_absolute_gencys_paths_are_portability_issue():
    assert deps()["M8.5-005"]["classification"] == "PORTABILITY_BLOCKER"


def test_gcs_core_is_compatibility_not_parent_rename_blocker():
    assert deps()["M8.5-006"]["classification"] == "COMPATIBILITY_REFERENCE"
    assert deps()["M8.5-014"]["classification"] == "SAFE_TO_RETAIN"


def test_legacy_env_aliases_are_compatibility():
    assert deps()["M8.5-007"]["classification"] == "COMPATIBILITY_REFERENCE"


def test_external_gencys_storage_is_safe_to_retain():
    assert deps()["M8.5-008"]["classification"] == "SAFE_TO_RETAIN"


def test_dynamic_repo_discovery_is_safe_to_retain():
    assert deps()["M8.5-009"]["classification"] == "SAFE_TO_RETAIN"


def test_current_branding_is_not_scientific_history():
    assert deps()["M8.5-010"]["classification"] == "DOCUMENTATION_ONLY"


def test_historical_lineage_branding_is_preserved():
    assert deps()["M8.5-011"]["classification"] == "HISTORICAL_REFERENCE"


def test_mixed_makefile_and_model_template_require_review():
    assert deps()["M8.5-012"]["classification"] == "REQUIRES_REVIEW"
    assert deps()["M8.5-013"]["classification"] == "REQUIRES_REVIEW"


def test_tests_and_tooling_are_separate_classes():
    assert deps()["M8.5-015"]["classification"] == "TEST_ONLY"
    assert deps()["M8.5-016"]["classification"] == "TOOLING_ONLY"


def test_only_one_current_rename_blocker_group():
    assert data()["summary"]["rename_blocker_ids"] == ["M8.5-001"]
    assert data()["summary"]["rename_blocker_count"] == 1


def test_principles_prohibit_bulk_substitution_logic():
    p = set(data()["principles"])
    assert "OLD_NAME_PRESENT != RENAME_BLOCKER" in p
    assert "HISTORICAL_PATH != CURRENT_CONFIGURATION" in p
    assert "PORTABILITY_BLOCKER != REPOSITORY_NAME_BLOCKER" in p
    assert "BULK_RENAME != GOVERNED_MIGRATION" in p


def test_summary_matches_dependencies():
    d = data()
    counts = {name: 0 for name in sorted(m8.CLASSES)}
    for item in d["dependencies"]:
        counts[item["classification"]] += 1
    assert d["summary"]["dependency_group_count"] == len(d["dependencies"])
    assert d["summary"]["class_counts"] == counts


def test_markdown_deterministic():
    first = m8.render_markdown(data())
    second = m8.render_markdown(data())
    assert first == second
    assert first.endswith("\n")
    assert not first.endswith("\n\n")


def test_no_destructive_operations():
    source = SCRIPT.read_text(encoding="utf-8")
    for token in [
        "git remote set-url",
        "git mv",
        "git clean",
        "git reset",
        "rm -rf",
        ".unlink(",
        "shutil.move",
        "shutil.copy",
    ]:
        assert token not in source
