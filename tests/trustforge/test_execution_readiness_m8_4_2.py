from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "define_execution_readiness_m8_4_2.py"
SCHEMA = (
    REPO_ROOT
    / "studies"
    / "repository_migration"
    / "audits"
    / "m8_4_2"
    / "execution_readiness.schema.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location("m8_4_2_readiness", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m8 = load_module()


def data():
    return m8.build_readiness()


def actions():
    return {x["action_id"]: x for x in data()["actions"]}


def test_schema_loads():
    parsed = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    assert parsed["schema_type"] == "m8_4_2_execution_readiness"


def test_readiness_validates():
    m8.validate(data(), m8.load_schema(REPO_ROOT))


def test_m8_4_1_anchor():
    assert data()["source_structural_plan"]["m8_4_1_acceptance_commit"] == "22f3df2"


def test_authority_boundary_is_narrow():
    boundary = data()["authority_boundary"]
    assert boundary["safe_scaffold_creation_authorized"] is True
    assert boundary["historical_moves_authorized"] is False
    assert boundary["import_rewrites_authorized"] is False
    assert boundary["paper_migrations_authorized"] is False
    assert boundary["plugin_relocations_authorized"] is False
    assert boundary["historical_cleanup_authorized"] is False


def test_only_safe_scaffold_actions_are_authorized():
    for record in data()["actions"]:
        assert record["authorized"] is (
            record["readiness_class"] == "AUTHORIZED_SAFE_SCAFFOLD"
        )


def test_existing_native_namespaces_are_preserved_not_recreated():
    a = actions()
    for rule_id in [
        "SAFE-001",
        "SAFE-002",
        "SAFE-003",
        "SAFE-004",
        "SAFE-005",
        "SAFE-006",
    ]:
        assert a[rule_id]["readiness_class"] == "PRESERVE_AS_IS"
        assert a[rule_id]["authorized"] is False


def test_hpc_scaffold_is_authorized():
    a = actions()
    assert a["SAFE-007"]["target"] == "hpc/"
    assert a["SAFE-007"]["readiness_class"] == "AUTHORIZED_SAFE_SCAFFOLD"
    assert a["SAFE-007"]["authorized"] is True


def test_future_study_scaffolds_are_authorized_without_migration():
    a = actions()
    expected = {
        "SAFE-008": "studies/paper02_conditioning_audit/",
        "SAFE-009": "studies/paper03_augmentation_regimes/",
        "SAFE-010": "studies/paper04_selective_policies/",
    }
    for rule_id, target in expected.items():
        assert a[rule_id]["target"] == target
        assert a[rule_id]["readiness_class"] == "AUTHORIZED_SAFE_SCAFFOLD"
        assert a[rule_id]["authorized"] is True
        assert a[rule_id]["paper_migration_authorized"] is False


def test_historical_runtime_is_guarded_compatibility():
    a = actions()
    for rule_id in [
        "GUARD-001",
        "GUARD-002",
        "GUARD-003",
        "GUARD-004",
        "GUARD-005",
        "GUARD-006",
    ]:
        assert a[rule_id]["readiness_class"] == "GUARDED_COMPATIBILITY"


def test_paper_migrations_are_not_authorized():
    a = actions()
    for rule_id in ["BLOCK-001", "BLOCK-002", "BLOCK-003"]:
        assert (
            a[rule_id]["readiness_class"]
            == "MIGRATION_NOT_YET_AUTHORIZED"
        )
        assert a[rule_id]["authorized"] is False


def test_plugin_relocation_is_not_authorized():
    a = actions()
    assert (
        a["BLOCK-004"]["readiness_class"]
        == "MIGRATION_NOT_YET_AUTHORIZED"
    )
    assert a["BLOCK-004"]["authorized"] is False


def test_uncertainty_remains_deferred():
    assert actions()["DEFER-001"]["readiness_class"] == "DEFERRED"


def test_no_action_authorizes_historical_move_or_rewrite():
    for record in data()["actions"]:
        assert record["historical_move_authorized"] is False
        assert record["import_rewrite_authorized"] is False
        assert record["paper_migration_authorized"] is False


def test_principles_present():
    p = set(data()["principles"])
    assert "SCAFFOLD_CREATION != MIGRATION" in p
    assert "DESTINATION_EXISTENCE != AUTHORITY_TRANSFER" in p
    assert "HISTORICAL_SOURCE_REMAINS_AUTHORITATIVE_UNTIL_MIGRATION_ACCEPTED" in p


def test_summary_matches_actions():
    d = data()
    counts = {name: 0 for name in sorted(m8.READINESS_CLASSES)}
    for record in d["actions"]:
        counts[record["readiness_class"]] += 1
    assert d["summary"]["action_count"] == len(d["actions"])
    assert d["summary"]["class_counts"] == counts


def test_action_ids_unique():
    ids = [record["action_id"] for record in data()["actions"]]
    assert len(ids) == len(set(ids))


def test_markdown_deterministic_eof():
    first = m8.render_markdown(data())
    second = m8.render_markdown(data())
    assert first == second
    assert first.endswith("\n")
    assert not first.endswith("\n\n")


def test_no_destructive_operations():
    source = SCRIPT.read_text(encoding="utf-8")
    for token in [
        "shutil.rmtree(",
        ".unlink(",
        "os.remove(",
        "git clean",
        "git reset",
        "git checkout",
        "git mv",
        "rm -rf",
    ]:
        assert token not in source
