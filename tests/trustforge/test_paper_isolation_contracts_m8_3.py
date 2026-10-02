from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "define_paper_isolation_contracts_m8_3.py"
SCHEMA = (
    REPO_ROOT
    / "studies"
    / "repository_migration"
    / "audits"
    / "m8_3"
    / "paper_isolation_contracts.schema.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location("m8_3_isolation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m8 = load_module()


def data():
    return m8.build_contract()


def rules():
    return {item["rule_id"]: item for item in data()["rules"]}


def test_schema_loads():
    parsed = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    assert parsed["schema_type"] == "m8_3_paper_isolation_contracts"


def test_contract_validates():
    m8.validate(data(), m8.load_schema(REPO_ROOT))


def test_m8_2_anchor():
    assert data()["source_boundary"]["m8_2_acceptance_commit"] == "d42e022"


def test_no_migration_authority():
    d = data()
    assert not any(d["authority_boundary"].values())
    for item in d["rules"]:
        assert item["migration_authorized"] is False
        assert item["implementation_move_authorized"] is False


def test_each_paper_has_core_transitional_local_and_cross_paper_rules():
    by_paper = {"paper02": set(), "paper03": set(), "paper04": set()}
    for item in data()["rules"]:
        if item["paper"] in by_paper:
            by_paper[item["paper"]].add(item["dependency_class"])

    expected = {
        "CORE_ALLOWED",
        "TRANSITIONAL_ALLOWED",
        "STUDY_LOCAL_ONLY",
        "FORBIDDEN_CROSS_PAPER",
    }
    for classes in by_paper.values():
        assert expected <= classes


def test_cross_paper_science_is_forbidden():
    r = rules()
    assert r["P2-XPAPER-001"]["dependency_class"] == "FORBIDDEN_CROSS_PAPER"
    assert r["P3-XPAPER-001"]["dependency_class"] == "FORBIDDEN_CROSS_PAPER"
    assert r["P4-XPAPER-001"]["dependency_class"] == "FORBIDDEN_CROSS_PAPER"


def test_historical_shared_runtime_is_transitional_only():
    r = rules()
    assert r["P2-TRANS-001"]["dependency_class"] == "TRANSITIONAL_ALLOWED"
    assert r["P3-TRANS-001"]["dependency_class"] == "TRANSITIONAL_ALLOWED"
    assert r["P4-TRANS-001"]["dependency_class"] == "TRANSITIONAL_ALLOWED"


def test_paper_specific_science_remains_local():
    r = rules()
    assert r["P2-LOCAL-001"]["dependency_class"] == "STUDY_LOCAL_ONLY"
    assert r["P3-LOCAL-001"]["dependency_class"] == "STUDY_LOCAL_ONLY"
    assert r["P4-LOCAL-001"]["dependency_class"] == "STUDY_LOCAL_ONLY"


def test_historical_compatibility_is_not_native_default():
    r = rules()
    assert (
        r["ALL-HIST-001"]["dependency_class"]
        == "HISTORICAL_COMPATIBILITY_ONLY"
    )


def test_model_plugin_migration_is_deferred():
    r = rules()
    assert r["ALL-PLUGIN-001"]["dependency_class"] == "DEFERRED"


def test_principles_present():
    p = set(data()["principles"])
    assert "PAPER_SPECIFIC_SCIENCE != SHARED_FRAMEWORK_DEFAULT" in p
    assert "TRANSITIONAL_ALLOWED != PERMANENTLY_ALLOWED" in p
    assert "CROSS_PAPER_IMPORT != SHARED_CORE" in p


def test_summary_matches_rules():
    d = data()
    counts = {name: 0 for name in sorted(m8.DEPENDENCY_CLASSES)}
    for item in d["rules"]:
        counts[item["dependency_class"]] += 1

    assert d["summary"]["rule_count"] == len(d["rules"])
    assert d["summary"]["class_counts"] == counts


def test_rule_ids_are_unique():
    ids = [item["rule_id"] for item in data()["rules"]]
    assert len(ids) == len(set(ids))


def test_markdown_deterministic_and_single_eof_newline():
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
        "rm -rf",
    ]:
        assert token not in source
