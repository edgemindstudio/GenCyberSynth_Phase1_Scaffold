from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "define_structural_normalization_plan_m8_4.py"
SCHEMA = (
    REPO_ROOT
    / "studies"
    / "repository_migration"
    / "audits"
    / "m8_4"
    / "structural_normalization_plan.schema.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location("m8_4_plan", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m8 = load_module()


def data():
    return m8.build_plan()


def components():
    return {x["component"]: x for x in data()["components"]}


def test_schema_loads():
    parsed = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    assert parsed["schema_type"] == "m8_4_structural_normalization_plan"


def test_plan_validates():
    m8.validate(data(), m8.load_schema(REPO_ROOT))


def test_m8_3_anchor():
    assert data()["source_isolation_contract"]["m8_3_acceptance_commit"] == "c2e06fa"


def test_no_structural_mutation_authority():
    d = data()
    assert not any(d["authority_boundary"].values())
    for record in d["components"]:
        assert record["move_authorized"] is False
        assert record["rename_authorized"] is False
        assert record["rewrite_authorized"] is False


def test_paper1_is_preserved():
    c = components()
    assert c["paper01_native_study"]["normalization_class"] == "PRESERVE_CURRENT"
    assert c["paper01_native_study"]["target_location"] == "studies/paper01_benchmark/"


def test_papers_2_3_4_are_migrate_later():
    c = components()
    assert c["paper02_study"]["normalization_class"] == "MIGRATE_LATER"
    assert c["paper03_study"]["normalization_class"] == "MIGRATE_LATER"
    assert c["paper04_study"]["normalization_class"] == "MIGRATE_LATER"


def test_active_paper3_and_placeholder_are_distinct():
    c = components()
    assert c["paper03_study"]["current_location"] == "papers/paper3_when_does_synth_help/"
    assert c["paper03_placeholder"]["current_location"] == "papers/paper03_when_does_synth_help/"
    assert c["paper03_placeholder"]["normalization_class"] == "COMPATIBILITY_ONLY"


def test_historical_runtime_remains_compatibility_only():
    c = components()
    for name in [
        "historical_cli_runtime",
        "historical_adapters",
        "historical_dataset_runtime",
        "historical_evaluator",
    ]:
        assert c[name]["normalization_class"] == "COMPATIBILITY_ONLY"


def test_model_packages_are_plugins():
    c = components()
    for name in [
        "gan_plugin",
        "vae_plugin",
        "diffusion_plugin",
        "autoregressive_plugin",
        "gaussianmixture_plugin",
        "restrictedboltzmann_plugin",
        "maskedautoflow_plugin",
    ]:
        assert c[name]["normalization_class"] == "PLUGIN"


def test_native_namespaces_are_created_not_backfilled_by_move():
    c = components()
    for name in [
        "trustforge_data_namespace",
        "trustforge_evaluation_namespace",
        "trustforge_orchestration_namespace",
        "trustforge_models_namespace",
        "trustforge_policies_namespace",
        "trustforge_augmentation_namespace",
        "hpc_native_root",
    ]:
        assert c[name]["normalization_class"] == "CREATE_NATIVE"
        assert c[name]["move_authorized"] is False


def test_provenance_and_storage_are_preserved():
    c = components()
    assert c["trustforge_provenance"]["normalization_class"] == "PRESERVE_CURRENT"
    assert c["trustforge_storage"]["normalization_class"] == "PRESERVE_CURRENT"


def test_target_shape_contains_all_native_studies():
    shape = set(data()["target_repository_shape"])
    assert "studies/paper01_benchmark/" in shape
    assert "studies/paper02_conditioning_audit/" in shape
    assert "studies/paper03_augmentation_regimes/" in shape
    assert "studies/paper04_selective_policies/" in shape


def test_principles_preserve_interface_first_migration():
    p = set(data()["principles"])
    assert "NORMALIZED_TARGET != IMMEDIATE_MOVE" in p
    assert "COMPATIBILITY_SURFACE != NATIVE_CORE" in p
    assert "INTERFACE_FIRST_BEFORE_IMPLEMENTATION_MIGRATION" in p
    assert "HISTORICAL_EVIDENCE_REMAINS_PRESERVED" in p


def test_summary_matches_components():
    d = data()
    counts = {name: 0 for name in sorted(m8.NORMALIZATION_CLASSES)}
    for record in d["components"]:
        counts[record["normalization_class"]] += 1
    assert d["summary"]["component_count"] == len(d["components"])
    assert d["summary"]["class_counts"] == counts


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

def test_configs_root_is_mixed_and_requires_later_partition():
    c = components()
    cfg = c["configs_root"]
    assert cfg["normalization_class"] == "MIGRATE_LATER"
    assert cfg["target_location"] == "configs/"
    assert cfg["move_authorized"] is False
    assert "mixed" in cfg["rationale"].lower()


def test_target_configs_surface_is_not_classified_compatibility_only():
    c = components()
    assert "configs/" in set(data()["target_repository_shape"])
    assert c["configs_root"]["normalization_class"] != "COMPATIBILITY_ONLY"
