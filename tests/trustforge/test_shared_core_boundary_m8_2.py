from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "define_shared_core_boundary_m8_2.py"
SCHEMA = REPO_ROOT / "studies/repository_migration/audits/m8_2/shared_core_boundary.schema.yaml"

def load_module():
    spec = importlib.util.spec_from_file_location("m8_2_boundary", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

m8 = load_module()

def data():
    return m8.build_boundary()

def by_name():
    return {x["capability"]: x for x in data()["capabilities"]}

def test_schema_loads():
    assert yaml.safe_load(SCHEMA.read_text())["schema_type"] == "m8_2_shared_core_boundary"

def test_boundary_validates():
    d = data()
    m8.validate(d, m8.load_schema(REPO_ROOT))

def test_m8_1_anchor():
    d = data()
    assert d["source_baseline"]["m8_1_repository_baseline"] == "d60f9c97e6b01b78cd615aa0fdb4aa61f3acce51"
    assert d["source_baseline"]["m8_1_acceptance_commit"] == "9487df0"

def test_no_migration_authority():
    d = data()
    assert not any(d["authority_boundary"].values())
    for x in d["capabilities"]:
        assert x["migration_authorized"] is False
        assert x["implementation_move_authorized"] is False

def test_shared_runtime_not_core_impl():
    b = by_name()
    assert b["historical_shared_cli_runtime"]["boundary_class"] == "TRANSITIONAL_RUNTIME"
    assert b["historical_shared_evaluator"]["boundary_class"] == "TRANSITIONAL_RUNTIME"
    assert b["historical_shared_dataset_runtime"]["boundary_class"] == "TRANSITIONAL_RUNTIME"

def test_interfaces_separate_from_implementations():
    b = by_name()
    assert b["adapter_interface"]["boundary_class"] == "CORE_INTERFACE"
    assert b["historical_concrete_adapters"]["boundary_class"] == "TRANSITIONAL_RUNTIME"
    assert b["evaluation_interface"]["boundary_class"] == "CORE_INTERFACE"
    assert b["historical_shared_evaluator"]["boundary_class"] == "TRANSITIONAL_RUNTIME"

def test_models_are_plugins():
    assert by_name()["historical_model_implementations"]["boundary_class"] == "MODEL_PLUGIN"

def test_papers_are_study_owned():
    b = by_name()
    assert b["paper02_conditioning_study"]["boundary_class"] == "STUDY_OWNED"
    assert b["paper03_augmentation_study"]["boundary_class"] == "STUDY_OWNED"
    assert b["paper04_selective_policy_study"]["boundary_class"] == "STUDY_OWNED"

def test_accepted_evidence_deferred():
    assert by_name()["accepted_evidence_contract"]["boundary_class"] == "DEFERRED"

def test_core_contracts():
    b = by_name()
    for n in {
        "study_identity_contract", "experiment_identity_contract", "execution_identity_contract",
        "storage_resolution", "provenance_hashing", "git_provenance", "native_manifest_contract"
    }:
        assert b[n]["boundary_class"] == "CORE_CONTRACT"

def test_principles():
    p = set(data()["principles"])
    assert "MULTI_PAPER_SHARED_RUNTIME != SHARED_TRUSTFORGE_CORE" in p
    assert "REUSABLE != CORE" in p
    assert "INTERFACE_OWNERSHIP != CURRENT_IMPLEMENTATION_OWNERSHIP" in p
    assert "HPC_BACKEND != SCIENTIFIC_EXPERIMENT_DEFINITION" in p

def test_markdown_deterministic_eof():
    a = m8.render_markdown(data())
    b = m8.render_markdown(data())
    assert a == b
    assert a.endswith("\n")
    assert not a.endswith("\n\n")

def test_summary_matches_records():
    d = data()
    counts = {c: 0 for c in sorted(m8.BOUNDARY_CLASSES)}
    for x in d["capabilities"]:
        counts[x["boundary_class"]] += 1
    assert d["summary"]["capability_count"] == len(d["capabilities"])
    assert d["summary"]["class_counts"] == counts

def test_no_destructive_operations():
    s = SCRIPT.read_text()
    for token in ["shutil.rmtree(", ".unlink(", "os.remove(", "git clean", "git reset", "git checkout", "rm -rf"]:
        assert token not in s
