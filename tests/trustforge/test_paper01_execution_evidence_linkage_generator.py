from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_paper01_execution_evidence_linkage.py"
INDEX_PATH = REPO_ROOT / "studies" / "paper01_benchmark" / "experiment_index.yaml"
INVENTORY_PATH = (
    REPO_ROOT
    / "studies"
    / "paper01_benchmark"
    / "audits"
    / "m6_4_1"
    / "paper01_execution_evidence_inventory.json"
)
SCHEMA_PATH = (
    REPO_ROOT
    / "manifests"
    / "schemas"
    / "paper01_execution_evidence_linkage.schema.json"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_paper01_execution_evidence_linkage",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generator: {GENERATOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


def _inputs():
    with INDEX_PATH.open(encoding="utf-8") as handle:
        experiment_index = yaml.safe_load(handle)

    with INVENTORY_PATH.open(encoding="utf-8") as handle:
        inventory = json.load(handle)

    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        schema = json.load(handle)

    return experiment_index, inventory, schema


def _records():
    experiment_index, inventory, schema = _inputs()
    return generator.build_all_records(
        experiment_index,
        inventory,
        schema=schema,
    )


def _record_by_id(experiment_id: str):
    for index_entry, record in _records():
        if index_entry["experiment_id"] == experiment_id:
            return record
    raise AssertionError(f"Missing linkage record: {experiment_id}")


def test_generator_builds_exactly_42_schema_valid_records():
    records = _records()
    assert len(records) == 42
    assert len({entry["experiment_id"] for entry, _ in records}) == 42


def test_generator_preserves_frozen_scientific_ids():
    experiment_index, _, _ = _inputs()
    expected_ids = [entry["experiment_id"] for entry in experiment_index["entries"]]
    generated_ids = [
        record["experiment"]["experiment_id"]
        for _, record in _records()
    ]
    assert generated_ids == expected_ids


def test_all_records_have_exactly_one_accepted_evaluator():
    for _, record in _records():
        accepted = [
            execution
            for execution in record["historical_executions"]
            if "accepted_evaluator" in execution["role"]
        ]

        assert len(accepted) == 1
        assert accepted[0]["do_eval"] is True
        assert (
            accepted[0]["config_sha1"]
            == record["accepted_evaluation"]["execution"]["config_sha1"]
        )


def test_latest_json_never_becomes_accepted_summary():
    alias_count = 0

    for _, record in _records():
        accepted_path = Path(
            record["accepted_evaluation"]["accepted_summary"]["path"]
        )
        assert accepted_path.name != "latest.json"

        if record["non_authoritative_aliases"]["latest_summary"]["present"]:
            alias_count += 1

    assert alias_count == 14


def test_all_current_artifact_lineages_are_strong_candidates_not_proven():
    statuses = {}

    for _, record in _records():
        artifact = record["artifact_production"]
        statuses[artifact["status"]] = statuses.get(artifact["status"], 0) + 1

        assert artifact["status"] != "verified"
        assert artifact["lineage_claim"]["proven"] is False

    assert statuses == {"strong_candidate": 42}


def test_cic_gmm_seed42_keeps_execution_roles_separate():
    record = _record_by_id("cicmaldroid_gaussianmixture_b2000_seed42")

    assert record["accepted_evaluation"]["execution"]["job_id"] == "246862"
    assert (
        record["accepted_evaluation"]["execution"]["config_sha1"]
        == "0d8e51e3e3a54b2816ac72bed5ee001973498cbb"
    )
    assert record["accepted_evaluation"]["execution"]["do_train"] is False
    assert record["accepted_evaluation"]["execution"]["do_synth"] is False
    assert record["accepted_evaluation"]["execution"]["do_eval"] is True

    artifact = record["artifact_production"]
    assert artifact["status"] == "strong_candidate"
    assert len(artifact["executions"]) == 1
    assert artifact["executions"][0]["job_id"] == "246427"
    assert (
        artifact["executions"][0]["config_sha1"]
        == "5d601b08f992905c7764f13d2e8c811550461eda"
    )
    assert artifact["executions"][0]["do_synth"] is True
    assert artifact["lineage_claim"]["proven"] is False

    jobs = {
        execution["job_id"]: set(execution["role"])
        for execution in record["historical_executions"]
    }

    assert set(jobs) == {"246364", "246427", "246862"}
    assert jobs["246364"] == {"historical_execution"}
    assert jobs["246427"] == {
        "historical_execution",
        "artifact_producer_candidate",
    }
    assert jobs["246862"] == {
        "historical_execution",
        "accepted_evaluator",
    }


def test_cic_gmm_seed42_preserves_missing_manifest_and_special_layout():
    record = _record_by_id("cicmaldroid_gaussianmixture_b2000_seed42")

    assert record["manifest_evidence"]["seed_specific"] == {
        "status": "absent",
        "path": None,
        "authority": "none",
    }
    assert record["manifest_evidence"]["shared_root"]["status"] == "present"
    assert (
        record["manifest_evidence"]["shared_root"]["authority"]
        == "compatibility_evidence"
    )

    synthetic = record["synthetic_evidence"]
    assert synthetic["status"] == "present"
    assert synthetic["file_count"] == 10000
    assert synthetic["layout"] == "class_then_seed"
    assert synthetic["layout_counts"] == {"class_then_seed": 10000}


def test_ustc_gmm_seed42_preserves_dual_historical_layout():
    record = _record_by_id("ustc_gaussianmixture_b2000_seed42")

    synthetic = record["synthetic_evidence"]
    assert synthetic["file_count"] == 36000
    assert synthetic["layout"] == "mixed:class_then_seed,seed_directory"
    assert synthetic["layout_counts"] == {
        "class_then_seed": 18000,
        "seed_directory": 18000,
    }


def test_generation_is_semantically_deterministic():
    first = _records()
    second = _records()

    first_hashes = {
        entry["experiment_id"]: generator.semantic_sha256(record)
        for entry, record in first
    }
    second_hashes = {
        entry["experiment_id"]: generator.semantic_sha256(record)
        for entry, record in second
    }

    assert first_hashes == second_hashes


def test_study_index_is_deterministic_and_complete():
    records = _records()

    first = generator.build_index(records)
    second = generator.build_index(records)

    assert first == second
    assert first["record_count"] == 42
    assert len(first["entries"]) == 42
    assert len({entry["semantic_sha256"] for entry in first["entries"]}) == 42


def test_output_location_rejects_outside_repository(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(
        generator.LinkageGenerationError,
        match="must be inside repository",
    ):
        generator.validate_output_location(
            tmp_path / "outside",
            repo,
            [],
        )


def test_output_location_rejects_historical_root_inside_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    historical = repo / "historical_artifacts"
    output = historical / "linkage"

    output.mkdir(parents=True)

    with pytest.raises(
        generator.LinkageGenerationError,
        match="must not be inside historical artifact root",
    ):
        generator.validate_output_location(
            output,
            repo,
            [historical],
        )


def test_write_outputs_is_repeatable(tmp_path: Path):
    records = _records()
    output = tmp_path / "execution_evidence"

    generator.write_outputs(output, records)

    first = {
        path.name: path.read_bytes()
        for path in sorted(output.glob("*.yaml"))
    }

    assert len(first) == 43
    assert "index.yaml" in first

    generator.write_outputs(output, records)

    second = {
        path.name: path.read_bytes()
        for path in sorted(output.glob("*.yaml"))
    }

    assert first == second
