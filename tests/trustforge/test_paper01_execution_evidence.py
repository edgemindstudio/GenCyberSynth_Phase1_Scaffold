from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from trustforge.paper01_execution_evidence import (
    Paper01LinkageValidationError,
    load_paper01_linkage_study,
    semantic_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_LINKAGE_DIR = (
    REPO_ROOT
    / "studies"
    / "paper01_benchmark"
    / "execution_evidence"
)
SOURCE_SCHEMA = (
    REPO_ROOT
    / "manifests"
    / "schemas"
    / "paper01_execution_evidence_linkage.schema.json"
)


def _copy_materialization(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    target = (
        repo
        / "studies"
        / "paper01_benchmark"
        / "execution_evidence"
    )
    schema_target = (
        repo
        / "manifests"
        / "schemas"
        / "paper01_execution_evidence_linkage.schema.json"
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    schema_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_LINKAGE_DIR, target)
    shutil.copy2(SOURCE_SCHEMA, schema_target)
    return repo


def _load_yaml(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            value,
            handle,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=1000,
        )


def test_loads_complete_canonical_study():
    study = load_paper01_linkage_study(REPO_ROOT)

    assert study.index["record_count"] == 42
    assert len(study.records) == 42
    assert len(study.by_experiment_id()) == 42


def test_lookup_by_experiment_id():
    study = load_paper01_linkage_study(REPO_ROOT)

    record = study.get("cicmaldroid_gaussianmixture_b2000_seed42")

    assert record.experiment_id == "cicmaldroid_gaussianmixture_b2000_seed42"
    assert (
        record.data["accepted_evaluation"]["execution"]["job_id"]
        == "246862"
    )
    assert (
        record.data["artifact_production"]["status"]
        == "strong_candidate"
    )


def test_unknown_experiment_id_raises_key_error():
    study = load_paper01_linkage_study(REPO_ROOT)

    with pytest.raises(KeyError, match="Unknown Paper 1 experiment_id"):
        study.get("does_not_exist")


def test_all_records_have_exactly_one_accepted_evaluator():
    study = load_paper01_linkage_study(REPO_ROOT)

    for record in study.records:
        accepted = [
            execution
            for execution in record.data["historical_executions"]
            if "accepted_evaluator" in execution["role"]
        ]
        assert len(accepted) == 1


def test_all_current_artifact_statuses_are_strong_candidate():
    study = load_paper01_linkage_study(REPO_ROOT)

    statuses = {
        record.data["artifact_production"]["status"]
        for record in study.records
    }
    assert statuses == {"strong_candidate"}


def test_semantic_hash_matches_index():
    study = load_paper01_linkage_study(REPO_ROOT)
    index_by_id = {
        entry["experiment_id"]: entry
        for entry in study.index["entries"]
    }

    for record in study.records:
        assert semantic_sha256(record.data) == (
            index_by_id[record.experiment_id]["semantic_sha256"]
        )


def test_rejects_modified_record_with_stale_index_hash(tmp_path: Path):
    repo = _copy_materialization(tmp_path)
    path = (
        repo
        / "studies"
        / "paper01_benchmark"
        / "execution_evidence"
        / "cicmaldroid_gaussianmixture_b2000_seed42.yaml"
    )

    record = _load_yaml(path)
    record["artifact_production"]["lineage_claim"]["reason"] += " changed"
    _write_yaml(path, record)

    with pytest.raises(
        Paper01LinkageValidationError,
        match="semantic SHA256 mismatch",
    ):
        load_paper01_linkage_study(repo)


def test_rejects_latest_json_as_accepted_summary(tmp_path: Path):
    repo = _copy_materialization(tmp_path)
    path = (
        repo
        / "studies"
        / "paper01_benchmark"
        / "execution_evidence"
        / "cicmaldroid_gaussianmixture_b2000_seed42.yaml"
    )

    record = _load_yaml(path)
    record["accepted_evaluation"]["accepted_summary"]["path"] = (
        "/historical/summaries/latest.json"
    )
    _write_yaml(path, record)

    index_path = path.parent / "index.yaml"
    index = _load_yaml(index_path)
    for entry in index["entries"]:
        if entry["experiment_id"] == record["experiment"]["experiment_id"]:
            entry["semantic_sha256"] = semantic_sha256(record)
            break
    _write_yaml(index_path, index)

    with pytest.raises(
        Paper01LinkageValidationError,
        match="latest.json",
    ):
        load_paper01_linkage_study(repo)


def test_rejects_duplicate_experiment_id_in_index(tmp_path: Path):
    repo = _copy_materialization(tmp_path)
    index_path = (
        repo
        / "studies"
        / "paper01_benchmark"
        / "execution_evidence"
        / "index.yaml"
    )

    index = _load_yaml(index_path)
    index["entries"][1]["experiment_id"] = (
        index["entries"][0]["experiment_id"]
    )
    _write_yaml(index_path, index)

    with pytest.raises(
        Paper01LinkageValidationError,
        match="duplicate experiment_id",
    ):
        load_paper01_linkage_study(repo)


def test_rejects_missing_linkage_file(tmp_path: Path):
    repo = _copy_materialization(tmp_path)
    missing = (
        repo
        / "studies"
        / "paper01_benchmark"
        / "execution_evidence"
        / "ustc_gan_b2000_seed42.yaml"
    )
    missing.unlink()

    with pytest.raises(
        Paper01LinkageValidationError,
        match="Missing YAML file",
    ):
        load_paper01_linkage_study(repo)


def test_rejects_unindexed_extra_yaml(tmp_path: Path):
    repo = _copy_materialization(tmp_path)
    linkage_dir = (
        repo
        / "studies"
        / "paper01_benchmark"
        / "execution_evidence"
    )

    source = linkage_dir / "ustc_gan_b2000_seed42.yaml"
    shutil.copy2(source, linkage_dir / "unexpected_extra.yaml")

    with pytest.raises(
        Paper01LinkageValidationError,
        match="index/materialization file-set mismatch",
    ):
        load_paper01_linkage_study(repo)


def test_rejects_index_accepted_evaluator_job_mismatch(tmp_path: Path):
    repo = _copy_materialization(tmp_path)
    index_path = (
        repo
        / "studies"
        / "paper01_benchmark"
        / "execution_evidence"
        / "index.yaml"
    )

    index = _load_yaml(index_path)
    index["entries"][0]["accepted_evaluator_job_id"] = "wrong"
    _write_yaml(index_path, index)

    with pytest.raises(
        Paper01LinkageValidationError,
        match="accepted evaluator job ID mismatch",
    ):
        load_paper01_linkage_study(repo)


def test_rejects_index_artifact_status_mismatch(tmp_path: Path):
    repo = _copy_materialization(tmp_path)
    index_path = (
        repo
        / "studies"
        / "paper01_benchmark"
        / "execution_evidence"
        / "index.yaml"
    )

    index = _load_yaml(index_path)
    index["entries"][0]["artifact_production_status"] = "verified"
    _write_yaml(index_path, index)

    with pytest.raises(
        Paper01LinkageValidationError,
        match="artifact production status mismatch",
    ):
        load_paper01_linkage_study(repo)


def test_rejects_filename_experiment_id_mismatch(tmp_path: Path):
    repo = _copy_materialization(tmp_path)
    linkage_dir = (
        repo
        / "studies"
        / "paper01_benchmark"
        / "execution_evidence"
    )
    index_path = linkage_dir / "index.yaml"

    index = _load_yaml(index_path)
    entry = index["entries"][0]

    original = linkage_dir / entry["linkage_file"]
    renamed = linkage_dir / "wrong_name.yaml"
    original.rename(renamed)

    entry["linkage_file"] = "wrong_name.yaml"
    _write_yaml(index_path, index)

    with pytest.raises(
        Paper01LinkageValidationError,
        match="filename does not match experiment_id",
    ):
        load_paper01_linkage_study(repo)
