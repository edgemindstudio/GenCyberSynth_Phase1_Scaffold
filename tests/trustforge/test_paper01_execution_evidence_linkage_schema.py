from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPO_ROOT
    / "manifests"
    / "schemas"
    / "paper01_execution_evidence_linkage.schema.json"
)


def _load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _representative_record() -> dict:
    return {
        "schema_version": "trustforge.paper01.execution_evidence_linkage.v1",
        "experiment": {
            "experiment_id": "cicmaldroid2020_gaussianmixture_b2000_seed42",
            "dataset": "cicmaldroid2020",
            "family": "gaussianmixture",
            "seed": 42,
            "budget_per_class": 2000,
            "historical_run_id": "gaussianmixture_s42",
        },
        "authoritative_result": {
            "status": "verified",
            "table_path": (
                "/home/bruno.fonkeng/gencys/artifacts_paper1_cicmaldroid/"
                "phase1_scores_dedup.csv"
            ),
            "table_sha256": "a" * 64,
            "row_key": {"run_id": "gaussianmixture_s42"},
            "accepted_metric_match": {
                "status": "exact",
                "summary_count": 1,
            },
        },
        "accepted_evaluation": {
            "status": "verified",
            "execution": {
                "job_id": "246862",
                "task_id": "15",
                "host": "talon35",
                "config_sha1": "0d8e51e3e3a54b2816ac72bed5ee001973498cbb",
                "git_commit": "a9d6317a34e63df1e4ee445b618c29169e7a26e4",
                "do_train": False,
                "do_synth": False,
                "do_eval": True,
                "stages": ["eval"],
                "log_path": (
                    "papers/paper1_phase1_benchmark/logs/slurm_raw/"
                    "slurm-paper1.246846_15.out"
                ),
            },
            "accepted_summary": {
                "path": (
                    "/home/bruno.fonkeng/gencys/artifacts_paper1_cicmaldroid/"
                    "gaussianmixture/summaries/summary_20260410_091359.json"
                ),
                "authority": "accepted_timestamped_summary",
                "metric_match": "exact",
                "config_sha1_match_to_execution": True,
            },
        },
        "artifact_production": {
            "status": "strong_candidate",
            "executions": [
                {
                    "job_id": "246427",
                    "task_id": "15",
                    "host": "talon35",
                    "config_sha1": "5d601b08f992905c7764f13d2e8c811550461eda",
                    "git_commit": "a9d6317a34e63df1e4ee445b618c29169e7a26e4",
                    "do_train": True,
                    "do_synth": True,
                    "do_eval": True,
                    "stages": ["eval", "synth", "train"],
                    "log_path": (
                        "papers/paper1_phase1_benchmark/logs/slurm_raw/"
                        "slurm-paper1.246407_15.out"
                    ),
                    "evidence_basis": [
                        "synthesis_stage_observed",
                        "surviving_artifact_timestamps_align",
                    ],
                }
            ],
            "lineage_claim": {
                "level": "strong_candidate",
                "proven": False,
                "reason": (
                    "Synthesis is explicit and timestamps align, but timestamp "
                    "alignment alone does not prove artifact lineage."
                ),
            },
        },
        "historical_executions": [
            {
                "job_id": "246364",
                "task_id": "15",
                "host": "talon35",
                "role": ["historical_execution"],
                "config_sha1": "eae6301fea41a1a5fd9b8fe410a1145ae5a12316",
                "git_commit": "a9d6317a34e63df1e4ee445b618c29169e7a26e4",
                "do_train": True,
                "do_synth": True,
                "do_eval": True,
                "stages": ["eval", "synth", "train"],
                "log_path": (
                    "papers/paper1_phase1_benchmark/logs/slurm_raw/"
                    "slurm-paper1.246348_15.out"
                ),
            },
            {
                "job_id": "246427",
                "task_id": "15",
                "host": "talon35",
                "role": [
                    "historical_execution",
                    "artifact_producer_candidate",
                ],
                "config_sha1": "5d601b08f992905c7764f13d2e8c811550461eda",
                "git_commit": "a9d6317a34e63df1e4ee445b618c29169e7a26e4",
                "do_train": True,
                "do_synth": True,
                "do_eval": True,
                "stages": ["eval", "synth", "train"],
                "log_path": (
                    "papers/paper1_phase1_benchmark/logs/slurm_raw/"
                    "slurm-paper1.246407_15.out"
                ),
            },
            {
                "job_id": "246862",
                "task_id": "15",
                "host": "talon35",
                "role": [
                    "historical_execution",
                    "accepted_evaluator",
                ],
                "config_sha1": "0d8e51e3e3a54b2816ac72bed5ee001973498cbb",
                "git_commit": "a9d6317a34e63df1e4ee445b618c29169e7a26e4",
                "do_train": False,
                "do_synth": False,
                "do_eval": True,
                "stages": ["eval"],
                "log_path": (
                    "papers/paper1_phase1_benchmark/logs/slurm_raw/"
                    "slurm-paper1.246846_15.out"
                ),
            },
        ],
        "manifest_evidence": {
            "seed_specific": {
                "status": "absent",
                "path": None,
                "authority": "none",
            },
            "shared_root": {
                "status": "present",
                "path": (
                    "/home/bruno.fonkeng/gencys/artifacts_paper1_cicmaldroid/"
                    "gaussianmixture/synthetic/manifest.json"
                ),
                "authority": "compatibility_evidence",
            },
        },
        "synthetic_evidence": {
            "status": "present",
            "file_count": 10000,
            "layout": "class_then_seed",
            "layout_counts": {"class_then_seed": 10000},
        },
        "checkpoint_evidence": {
            "status": "present",
            "file_count": 6,
        },
        "non_authoritative_aliases": {
            "latest_summary": {
                "present": False,
                "authority": "none",
                "authoritative": False,
                "reason": "compatibility_alias",
            }
        },
        "provenance_assertions": {
            "accepted_summary_to_evaluation_execution": {
                "status": "verified",
                "basis": [
                    "matching_config_sha1",
                    "execution_log_reports_summary_path",
                ],
            },
            "accepted_summary_to_authoritative_result": {
                "status": "verified",
                "basis": ["exact_accepted_metric_match"],
            },
            "artifact_execution_to_surviving_artifacts": {
                "status": "strong_candidate",
                "basis": [
                    "synthesis_stage_observed",
                    "timestamp_alignment",
                ],
                "limitations": [
                    "timestamp_alignment_is_not_execution_proof"
                ],
            },
        },
        "source_inventory": {
            "audit_version": "M6.4.1-v3.2",
            "inventory_path": (
                "studies/paper01_benchmark/audits/m6_4_1/"
                "paper01_execution_evidence_inventory.json"
            ),
        },
    }


def _validator():
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft202012Validator(_load_schema())


def test_schema_file_exists_and_parses():
    assert SCHEMA_PATH.is_file()
    schema = _load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"


def test_schema_version_is_fixed():
    schema = _load_schema()
    assert (
        schema["properties"]["schema_version"]["const"]
        == "trustforge.paper01.execution_evidence_linkage.v1"
    )


def test_top_level_linkage_roles_are_required():
    required = set(_load_schema()["required"])
    assert {
        "experiment",
        "authoritative_result",
        "accepted_evaluation",
        "artifact_production",
        "historical_executions",
        "manifest_evidence",
        "synthetic_evidence",
        "checkpoint_evidence",
        "non_authoritative_aliases",
        "provenance_assertions",
        "source_inventory",
    } <= required


def test_latest_json_cannot_be_accepted_summary():
    schema = _load_schema()
    path_schema = (
        schema["properties"]["accepted_evaluation"]
        ["properties"]["accepted_summary"]
        ["properties"]["path"]
    )
    assert path_schema["not"]["pattern"] == r"(^|/)latest\.json$"

    latest_alias = (
        schema["properties"]["non_authoritative_aliases"]
        ["properties"]["latest_summary"]["properties"]
    )
    assert latest_alias["authority"]["const"] == "none"
    assert latest_alias["authoritative"]["const"] is False
    assert latest_alias["reason"]["const"] == "compatibility_alias"


def test_authoritative_result_requires_exactly_one_matching_summary():
    accepted_match = (
        _load_schema()["properties"]["authoritative_result"]
        ["properties"]["accepted_metric_match"]["properties"]
    )
    assert accepted_match["status"]["const"] == "exact"
    assert accepted_match["summary_count"]["const"] == 1


def test_accepted_evaluator_requires_eval_true():
    execution = _load_schema()["$defs"]["accepted_evaluation_execution"]
    assert execution["properties"]["do_eval"]["const"] is True


def test_generic_execution_does_not_require_eval_true():
    execution = _load_schema()["$defs"]["execution"]
    assert execution["properties"]["do_eval"] == {"type": "boolean"}


def test_artifact_execution_has_evidence_basis_without_allof_extension_bug():
    artifact = _load_schema()["$defs"]["artifact_execution"]
    assert "allOf" not in artifact
    assert artifact["additionalProperties"] is False
    assert "evidence_basis" in artifact["required"]
    assert "evidence_basis" in artifact["properties"]


def test_artifact_production_supports_uncertainty():
    statuses = set(
        _load_schema()["properties"]["artifact_production"]
        ["properties"]["status"]["enum"]
    )
    assert statuses == {
        "verified",
        "strong_candidate",
        "observed",
        "unresolved",
    }


def test_historical_execution_roles_remain_distinct():
    roles = set(
        _load_schema()["$defs"]["historical_execution"]
        ["properties"]["role"]["items"]["enum"]
    )
    assert roles == {
        "historical_execution",
        "artifact_producer",
        "artifact_producer_candidate",
        "accepted_evaluator",
    }


def test_manifest_schema_allows_absent_seed_manifest():
    statuses = set(
        _load_schema()["$defs"]["manifest_entry"]
        ["properties"]["status"]["enum"]
    )
    assert statuses == {"present", "absent", "unresolved"}


def test_hash_semantics_remain_distinct():
    schema = _load_schema()
    table_hash = (
        schema["properties"]["authoritative_result"]
        ["properties"]["table_sha256"]
    )
    config_hash = schema["$defs"]["execution"]["properties"]["config_sha1"]
    git_commit = schema["$defs"]["execution"]["properties"]["git_commit"]

    assert table_hash["pattern"] == r"^[0-9a-f]{64}$"
    assert config_hash["pattern"] == r"^[0-9a-f]{40}$"
    assert git_commit["pattern"] == r"^[0-9a-f]{40}$"


def test_schema_binds_to_m6_4_1_inventory_version():
    source = _load_schema()["properties"]["source_inventory"]["properties"]
    assert source["audit_version"]["const"] == "M6.4.1-v3.2"
    assert source["inventory_path"]["const"] == (
        "studies/paper01_benchmark/audits/m6_4_1/"
        "paper01_execution_evidence_inventory.json"
    )


def test_representative_cic_gmm_record_validates():
    validator = _validator()
    errors = sorted(
        validator.iter_errors(_representative_record()),
        key=lambda error: list(error.path),
    )
    assert errors == [], "\n".join(
        f"{'/'.join(map(str, error.path))}: {error.message}"
        for error in errors
    )


def test_artifact_execution_may_be_synth_only():
    validator = _validator()
    record = _representative_record()
    artifact = record["artifact_production"]["executions"][0]
    artifact["do_eval"] = False
    artifact["stages"] = ["synth"]

    errors = list(validator.iter_errors(record))
    assert errors == []


def test_latest_json_is_rejected_as_accepted_summary():
    validator = _validator()
    record = deepcopy(_representative_record())
    record["accepted_evaluation"]["accepted_summary"]["path"] = (
        "/historical/summaries/latest.json"
    )

    errors = list(validator.iter_errors(record))
    assert errors
