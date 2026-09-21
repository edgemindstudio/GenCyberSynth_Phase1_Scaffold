from __future__ import annotations

import json
from pathlib import Path


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
    schema = _load_schema()
    required = set(schema["required"])
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
        ["properties"]["latest_summary"]
        ["properties"]
    )
    assert latest_alias["authority"]["const"] == "none"
    assert latest_alias["authoritative"]["const"] is False
    assert latest_alias["reason"]["const"] == "compatibility_alias"


def test_authoritative_result_requires_exactly_one_matching_summary():
    schema = _load_schema()
    accepted_match = (
        schema["properties"]["authoritative_result"]
        ["properties"]["accepted_metric_match"]
        ["properties"]
    )
    assert accepted_match["status"]["const"] == "exact"
    assert accepted_match["summary_count"]["const"] == 1


def test_accepted_evaluator_requires_eval_stage_identity():
    schema = _load_schema()
    execution = schema["$defs"]["execution"]
    assert execution["properties"]["do_eval"]["const"] is True
    assert {
        "job_id",
        "task_id",
        "host",
        "config_sha1",
        "git_commit",
        "do_train",
        "do_synth",
        "do_eval",
        "stages",
        "log_path",
    } <= set(execution["required"])


def test_artifact_production_supports_uncertainty():
    schema = _load_schema()
    statuses = set(
        schema["properties"]["artifact_production"]
        ["properties"]["status"]["enum"]
    )
    assert statuses == {
        "verified",
        "strong_candidate",
        "observed",
        "unresolved",
    }
    lineage = (
        schema["properties"]["artifact_production"]
        ["properties"]["lineage_claim"]
        ["properties"]
    )
    assert lineage["proven"]["type"] == "boolean"


def test_historical_execution_roles_remain_distinct():
    schema = _load_schema()
    roles = set(
        schema["$defs"]["historical_execution"]
        ["properties"]["role"]["items"]["enum"]
    )
    assert roles == {
        "historical_execution",
        "artifact_producer",
        "artifact_producer_candidate",
        "accepted_evaluator",
    }


def test_manifest_schema_allows_absent_seed_manifest():
    schema = _load_schema()
    statuses = set(
        schema["$defs"]["manifest_entry"]["properties"]["status"]["enum"]
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
    schema = _load_schema()
    source = schema["properties"]["source_inventory"]["properties"]
    assert source["audit_version"]["const"] == "M6.4.1-v3.2"
    assert source["inventory_path"]["const"] == (
        "studies/paper01_benchmark/audits/m6_4_1/"
        "paper01_execution_evidence_inventory.json"
    )
