"""
Canonical Paper 1 execution-evidence loading and validation.

M6.4.2C provides a reusable TrustForge consumption interface for the
repository-owned canonical linkage records produced by M6.4.2B.

This module does not inspect, modify, or repair historical Paper 1 artifacts.
It validates only the canonical repository-owned linkage materialization.

Permanent distinctions preserved:
    SCIENTIFIC EXPERIMENT IDENTITY != HISTORICAL EXECUTION IDENTITY
    ARTIFACT PRODUCER != ACCEPTED EVALUATOR != AUTHORITATIVE RESULT ROW
    OBSERVED EVIDENCE != PROVEN LINEAGE
    latest.json != AUTHORITATIVE EVIDENCE
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


SCHEMA_VERSION = "trustforge.paper01.execution_evidence_linkage.v1"
INDEX_VERSION = "1.0"
STUDY_ID = "paper01_benchmark"
SOURCE_AUDIT_VERSION = "M6.4.1-v3.2"

DEFAULT_LINKAGE_DIR = Path("studies/paper01_benchmark/execution_evidence")
DEFAULT_SCHEMA_PATH = Path(
    "manifests/schemas/paper01_execution_evidence_linkage.schema.json"
)


class Paper01LinkageValidationError(RuntimeError):
    """Raised when canonical Paper 1 linkage evidence is invalid."""


@dataclass(frozen=True)
class Paper01LinkageRecord:
    """One validated canonical linkage record."""

    experiment_id: str
    path: Path
    data: Mapping[str, Any]


@dataclass(frozen=True)
class Paper01LinkageStudy:
    """Validated canonical Paper 1 linkage materialization."""

    index_path: Path
    index: Mapping[str, Any]
    records: tuple[Paper01LinkageRecord, ...]

    def by_experiment_id(self) -> dict[str, Paper01LinkageRecord]:
        return {
            record.experiment_id: record
            for record in self.records
        }

    def get(self, experiment_id: str) -> Paper01LinkageRecord:
        try:
            return self.by_experiment_id()[experiment_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Paper 1 experiment_id: {experiment_id}"
            ) from exc


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise Paper01LinkageValidationError(
            f"Missing YAML file: {path}"
        ) from exc

    if not isinstance(value, dict):
        raise Paper01LinkageValidationError(
            f"Expected YAML mapping at root: {path}"
        )
    return value


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise Paper01LinkageValidationError(
            f"Missing JSON file: {path}"
        ) from exc

    if not isinstance(value, dict):
        raise Paper01LinkageValidationError(
            f"Expected JSON mapping at root: {path}"
        )
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    """Return the semantic SHA256 used by the canonical linkage index."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_schema(
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise Paper01LinkageValidationError(
            "jsonschema is required to validate Paper 1 linkage records"
        ) from exc

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(record),
        key=lambda error: list(error.path),
    )

    if errors:
        details = "\n".join(
            f"{'/'.join(map(str, error.path))}: {error.message}"
            for error in errors
        )
        raise Paper01LinkageValidationError(
            f"Schema validation failed for {path}:\n{details}"
        )


def _validate_record_semantics(
    record: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    experiment = record.get("experiment")
    if not isinstance(experiment, Mapping):
        raise Paper01LinkageValidationError(
            f"{path}: experiment must be a mapping"
        )

    experiment_id = experiment.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise Paper01LinkageValidationError(
            f"{path}: experiment.experiment_id is missing"
        )

    accepted_evaluation = record.get("accepted_evaluation")
    if not isinstance(accepted_evaluation, Mapping):
        raise Paper01LinkageValidationError(
            f"{path}: accepted_evaluation must be a mapping"
        )

    accepted_summary = accepted_evaluation.get("accepted_summary")
    if not isinstance(accepted_summary, Mapping):
        raise Paper01LinkageValidationError(
            f"{path}: accepted_summary must be a mapping"
        )

    accepted_summary_path = Path(str(accepted_summary.get("path", "")))
    if accepted_summary_path.name == "latest.json":
        raise Paper01LinkageValidationError(
            f"{path}: latest.json cannot be authoritative evidence"
        )

    accepted_execution = accepted_evaluation.get("execution")
    if not isinstance(accepted_execution, Mapping):
        raise Paper01LinkageValidationError(
            f"{path}: accepted evaluation execution must be a mapping"
        )

    if accepted_execution.get("do_eval") is not True:
        raise Paper01LinkageValidationError(
            f"{path}: accepted evaluator must have do_eval=true"
        )

    historical_executions = record.get("historical_executions")
    if not isinstance(historical_executions, list):
        raise Paper01LinkageValidationError(
            f"{path}: historical_executions must be a list"
        )

    accepted_roles: list[Mapping[str, Any]] = []
    for execution in historical_executions:
        if not isinstance(execution, Mapping):
            raise Paper01LinkageValidationError(
                f"{path}: historical execution must be a mapping"
            )
        if "accepted_evaluator" in execution.get("role", []):
            accepted_roles.append(execution)

    if len(accepted_roles) != 1:
        raise Paper01LinkageValidationError(
            f"{path}: expected exactly one accepted_evaluator role, "
            f"found {len(accepted_roles)}"
        )

    role_execution = accepted_roles[0]
    for field in ("job_id", "task_id", "config_sha1", "git_commit", "log_path"):
        if role_execution.get(field) != accepted_execution.get(field):
            raise Paper01LinkageValidationError(
                f"{path}: accepted evaluator mismatch for {field}"
            )

    authoritative_result = record.get("authoritative_result")
    if not isinstance(authoritative_result, Mapping):
        raise Paper01LinkageValidationError(
            f"{path}: authoritative_result must be a mapping"
        )

    accepted_metric_match = authoritative_result.get(
        "accepted_metric_match"
    )
    if not isinstance(accepted_metric_match, Mapping):
        raise Paper01LinkageValidationError(
            f"{path}: accepted_metric_match must be a mapping"
        )

    if accepted_metric_match.get("status") != "exact":
        raise Paper01LinkageValidationError(
            f"{path}: accepted metric match must be exact"
        )

    if accepted_metric_match.get("summary_count") != 1:
        raise Paper01LinkageValidationError(
            f"{path}: accepted summary_count must be 1"
        )

    artifact_production = record.get("artifact_production")
    if not isinstance(artifact_production, Mapping):
        raise Paper01LinkageValidationError(
            f"{path}: artifact_production must be a mapping"
        )

    lineage_claim = artifact_production.get("lineage_claim")
    if not isinstance(lineage_claim, Mapping):
        raise Paper01LinkageValidationError(
            f"{path}: artifact lineage_claim must be a mapping"
        )

    if (
        artifact_production.get("status") != "verified"
        and lineage_claim.get("proven") is True
    ):
        raise Paper01LinkageValidationError(
            f"{path}: non-verified artifact lineage cannot be proven=true"
        )

    source_inventory = record.get("source_inventory")
    if not isinstance(source_inventory, Mapping):
        raise Paper01LinkageValidationError(
            f"{path}: source_inventory must be a mapping"
        )

    if source_inventory.get("audit_version") != SOURCE_AUDIT_VERSION:
        raise Paper01LinkageValidationError(
            f"{path}: unexpected source audit version "
            f"{source_inventory.get('audit_version')!r}"
        )


def _validate_index_header(
    index: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    expected = {
        "index_version": INDEX_VERSION,
        "study_id": STUDY_ID,
        "schema_version": SCHEMA_VERSION,
        "source_audit_version": SOURCE_AUDIT_VERSION,
        "record_count": 42,
    }

    for field, expected_value in expected.items():
        actual = index.get(field)
        if actual != expected_value:
            raise Paper01LinkageValidationError(
                f"{path}: {field}={actual!r}, expected {expected_value!r}"
            )

    entries = index.get("entries")
    if not isinstance(entries, list):
        raise Paper01LinkageValidationError(
            f"{path}: entries must be a list"
        )

    if len(entries) != 42:
        raise Paper01LinkageValidationError(
            f"{path}: expected 42 index entries, found {len(entries)}"
        )


def _validate_index_entries(
    index: Mapping[str, Any],
    *,
    linkage_dir: Path,
    schema: Mapping[str, Any],
) -> tuple[Paper01LinkageRecord, ...]:
    entries = index["entries"]
    experiment_ids: set[str] = set()
    linkage_files: set[str] = set()
    records: list[Paper01LinkageRecord] = []

    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise Paper01LinkageValidationError(
                f"index entry {position} must be a mapping"
            )

        experiment_id = entry.get("experiment_id")
        linkage_file = entry.get("linkage_file")
        expected_hash = entry.get("semantic_sha256")

        if not isinstance(experiment_id, str) or not experiment_id:
            raise Paper01LinkageValidationError(
                f"index entry {position}: missing experiment_id"
            )
        if experiment_id in experiment_ids:
            raise Paper01LinkageValidationError(
                f"duplicate experiment_id in index: {experiment_id}"
            )
        experiment_ids.add(experiment_id)

        if not isinstance(linkage_file, str) or not linkage_file:
            raise Paper01LinkageValidationError(
                f"{experiment_id}: missing linkage_file"
            )
        if linkage_file in linkage_files:
            raise Paper01LinkageValidationError(
                f"duplicate linkage_file in index: {linkage_file}"
            )
        linkage_files.add(linkage_file)

        if Path(linkage_file).name != linkage_file:
            raise Paper01LinkageValidationError(
                f"{experiment_id}: linkage_file must be a basename"
            )
        if not linkage_file.endswith(".yaml"):
            raise Paper01LinkageValidationError(
                f"{experiment_id}: linkage_file must end in .yaml"
            )

        record_path = linkage_dir / linkage_file
        record = _load_yaml_mapping(record_path)

        _validate_schema(record, schema, path=record_path)
        _validate_record_semantics(record, path=record_path)

        record_experiment_id = record["experiment"]["experiment_id"]
        if record_experiment_id != experiment_id:
            raise Paper01LinkageValidationError(
                f"{record_path}: experiment_id={record_experiment_id!r} "
                f"does not match index experiment_id={experiment_id!r}"
            )

        if record_path.name != f"{experiment_id}.yaml":
            raise Paper01LinkageValidationError(
                f"{record_path}: filename does not match experiment_id"
            )

        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_hash)
        ):
            raise Paper01LinkageValidationError(
                f"{experiment_id}: semantic_sha256 must be 64 lowercase hex characters"
            )

        actual_hash = semantic_sha256(record)
        if actual_hash != expected_hash:
            raise Paper01LinkageValidationError(
                f"{experiment_id}: semantic SHA256 mismatch; "
                f"index={expected_hash}, actual={actual_hash}"
            )

        accepted_job_id = (
            record["accepted_evaluation"]["execution"]["job_id"]
        )
        if entry.get("accepted_evaluator_job_id") != accepted_job_id:
            raise Paper01LinkageValidationError(
                f"{experiment_id}: accepted evaluator job ID mismatch"
            )

        artifact_status = record["artifact_production"]["status"]
        if entry.get("artifact_production_status") != artifact_status:
            raise Paper01LinkageValidationError(
                f"{experiment_id}: artifact production status mismatch"
            )

        records.append(
            Paper01LinkageRecord(
                experiment_id=experiment_id,
                path=record_path,
                data=record,
            )
        )

    materialized_files = {
        path.name
        for path in linkage_dir.glob("*.yaml")
        if path.name != "index.yaml"
    }

    if materialized_files != linkage_files:
        unindexed = sorted(materialized_files - linkage_files)
        missing = sorted(linkage_files - materialized_files)
        raise Paper01LinkageValidationError(
            "index/materialization file-set mismatch: "
            f"unindexed={unindexed}, missing={missing}"
        )

    return tuple(records)


def load_paper01_linkage_study(
    repo_root: Path | str,
    *,
    linkage_dir: Path | str = DEFAULT_LINKAGE_DIR,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> Paper01LinkageStudy:
    """
    Load and fully validate the canonical Paper 1 execution-evidence set.

    Validation covers:
    - index/study identity and counts;
    - exactly 42 unique scientific experiment IDs;
    - JSON-Schema conformance for every record;
    - filename and record/index identity consistency;
    - semantic SHA256 verification;
    - accepted evaluator consistency;
    - latest.json authority safeguards;
    - index/record accepted evaluator and artifact-status agreement;
    - exact index/materialized file-set agreement.

    No historical artifact file is opened or modified.
    """
    root = Path(repo_root).resolve()

    linkage_root = Path(linkage_dir)
    if not linkage_root.is_absolute():
        linkage_root = root / linkage_root
    linkage_root = linkage_root.resolve()

    schema_file = Path(schema_path)
    if not schema_file.is_absolute():
        schema_file = root / schema_file
    schema_file = schema_file.resolve()

    index_path = linkage_root / "index.yaml"
    index = _load_yaml_mapping(index_path)
    schema = _load_json_mapping(schema_file)

    _validate_index_header(index, path=index_path)
    records = _validate_index_entries(
        index,
        linkage_dir=linkage_root,
        schema=schema,
    )

    return Paper01LinkageStudy(
        index_path=index_path,
        index=index,
        records=records,
    )


def iter_paper01_linkage_records(
    repo_root: Path | str,
) -> Iterable[Paper01LinkageRecord]:
    """Yield validated canonical Paper 1 linkage records in index order."""
    study = load_paper01_linkage_study(repo_root)
    yield from study.records
