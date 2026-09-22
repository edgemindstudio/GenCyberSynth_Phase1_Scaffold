#!/usr/bin/env python3
"""
Generate canonical TrustForge Paper 1 execution-evidence linkage records.

M6.4.2B-1 goals:
- consume the frozen Paper 1 experiment index;
- consume the validated M6.4.1 execution-evidence inventory;
- preserve scientific experiment identity separately from historical execution identity;
- identify exactly one accepted evaluator from exact accepted-summary evidence;
- represent artifact-production lineage conservatively;
- never treat latest.json as authoritative evidence;
- never modify historical Paper 1 artifacts;
- produce deterministic repository-owned linkage records.

This generator intentionally does NOT infer provenance from "latest" files or
from timestamp proximity alone.

Permanent distinctions:
    SCIENTIFIC EXPERIMENT IDENTITY != HISTORICAL EXECUTION IDENTITY
    ARTIFACT PRODUCER != ACCEPTED EVALUATOR != AUTHORITATIVE RESULT ROW
    OBSERVED EVIDENCE != PROVEN LINEAGE
    TIMESTAMP ALIGNMENT != EXECUTION PROOF
    latest.json != AUTHORITATIVE EVIDENCE
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA_VERSION = "trustforge.paper01.execution_evidence_linkage.v1"
AUDIT_VERSION = "M6.4.1-v3.2"

DEFAULT_EXPERIMENT_INDEX = Path(
    "studies/paper01_benchmark/experiment_index.yaml"
)
DEFAULT_INVENTORY = Path(
    "studies/paper01_benchmark/audits/m6_4_1/"
    "paper01_execution_evidence_inventory.json"
)
DEFAULT_SCHEMA = Path(
    "manifests/schemas/paper01_execution_evidence_linkage.schema.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "studies/paper01_benchmark/execution_evidence"
)

# Authoritative Paper 1 deduplicated result-table identities established in
# the prior Paper 1 evidence-verification stages.
AUTHORITATIVE_RESULT_TABLE_SHA256 = {
    "ustc_tfc2016": (
        "c3c731f355f4cecfb930d024dd8ee42d97ce952d5abfca73c717af6b2a7a6031"
    ),
    "cicmaldroid2020": (
        "63fc2cd7b2cde81f2b964cc4407c6efdb1c50ee8bd61bfbbd4384d7eff8fc4f4"
    ),
}

INDEX_DATASET_TO_AUDIT_DATASET = {
    "ustc_tfc2016_malware_nhwc": "ustc_tfc2016",
    "cicmaldroid2020_paper1": "cicmaldroid2020",
}

AUDIT_DATASET_TO_OUTPUT_PREFIX = {
    "ustc_tfc2016": "ustc",
    "cicmaldroid2020": "cicmaldroid",
}

TIMESTAMP_ALIGNMENT_SECONDS = 15.0


class LinkageGenerationError(RuntimeError):
    """Raised when linkage generation cannot make an evidence-safe decision."""


@dataclass(frozen=True)
class ExperimentKey:
    dataset: str
    family: str
    seed: int
    historical_run_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Paper 1 execution-evidence linkage records."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="TrustForge repository root.",
    )
    parser.add_argument(
        "--experiment-index",
        type=Path,
        default=DEFAULT_EXPERIMENT_INDEX,
        help="Frozen Paper 1 experiment index.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help="Validated M6.4.1 evidence inventory JSON.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Paper 1 linkage JSON Schema.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Repository-owned linkage output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and validate all records without writing linkage files."
        ),
    )
    parser.add_argument(
        "--validate-schema",
        action="store_true",
        help=(
            "Validate generated records with jsonschema when available."
        ),
    )
    return parser.parse_args()


def resolve_under_repo(repo_root: Path, path: Path) -> Path:
    repo_root = repo_root.resolve()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_location(
    output_dir: Path,
    repo_root: Path,
    historical_roots: Sequence[Path],
) -> None:
    output_dir = output_dir.resolve()
    repo_root = repo_root.resolve()

    if not is_relative_to(output_dir, repo_root):
        raise LinkageGenerationError(
            f"Output directory must be inside repository: {output_dir}"
        )

    for historical_root in historical_roots:
        historical_root = historical_root.resolve()
        if is_relative_to(output_dir, historical_root):
            raise LinkageGenerationError(
                "Output directory must not be inside historical artifact root: "
                f"{historical_root}"
            )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise LinkageGenerationError(
            f"Expected mapping at YAML root: {path}"
        )
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise LinkageGenerationError(
            f"Expected mapping at JSON root: {path}"
        )
    return data


def parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise LinkageGenerationError(
            f"Invalid ISO-8601 timestamp: {value}"
        ) from exc


def seconds_between(
    left: datetime | None,
    right: datetime | None,
) -> float | None:
    if left is None or right is None:
        return None
    return abs((left - right).total_seconds())


def experiment_key_from_index(entry: Mapping[str, Any]) -> ExperimentKey:
    dataset_id = str(entry["dataset_id"])
    try:
        audit_dataset = INDEX_DATASET_TO_AUDIT_DATASET[dataset_id]
    except KeyError as exc:
        raise LinkageGenerationError(
            f"Unrecognized experiment-index dataset_id: {dataset_id}"
        ) from exc

    return ExperimentKey(
        dataset=audit_dataset,
        family=str(entry["model_id"]),
        seed=int(entry["seed"]),
        historical_run_id=str(entry["historical_run_id"]),
    )


def experiment_key_from_inventory(
    entry: Mapping[str, Any],
) -> ExperimentKey:
    return ExperimentKey(
        dataset=str(entry["dataset"]),
        family=str(entry["family"]),
        seed=int(entry["seed"]),
        historical_run_id=str(entry["historical_run_id"]),
    )


def build_inventory_lookup(
    inventory: Mapping[str, Any],
) -> dict[ExperimentKey, dict[str, Any]]:
    raw_experiments = inventory.get("experiments")
    if not isinstance(raw_experiments, list):
        raise LinkageGenerationError(
            "M6.4.1 inventory must contain experiments[]"
        )

    lookup: dict[ExperimentKey, dict[str, Any]] = {}
    for raw in raw_experiments:
        if not isinstance(raw, dict):
            raise LinkageGenerationError(
                "Inventory experiment entry is not a mapping"
            )
        key = experiment_key_from_inventory(raw)
        if key in lookup:
            raise LinkageGenerationError(
                f"Duplicate inventory experiment key: {key}"
            )
        lookup[key] = raw
    return lookup


def index_entries(
    experiment_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_entries = experiment_index.get("entries")
    if not isinstance(raw_entries, list):
        raise LinkageGenerationError(
            "Experiment index must contain entries[]"
        )

    entries: list[dict[str, Any]] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise LinkageGenerationError(
                "Experiment-index entry is not a mapping"
            )
        entries.append(raw)

    expected_count = experiment_index.get("counts", {}).get("experiments")
    if expected_count is not None and len(entries) != int(expected_count):
        raise LinkageGenerationError(
            "Experiment-index entry count does not match declared count: "
            f"{len(entries)} != {expected_count}"
        )
    return entries


def exact_accepted_summary(
    audit_exp: Mapping[str, Any],
) -> dict[str, Any]:
    summaries = audit_exp.get("summaries", [])
    exact = [
        summary
        for summary in summaries
        if summary.get("metric_match_status")
        == "exact_all_accepted_metrics"
    ]

    if len(exact) != 1:
        raise LinkageGenerationError(
            f"{audit_exp['experiment_id']}: expected exactly one "
            f"timestamped accepted summary, found {len(exact)}"
        )

    summary = exact[0]
    if Path(str(summary["path"])).name == "latest.json":
        raise LinkageGenerationError(
            f"{audit_exp['experiment_id']}: latest.json cannot be accepted"
        )
    return summary


def log_lookup_by_path(
    audit_exp: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for log in audit_exp.get("slurm_logs", []):
        path = str(log["path"])
        if path in lookup:
            raise LinkageGenerationError(
                f"{audit_exp['experiment_id']}: duplicate log path {path}"
            )
        lookup[path] = log
    return lookup


def accepted_evaluator(
    audit_exp: Mapping[str, Any],
    accepted_summary: Mapping[str, Any],
) -> dict[str, Any]:
    linked_paths = list(accepted_summary.get("linked_log_paths", []))
    if not linked_paths:
        raise LinkageGenerationError(
            f"{audit_exp['experiment_id']}: accepted summary has no linked log"
        )

    logs_by_path = log_lookup_by_path(audit_exp)

    candidates = []
    for path in linked_paths:
        log = logs_by_path.get(str(path))
        if log is None:
            raise LinkageGenerationError(
                f"{audit_exp['experiment_id']}: accepted summary references "
                f"unknown log {path}"
            )
        if (
            log.get("config_sha1")
            == accepted_summary.get("config_sha1")
            and bool(log.get("do_eval"))
        ):
            candidates.append(log)

    if len(candidates) != 1:
        raise LinkageGenerationError(
            f"{audit_exp['experiment_id']}: expected exactly one accepted "
            f"evaluator linked by config SHA1, found {len(candidates)}"
        )
    return candidates[0]


def artifact_time_window(
    audit_exp: Mapping[str, Any],
) -> tuple[datetime | None, datetime | None]:
    times: list[datetime] = []

    synthetic = audit_exp.get("synthetic", {})
    for key in ("earliest_mtime_utc", "latest_mtime_utc"):
        value = parse_iso8601(synthetic.get(key))
        if value is not None:
            times.append(value)

    for checkpoint in audit_exp.get("checkpoints", []):
        value = parse_iso8601(checkpoint.get("mtime_utc"))
        if value is not None:
            times.append(value)

    if not times:
        return None, None
    return min(times), max(times)


def summary_for_log(
    audit_exp: Mapping[str, Any],
    log: Mapping[str, Any],
) -> dict[str, Any] | None:
    config_sha1 = log.get("config_sha1")
    if not config_sha1:
        return None

    candidates = [
        summary
        for summary in audit_exp.get("summaries", [])
        if summary.get("config_sha1") == config_sha1
    ]

    if len(candidates) > 1:
        raise LinkageGenerationError(
            f"{audit_exp['experiment_id']}: multiple timestamped summaries "
            f"share config SHA1 {config_sha1}"
        )
    return candidates[0] if candidates else None


def artifact_alignment_score(
    audit_exp: Mapping[str, Any],
    log: Mapping[str, Any],
) -> tuple[int, float]:
    """
    Score explicit synthesis executions against surviving artifacts.

    Higher integer score is better. The float is a deterministic temporal
    distance tie-breaker.

    This is used ONLY to identify a strong candidate, never to claim proven
    artifact lineage.
    """
    if not bool(log.get("do_synth")):
        return (-1, float("inf"))

    window_start, window_end = artifact_time_window(audit_exp)
    log_time = parse_iso8601(log.get("mtime_utc"))
    related_summary = summary_for_log(audit_exp, log)
    summary_time = (
        parse_iso8601(related_summary.get("mtime_utc"))
        if related_summary is not None
        else None
    )

    score = 1  # explicit synthesis
    distances: list[float] = []

    for observed_time in (log_time, summary_time):
        if observed_time is None:
            continue
        for boundary in (window_start, window_end):
            distance = seconds_between(observed_time, boundary)
            if distance is not None:
                distances.append(distance)

    minimum_distance = min(distances) if distances else float("inf")

    if minimum_distance <= TIMESTAMP_ALIGNMENT_SECONDS:
        score += 2
    elif minimum_distance <= 300.0:
        score += 1

    return score, minimum_distance


def select_artifact_producer_candidate(
    audit_exp: Mapping[str, Any],
) -> dict[str, Any] | None:
    synth_logs = [
        log
        for log in audit_exp.get("slurm_logs", [])
        if bool(log.get("do_synth"))
    ]

    if not synth_logs:
        return None

    ranked = sorted(
        synth_logs,
        key=lambda log: (
            -artifact_alignment_score(audit_exp, log)[0],
            artifact_alignment_score(audit_exp, log)[1],
            str(log.get("job_id") or ""),
            str(log.get("path") or ""),
        ),
    )

    best = ranked[0]
    best_score, best_distance = artifact_alignment_score(audit_exp, best)

    if best_score < 2:
        # Explicit synthesis exists, but surviving artifact alignment is not
        # strong enough to identify one producer candidate safely.
        return None

    if len(ranked) > 1:
        second_score, second_distance = artifact_alignment_score(
            audit_exp,
            ranked[1],
        )
        if (
            best_score == second_score
            and best_distance == second_distance
        ):
            # Evidence does not distinguish the candidates.
            return None

    return best


def execution_payload(
    log: Mapping[str, Any],
) -> dict[str, Any]:
    config_sha1 = log.get("config_sha1")
    git_commit = log.get("git_commit")
    if not config_sha1 or not git_commit:
        raise LinkageGenerationError(
            f"Execution missing config SHA1 or Git commit: {log.get('path')}"
        )

    return {
        "job_id": (
            str(log["job_id"]) if log.get("job_id") is not None else None
        ),
        "task_id": (
            str(log["task_id"]) if log.get("task_id") is not None else None
        ),
        "host": log.get("host"),
        "config_sha1": str(config_sha1),
        "git_commit": str(git_commit),
        "do_train": bool(log.get("do_train")),
        "do_synth": bool(log.get("do_synth")),
        "do_eval": bool(log.get("do_eval")),
        "stages": sorted(str(stage) for stage in log.get("stages", [])),
        "log_path": str(log["path"]),
    }


def artifact_execution_payload(
    audit_exp: Mapping[str, Any],
    log: Mapping[str, Any],
) -> dict[str, Any]:
    payload = execution_payload(log)
    score, distance = artifact_alignment_score(audit_exp, log)

    evidence_basis = ["synthesis_stage_observed"]
    if distance <= TIMESTAMP_ALIGNMENT_SECONDS:
        evidence_basis.append("surviving_artifact_timestamps_align")
    elif distance <= 300.0:
        evidence_basis.append("surviving_artifact_timestamps_nearby")

    if audit_exp.get("checkpoints"):
        evidence_basis.append("checkpoint_evidence_present")

    payload["evidence_basis"] = evidence_basis
    return payload


def historical_execution_payload(
    log: Mapping[str, Any],
    *,
    accepted_log_path: str,
    artifact_candidate_path: str | None,
) -> dict[str, Any]:
    payload = execution_payload(log)
    roles = ["historical_execution"]

    if str(log["path"]) == accepted_log_path:
        roles.append("accepted_evaluator")
    if (
        artifact_candidate_path is not None
        and str(log["path"]) == artifact_candidate_path
    ):
        roles.append("artifact_producer_candidate")

    return {
        "job_id": payload["job_id"],
        "task_id": payload["task_id"],
        "host": payload["host"],
        "role": roles,
        "config_sha1": payload["config_sha1"],
        "git_commit": payload["git_commit"],
        "do_train": payload["do_train"],
        "do_synth": payload["do_synth"],
        "do_eval": payload["do_eval"],
        "stages": payload["stages"],
        "log_path": payload["log_path"],
    }


def manifest_entry(
    manifests: Sequence[Mapping[str, Any]],
    kind: str,
) -> dict[str, Any]:
    matches = [
        manifest
        for manifest in manifests
        if manifest.get("kind") == kind
    ]

    if not matches:
        return {
            "status": "absent",
            "path": None,
            "authority": "none",
        }

    if len(matches) != 1:
        raise LinkageGenerationError(
            f"Expected at most one {kind}, found {len(matches)}"
        )

    authority = (
        "historical_seed_manifest"
        if kind == "seed_specific_manifest"
        else "compatibility_evidence"
    )
    return {
        "status": "present",
        "path": str(matches[0]["path"]),
        "authority": authority,
    }


def dataset_result_sha256(dataset: str) -> str:
    try:
        return AUTHORITATIVE_RESULT_TABLE_SHA256[dataset]
    except KeyError as exc:
        raise LinkageGenerationError(
            f"No authoritative result-table SHA256 for dataset {dataset}"
        ) from exc


def build_record(
    index_entry: Mapping[str, Any],
    audit_exp: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_summary = exact_accepted_summary(audit_exp)
    accepted_log = accepted_evaluator(audit_exp, accepted_summary)
    artifact_candidate = select_artifact_producer_candidate(audit_exp)

    accepted_row = audit_exp.get("accepted_score_row")
    if not isinstance(accepted_row, dict):
        raise LinkageGenerationError(
            f"{audit_exp['experiment_id']}: missing accepted score row"
        )

    dataset = str(audit_exp["dataset"])
    family = str(audit_exp["family"])
    seed = int(audit_exp["seed"])
    budget = int(audit_exp["budget"])
    run_id = str(audit_exp["historical_run_id"])

    manifests = list(audit_exp.get("manifests", []))
    seed_manifest = manifest_entry(manifests, "seed_specific_manifest")
    shared_manifest = manifest_entry(manifests, "shared_root_manifest")

    artifact_path = (
        str(artifact_candidate["path"])
        if artifact_candidate is not None
        else None
    )

    historical_executions = [
        historical_execution_payload(
            log,
            accepted_log_path=str(accepted_log["path"]),
            artifact_candidate_path=artifact_path,
        )
        for log in sorted(
            audit_exp.get("slurm_logs", []),
            key=lambda item: (
                str(item.get("job_id") or ""),
                str(item.get("path") or ""),
            ),
        )
    ]

    if artifact_candidate is not None:
        artifact_status = "strong_candidate"
        artifact_executions = [
            artifact_execution_payload(
                audit_exp,
                artifact_candidate,
            )
        ]
        lineage_claim = {
            "level": "strong_candidate",
            "proven": False,
            "reason": (
                "Historical execution explicitly performed synthesis and "
                "surviving artifact/checkpoint timing aligns with that "
                "execution wave, but timestamp alignment does not prove "
                "exclusive artifact lineage."
            ),
        }
        artifact_assertion = {
            "status": "strong_candidate",
            "basis": [
                "synthesis_stage_observed",
                "timestamp_alignment",
            ],
            "limitations": [
                "timestamp_alignment_is_not_execution_proof"
            ],
        }
    else:
        artifact_status = "unresolved"
        artifact_executions = []
        lineage_claim = {
            "level": "unresolved",
            "proven": False,
            "reason": (
                "Historical synthesis executions exist or may exist, but "
                "available evidence does not distinguish one surviving "
                "artifact producer strongly enough for canonical linkage."
            ),
        }
        artifact_assertion = {
            "status": "unresolved",
            "basis": [],
            "limitations": [
                "timestamp_alignment_is_not_execution_proof"
            ],
        }

    synthetic = audit_exp.get("synthetic", {})
    synthetic_count = int(synthetic.get("file_count", 0))
    synthetic_status = "present" if synthetic_count > 0 else "absent"

    checkpoint_count = len(audit_exp.get("checkpoints", []))
    checkpoint_status = "present" if checkpoint_count > 0 else "absent"

    latest_aliases = list(audit_exp.get("latest_aliases", []))

    accepted_execution = execution_payload(accepted_log)
    if not accepted_execution["do_eval"]:
        raise LinkageGenerationError(
            f"{audit_exp['experiment_id']}: accepted evaluator is not eval-enabled"
        )

    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            # Preserve the frozen scientific ID from experiment_index.yaml.
            "experiment_id": str(index_entry["experiment_id"]),
            "dataset": dataset,
            "family": family,
            "seed": seed,
            "budget_per_class": budget,
            "historical_run_id": run_id,
        },
        "authoritative_result": {
            "status": "verified",
            "table_path": str(accepted_row["source"]),
            "table_sha256": dataset_result_sha256(dataset),
            "row_key": {
                "run_id": str(accepted_row["run_id"]),
            },
            "accepted_metric_match": {
                "status": "exact",
                "summary_count": 1,
            },
        },
        "accepted_evaluation": {
            "status": "verified",
            "execution": accepted_execution,
            "accepted_summary": {
                "path": str(accepted_summary["path"]),
                "authority": "accepted_timestamped_summary",
                "metric_match": "exact",
                "config_sha1_match_to_execution": (
                    accepted_summary.get("config_sha1")
                    == accepted_log.get("config_sha1")
                ),
            },
        },
        "artifact_production": {
            "status": artifact_status,
            "executions": artifact_executions,
            "lineage_claim": lineage_claim,
        },
        "historical_executions": historical_executions,
        "manifest_evidence": {
            "seed_specific": seed_manifest,
            "shared_root": shared_manifest,
        },
        "synthetic_evidence": {
            "status": synthetic_status,
            "file_count": synthetic_count,
            "layout": str(synthetic.get("layout", "unresolved")),
            "layout_counts": {
                str(key): int(value)
                for key, value in sorted(
                    dict(synthetic.get("layout_counts", {})).items()
                )
            },
        },
        "checkpoint_evidence": {
            "status": checkpoint_status,
            "file_count": checkpoint_count,
        },
        "non_authoritative_aliases": {
            "latest_summary": {
                "present": bool(latest_aliases),
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
                "basis": [
                    "exact_accepted_metric_match",
                ],
            },
            "artifact_execution_to_surviving_artifacts": artifact_assertion,
        },
        "source_inventory": {
            "audit_version": AUDIT_VERSION,
            "inventory_path": str(DEFAULT_INVENTORY),
        },
    }

    return record


def validate_record_semantics(record: Mapping[str, Any]) -> None:
    experiment = record["experiment"]
    accepted = record["accepted_evaluation"]

    accepted_summary_path = Path(
        str(accepted["accepted_summary"]["path"])
    )
    if accepted_summary_path.name == "latest.json":
        raise LinkageGenerationError(
            f"{experiment['experiment_id']}: latest.json is not authoritative"
        )

    if (
        record["authoritative_result"]["accepted_metric_match"]["summary_count"]
        != 1
    ):
        raise LinkageGenerationError(
            f"{experiment['experiment_id']}: accepted summary count != 1"
        )

    accepted_execution = accepted["execution"]
    if not accepted_execution["do_eval"]:
        raise LinkageGenerationError(
            f"{experiment['experiment_id']}: accepted evaluator lacks eval"
        )

    historical = record["historical_executions"]
    accepted_roles = [
        execution
        for execution in historical
        if "accepted_evaluator" in execution["role"]
    ]
    if len(accepted_roles) != 1:
        raise LinkageGenerationError(
            f"{experiment['experiment_id']}: expected one accepted evaluator role"
        )

    if (
        accepted_roles[0]["config_sha1"]
        != accepted_execution["config_sha1"]
    ):
        raise LinkageGenerationError(
            f"{experiment['experiment_id']}: accepted evaluator SHA1 mismatch"
        )

    artifact = record["artifact_production"]
    if artifact["status"] == "verified":
        raise LinkageGenerationError(
            f"{experiment['experiment_id']}: generator must not automatically "
            "promote artifact lineage to verified"
        )

    if artifact["status"] == "strong_candidate":
        if artifact["lineage_claim"]["proven"]:
            raise LinkageGenerationError(
                f"{experiment['experiment_id']}: strong candidate cannot be proven"
            )


def validate_with_jsonschema(
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise LinkageGenerationError(
            "--validate-schema requested but jsonschema is unavailable"
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
        raise LinkageGenerationError(
            f"Generated linkage record failed schema validation:\n{details}"
        )


def canonical_json_bytes(value: Any) -> bytes:
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
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def deterministic_yaml_text(value: Any) -> str:
    # sort_keys=False preserves the deliberately constructed canonical field
    # order while all dynamic maps/lists are already sorted by the generator.
    return yaml.safe_dump(
        value,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=1000,
    )


def linkage_filename(index_entry: Mapping[str, Any]) -> str:
    return f"{index_entry['experiment_id']}.yaml"


def build_all_records(
    experiment_index: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    entries = index_entries(experiment_index)
    inventory_lookup = build_inventory_lookup(inventory)

    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_ids: set[str] = set()

    for entry in entries:
        scientific_id = str(entry["experiment_id"])
        if scientific_id in seen_ids:
            raise LinkageGenerationError(
                f"Duplicate scientific experiment ID: {scientific_id}"
            )
        seen_ids.add(scientific_id)

        key = experiment_key_from_index(entry)
        audit_exp = inventory_lookup.get(key)
        if audit_exp is None:
            raise LinkageGenerationError(
                f"No M6.4.1 evidence entry for frozen experiment {scientific_id} "
                f"using key {key}"
            )

        record = build_record(entry, audit_exp)
        validate_record_semantics(record)
        if schema is not None:
            validate_with_jsonschema(record, schema)

        records.append((entry, record))

    if len(records) != 42:
        raise LinkageGenerationError(
            f"Expected exactly 42 linkage records, built {len(records)}"
        )

    if len(inventory_lookup) != 42:
        raise LinkageGenerationError(
            f"Expected exactly 42 inventory experiments, found "
            f"{len(inventory_lookup)}"
        )

    return records


def build_index(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    entries = []
    for index_entry, record in records:
        experiment_id = str(index_entry["experiment_id"])
        entries.append(
            {
                "experiment_id": experiment_id,
                "linkage_file": f"{experiment_id}.yaml",
                "semantic_sha256": semantic_sha256(record),
                "accepted_evaluator_job_id": (
                    record["accepted_evaluation"]["execution"]["job_id"]
                ),
                "artifact_production_status": (
                    record["artifact_production"]["status"]
                ),
            }
        )

    return {
        "index_version": "1.0",
        "study_id": "paper01_benchmark",
        "schema_version": SCHEMA_VERSION,
        "source_audit_version": AUDIT_VERSION,
        "record_count": len(entries),
        "entries": entries,
    }


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_outputs(
    output_dir: Path,
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_files: set[str] = {"index.yaml"}

    for index_entry, record in records:
        filename = linkage_filename(index_entry)
        expected_files.add(filename)
        write_atomic(
            output_dir / filename,
            deterministic_yaml_text(record),
        )

    linkage_index = build_index(records)
    write_atomic(
        output_dir / "index.yaml",
        deterministic_yaml_text(linkage_index),
    )

    # Remove only stale generator-owned YAML files from the linkage output
    # directory. Never touch anything outside this repository-owned directory.
    for path in sorted(output_dir.glob("*.yaml")):
        if path.name not in expected_files:
            path.unlink()


def historical_roots_from_inventory(
    inventory: Mapping[str, Any],
) -> list[Path]:
    roots: set[Path] = set()

    for exp in inventory.get("experiments", []):
        accepted = exp.get("accepted_score_row", {})
        source = accepted.get("source")
        if source:
            roots.add(Path(str(source)).parent)

        for log in exp.get("slurm_logs", []):
            artifact_root = log.get("artifact_root")
            if artifact_root:
                roots.add(Path(str(artifact_root)))

    return sorted(roots, key=str)


def summarize(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    artifact_statuses: dict[str, int] = {}
    latest_aliases = 0

    for _, record in records:
        status = str(record["artifact_production"]["status"])
        artifact_statuses[status] = artifact_statuses.get(status, 0) + 1

        if record["non_authoritative_aliases"]["latest_summary"]["present"]:
            latest_aliases += 1

    print(f"[M6.4.2B] Linkage records built: {len(records)}")
    print(
        "[M6.4.2B] Artifact-production statuses: "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(artifact_statuses.items())
        )
    )
    print(f"[M6.4.2B] latest.json aliases observed: {latest_aliases}")
    print("[M6.4.2B] Accepted evaluators: exactly one per experiment")
    print("[M6.4.2B] Historical artifacts modified: NO")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    experiment_index_path = resolve_under_repo(
        repo_root,
        args.experiment_index,
    )
    inventory_path = resolve_under_repo(
        repo_root,
        args.inventory,
    )
    schema_path = resolve_under_repo(
        repo_root,
        args.schema,
    )
    output_dir = resolve_under_repo(
        repo_root,
        args.output_dir,
    )

    experiment_index = load_yaml(experiment_index_path)
    inventory = load_json(inventory_path)

    schema = None
    if args.validate_schema:
        schema = load_json(schema_path)

    historical_roots = historical_roots_from_inventory(inventory)
    validate_output_location(
        output_dir,
        repo_root,
        historical_roots,
    )

    records = build_all_records(
        experiment_index,
        inventory,
        schema=schema,
    )

    summarize(records)

    if args.dry_run:
        print("[M6.4.2B] Dry run: no linkage files written.")
        return 0

    write_outputs(output_dir, records)
    print(f"[M6.4.2B] Output directory: {output_dir}")
    print("[M6.4.2B] Canonical linkage files written: YES")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LinkageGenerationError as exc:
        print(f"[M6.4.2B] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
