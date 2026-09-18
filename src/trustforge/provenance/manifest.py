"""
TrustForge-native execution manifest construction.

A TrustForge manifest records one concrete execution attempt.

This module is intended for new TrustForge-native executions. Historical
studies must preserve the provenance that was actually recorded at execution
time and should use reconstruction metadata rather than silently replacing
historical provenance with modern values.

Native manifests use:

- SHA256 for resolved configuration identity;
- full Git commit identity;
- explicit Git dirty/clean state;
- concrete runtime and filesystem paths;
- stable study, experiment, and execution identifiers.

The manifest describes execution reality. It does not by itself declare that
an execution is scientifically accepted evidence.
"""

from __future__ import annotations

import json
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from trustforge.provenance.git import (
    GitState,
    capture_git_state,
)
from trustforge.provenance.hashing import (
    sha256_mapping,
)
from trustforge.storage import (
    validate_storage_identifier,
)


_ALLOWED_STATUSES = {
    "planned",
    "running",
    "completed",
    "failed",
    "cancelled",
    "invalidated",
}


class ManifestConstructionError(ValueError):
    """Raised when a TrustForge-native manifest cannot be constructed."""


class ManifestWriteError(RuntimeError):
    """Raised when a TrustForge manifest cannot be written safely."""


def _normalize_path(path: Path | str) -> Path:
    """Return an absolute normalized path without requiring existence."""

    return Path(path).expanduser().resolve(strict=False)


def _iso8601(value: datetime) -> str:
    """
    Serialize a timezone-aware datetime using ISO 8601.

    Naive datetimes are rejected because execution timestamps must carry
    timezone information.
    """

    if not isinstance(value, datetime):
        raise ManifestConstructionError(
            "Manifest timestamp must be a datetime."
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ManifestConstructionError(
            "Manifest timestamp must include timezone information."
        )

    return value.isoformat()


def _validate_status(status: str) -> str:
    """Validate an execution status against the TrustForge manifest contract."""

    if status not in _ALLOWED_STATUSES:
        allowed = ", ".join(sorted(_ALLOWED_STATUSES))

        raise ManifestConstructionError(
            f"Unsupported manifest status {status!r}. "
            f"Allowed values: {allowed}."
        )

    return status


def _validate_repository(repository: str) -> str:
    """Validate the logical repository identifier."""

    if not isinstance(repository, str) or not repository.strip():
        raise ManifestConstructionError(
            "repository must be a non-empty string."
        )

    if repository != repository.strip():
        raise ManifestConstructionError(
            "repository must not contain leading or trailing whitespace."
        )

    return repository


def build_native_manifest(
    *,
    study_id: str,
    experiment_id: str,
    execution_id: str,
    repository: str,
    repo_root: Path | str,
    resolved_config: Mapping[str, Any],
    dataset_id: str,
    dataset_path: Path | str,
    artifact_root: Path | str,
    status: str = "running",
    config_source: Path | str | None = None,
    seed: int | None = None,
    started_at: datetime | None = None,
    machine_role: str | None = None,
    scheduler: Mapping[str, Any] | None = None,
    git_state: GitState | None = None,
) -> dict[str, Any]:
    """
    Construct a TrustForge-native execution manifest.

    Parameters
    ----------
    study_id:
        Stable TrustForge study identifier.

    experiment_id:
        Stable scientific experiment identifier.

    execution_id:
        Unique concrete execution identifier.

    repository:
        Logical repository identity, for example
        ``edgemindstudio/trustforge``.

    repo_root:
        Concrete repository checkout used for the execution.

    resolved_config:
        Fully resolved scientific configuration. Its canonical JSON
        representation is hashed with SHA256.

    dataset_id:
        Stable logical dataset identity.

    dataset_path:
        Concrete dataset path used by this execution.

    artifact_root:
        Concrete output root assigned to this execution.

    status:
        Current execution status.

    config_source:
        Optional source configuration path.

    seed:
        Optional execution seed.

    started_at:
        Optional timezone-aware execution start time.

    machine_role:
        Optional stable machine role such as ``talon_hpc``.

    scheduler:
        Optional scheduler provenance mapping.

    git_state:
        Optional already-captured Git state. If omitted, repository state is
        captured from ``repo_root``.

    Returns
    -------
    dict
        Manifest mapping conforming to the required TrustForge native manifest
        contract.
    """

    study_id = validate_storage_identifier(
        study_id,
        field_name="study_id",
    )

    experiment_id = validate_storage_identifier(
        experiment_id,
        field_name="experiment_id",
    )

    execution_id = validate_storage_identifier(
        execution_id,
        field_name="execution_id",
    )

    dataset_id = validate_storage_identifier(
        dataset_id,
        field_name="dataset_id",
    )

    repository = _validate_repository(repository)
    status = _validate_status(status)

    if not isinstance(resolved_config, Mapping):
        raise ManifestConstructionError(
            "resolved_config must be a mapping."
        )

    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ManifestConstructionError(
                "seed must be an integer when provided."
            )

    repo_path = _normalize_path(repo_root)
    resolved_dataset_path = _normalize_path(dataset_path)
    resolved_artifact_root = _normalize_path(artifact_root)

    state = (
        capture_git_state(repo_path)
        if git_state is None
        else git_state
    )

    if state.repo_root != repo_path:
        repo_path = state.repo_root

    provenance: dict[str, Any] = {
        "repository": repository,
        "git_commit": state.commit,
        "git_dirty": state.dirty,
        "config_digest": {
            "algorithm": "sha256",
            "value": sha256_mapping(resolved_config),
            "scope": "canonical_json_mapping",
        },
    }

    if state.branch is not None:
        provenance["git_branch"] = state.branch

    if config_source is not None:
        provenance["config_source"] = str(
            _normalize_path(config_source)
        )

    runtime: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    if machine_role is not None:
        machine_role = validate_storage_identifier(
            machine_role,
            field_name="machine_role",
        )
        runtime["machine_role"] = machine_role

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "schema_type": "manifest",
        "study_id": study_id,
        "experiment_id": experiment_id,
        "execution_id": execution_id,
        "status": status,
        "provenance": provenance,
        "runtime": runtime,
        "dataset": {
            "dataset_id": dataset_id,
            "resolved_path": str(resolved_dataset_path),
        },
        "artifacts": {
            "artifact_root": str(resolved_artifact_root),
        },
    }

    if started_at is not None:
        manifest["timestamps"] = {
            "started_at": _iso8601(started_at),
        }

    if seed is not None:
        manifest["seed"] = seed

    if scheduler is not None:
        if not isinstance(scheduler, Mapping):
            raise ManifestConstructionError(
                "scheduler must be a mapping when provided."
            )

        manifest["scheduler"] = dict(scheduler)

    return manifest


def write_manifest_json(
    manifest: Mapping[str, Any],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write a manifest as deterministic, human-readable JSON.

    The parent directory must already exist. This function deliberately does
    not create execution directories because directory creation belongs to
    orchestration rather than provenance.

    Existing files are protected unless ``overwrite=True`` is explicitly
    supplied.
    """

    if not isinstance(manifest, Mapping):
        raise ManifestWriteError(
            "manifest must be a mapping."
        )

    destination = _normalize_path(path)

    if not destination.parent.is_dir():
        raise ManifestWriteError(
            f"Manifest parent directory does not exist: "
            f"{destination.parent}"
        )

    if destination.exists() and not overwrite:
        raise ManifestWriteError(
            f"Refusing to overwrite existing manifest: {destination}"
        )

    try:
        serialized = json.dumps(
            dict(manifest),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestWriteError(
            "Manifest cannot be serialized as portable JSON."
        ) from exc

    temporary = destination.with_name(
        destination.name + ".tmp"
    )

    try:
        temporary.write_text(
            serialized + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

        raise ManifestWriteError(
            f"Unable to write manifest: {destination}"
        ) from exc

    return destination
