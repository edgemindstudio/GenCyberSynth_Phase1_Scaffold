"""
Canonical TrustForge storage layout construction.

TrustForge separates scientific identity from concrete filesystem location.

The storage roots describe where data and artifacts live on a particular
machine. Stable identifiers such as study_id, experiment_id, execution_id,
and dataset_key determine the relative layout beneath those roots.

Canonical layouts
-----------------

Dataset:

    DATA_ROOT / <dataset_key>

Study artifacts:

    ARTIFACTS_ROOT / <study_id>

Experiment artifacts:

    ARTIFACTS_ROOT / <study_id> / <experiment_id>

Execution artifacts:

    ARTIFACTS_ROOT / <study_id> / <experiment_id> / <execution_id>

These functions construct paths only. They do not create directories.
"""

from __future__ import annotations

from pathlib import Path

from trustforge.storage.paths import (
    resolve_artifacts_root,
    resolve_data_root,
)


class InvalidStorageIdentifier(ValueError):
    """Raised when an identifier is unsafe as a portable path segment."""


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}

_WINDOWS_INVALID_CHARACTERS = frozenset('<>:"/\\|?*')


def validate_storage_identifier(
    value: str,
    *,
    field_name: str = "identifier",
) -> str:
    """
    Validate an identifier for use as one portable filesystem path segment.

    The function intentionally does not impose a scientific naming grammar.
    Study, experiment, execution, and dataset identifiers remain scientific
    identifiers first.

    Validation here protects only filesystem portability and path safety.

    Parameters
    ----------
    value:
        Identifier to validate.

    field_name:
        Human-readable field name used in error messages.

    Returns
    -------
    str
        The original validated identifier.

    Raises
    ------
    InvalidStorageIdentifier
        If the identifier is empty, unsafe, or not portable across supported
        execution environments.
    """

    if not isinstance(value, str):
        raise InvalidStorageIdentifier(
            f"{field_name} must be a string."
        )

    if not value:
        raise InvalidStorageIdentifier(
            f"{field_name} must not be empty."
        )

    if value != value.strip():
        raise InvalidStorageIdentifier(
            f"{field_name} must not contain leading or trailing whitespace."
        )

    if value in {".", ".."}:
        raise InvalidStorageIdentifier(
            f"{field_name} must not be '.' or '..'."
        )

    if any(character in _WINDOWS_INVALID_CHARACTERS for character in value):
        raise InvalidStorageIdentifier(
            f"{field_name} contains a character that is not portable "
            "across supported filesystems."
        )

    if any(ord(character) < 32 for character in value):
        raise InvalidStorageIdentifier(
            f"{field_name} must not contain control characters."
        )

    if value.endswith(".") or value.endswith(" "):
        raise InvalidStorageIdentifier(
            f"{field_name} must not end with a period or space."
        )

    reserved_candidate = value.split(".", 1)[0].upper()

    if reserved_candidate in _WINDOWS_RESERVED_NAMES:
        raise InvalidStorageIdentifier(
            f"{field_name} uses a filesystem-reserved name: {value!r}."
        )

    return value


def _normalize_explicit_root(root: Path | str) -> Path:
    """Normalize a caller-provided filesystem root without requiring existence."""

    return Path(root).expanduser().resolve(strict=False)


def dataset_path(
    dataset_key: str,
    *,
    data_root: Path | str | None = None,
) -> Path:
    """
    Construct the canonical path for a TrustForge dataset identity.

    If ``data_root`` is omitted, TRUSTFORGE_DATA_ROOT resolution is used.
    """

    dataset_key = validate_storage_identifier(
        dataset_key,
        field_name="dataset_key",
    )

    root = (
        resolve_data_root(required=True)
        if data_root is None
        else _normalize_explicit_root(data_root)
    )

    if root is None:
        raise RuntimeError(
            "Data root resolution unexpectedly returned no path."
        )

    return root / dataset_key


def study_artifact_root(
    study_id: str,
    *,
    artifacts_root: Path | str | None = None,
) -> Path:
    """
    Construct the canonical artifact root for a TrustForge study.
    """

    study_id = validate_storage_identifier(
        study_id,
        field_name="study_id",
    )

    root = (
        resolve_artifacts_root(required=True)
        if artifacts_root is None
        else _normalize_explicit_root(artifacts_root)
    )

    if root is None:
        raise RuntimeError(
            "Artifact root resolution unexpectedly returned no path."
        )

    return root / study_id


def experiment_artifact_root(
    study_id: str,
    experiment_id: str,
    *,
    artifacts_root: Path | str | None = None,
) -> Path:
    """
    Construct the canonical artifact root for a TrustForge experiment.
    """

    study_id = validate_storage_identifier(
        study_id,
        field_name="study_id",
    )

    experiment_id = validate_storage_identifier(
        experiment_id,
        field_name="experiment_id",
    )

    return study_artifact_root(
        study_id,
        artifacts_root=artifacts_root,
    ) / experiment_id


def execution_artifact_root(
    study_id: str,
    experiment_id: str,
    execution_id: str,
    *,
    artifacts_root: Path | str | None = None,
) -> Path:
    """
    Construct the canonical artifact root for one execution attempt.
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

    return experiment_artifact_root(
        study_id,
        experiment_id,
        artifacts_root=artifacts_root,
    ) / execution_id
