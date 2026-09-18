"""
Execution identity primitives for TrustForge.

An experiment identifies scientific intent.

An execution identifies one concrete attempt to realize that experiment.

Execution identifiers are therefore operational provenance identifiers rather
than scientific experiment identifiers.
"""

from __future__ import annotations

from datetime import datetime, timezone

from trustforge.storage import (
    InvalidStorageIdentifier,
    validate_storage_identifier,
)


class ExecutionIdentityError(ValueError):
    """Raised when a TrustForge execution identity cannot be constructed."""


def format_execution_timestamp(value: datetime) -> str:
    """
    Format an aware datetime as a portable UTC execution timestamp.

    Result format:

        YYYYMMDDTHHMMSSZ
    """

    if not isinstance(value, datetime):
        raise ExecutionIdentityError(
            "Execution timestamp must be a datetime."
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ExecutionIdentityError(
            "Execution timestamp must include timezone information."
        )

    utc_value = value.astimezone(timezone.utc)

    return utc_value.strftime("%Y%m%dT%H%M%SZ")


def _validate_component(
    value: str,
    *,
    field_name: str,
) -> str:
    """Validate one execution-ID component using portable storage rules."""

    try:
        return validate_storage_identifier(
            value,
            field_name=field_name,
        )
    except InvalidStorageIdentifier as exc:
        raise ExecutionIdentityError(str(exc)) from exc


def build_execution_id(
    started_at: datetime,
    machine: str,
    *,
    scheduler_job_id: str | None = None,
    scheduler_task_id: str | None = None,
    attempt: int | None = None,
) -> str:
    """
    Build a portable TrustForge execution identifier.

    Examples
    --------

    Local execution:

        20260917T143000Z_talon_local

    Slurm execution:

        20260917T143000Z_talon_job123456

    Slurm array task:

        20260917T143000Z_talon_job123456_task7

    Explicit retry:

        20260917T143000Z_talon_job123456_task7_attempt2

    Parameters
    ----------
    started_at:
        Time the execution began. Must be timezone-aware.

    machine:
        Stable execution-machine or machine-role component.

    scheduler_job_id:
        Scheduler job identifier when applicable.

    scheduler_task_id:
        Scheduler array-task identifier when applicable. A task identifier
        cannot exist without a scheduler job identifier.

    attempt:
        Optional positive retry/attempt number.
    """

    timestamp = format_execution_timestamp(started_at)

    machine = _validate_component(
        machine,
        field_name="machine",
    )

    if scheduler_task_id is not None and scheduler_job_id is None:
        raise ExecutionIdentityError(
            "scheduler_task_id requires scheduler_job_id."
        )

    parts = [
        timestamp,
        machine,
    ]

    if scheduler_job_id is None:
        parts.append("local")
    else:
        scheduler_job_id = _validate_component(
            scheduler_job_id,
            field_name="scheduler_job_id",
        )
        parts.append(f"job{scheduler_job_id}")

    if scheduler_task_id is not None:
        scheduler_task_id = _validate_component(
            scheduler_task_id,
            field_name="scheduler_task_id",
        )
        parts.append(f"task{scheduler_task_id}")

    if attempt is not None:
        if not isinstance(attempt, int) or isinstance(attempt, bool):
            raise ExecutionIdentityError(
                "attempt must be a positive integer."
            )

        if attempt <= 0:
            raise ExecutionIdentityError(
                "attempt must be a positive integer."
            )

        parts.append(f"attempt{attempt}")

    execution_id = "_".join(parts)

    try:
        return validate_storage_identifier(
            execution_id,
            field_name="execution_id",
        )
    except InvalidStorageIdentifier as exc:
        raise ExecutionIdentityError(str(exc)) from exc
