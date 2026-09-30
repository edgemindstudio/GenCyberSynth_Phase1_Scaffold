"""
Canonical Paper 1 export records.

M6.5.4A establishes the read-only bridge between canonical Paper 1 authority
selection and downstream compatibility/export consumers.

The authority chain is:

    explicit dataset + seed
        -> canonical Paper 1 consumer view
        -> accepted timestamped summary path
        -> read-only summary payload

This module does not write JSONL, CSV, reports, or historical artifacts. It
does not select latest.json, paper1.json, newest files, or local glob results.

Permanent distinctions preserved:
    CANONICAL AUTHORITY SELECTION != LEGACY OUTPUT FORMAT
    HISTORICAL SUMMARY PAYLOAD != MUTABLE WORKING RECORD
    latest.json != AUTHORITATIVE EVIDENCE
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from trustforge.paper01_consumer_view import (
    Paper01ConsumerRecord,
    Paper01ConsumerView,
    load_paper01_consumer_view,
)


class Paper01ExportError(RuntimeError):
    """Raised when canonical export material cannot be produced safely."""


@dataclass(frozen=True)
class Paper01ExportRecord:
    """One canonical experiment plus its accepted historical summary payload."""

    experiment_id: str
    dataset: str
    family: str
    seed: int
    budget_per_class: int
    historical_run_id: str
    accepted_summary_path: Path
    consumer_record: Paper01ConsumerRecord
    _summary_payload: Mapping[str, Any]

    def summary_payload(self) -> dict[str, Any]:
        """Return a defensive copy of the accepted summary payload."""
        return deepcopy(dict(self._summary_payload))


@dataclass(frozen=True)
class Paper01ExportView:
    """Exactly seven canonical export records for one Paper 1 dataset/seed."""

    dataset: str
    seed: int
    records: tuple[Paper01ExportRecord, ...]

    def by_family(self) -> dict[str, Paper01ExportRecord]:
        return {record.family: record for record in self.records}

    def get(self, family: str) -> Paper01ExportRecord:
        try:
            return self.by_family()[family]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Paper 1 export family for "
                f"dataset={self.dataset!r}, seed={self.seed}: {family!r}"
            ) from exc

    def __iter__(self) -> Iterable[Paper01ExportRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_budget(summary: Mapping[str, Any]) -> int | None:
    run_meta = summary.get("run_meta")
    if not isinstance(run_meta, Mapping):
        run_meta = {}

    return _as_int(
        _first(
            summary.get("budget_per_class"),
            run_meta.get("budget_per_class"),
        )
    )


def _read_summary(path: Path, *, experiment_id: str) -> dict[str, Any]:
    if path.name in {"latest.json", "paper1.json"}:
        raise Paper01ExportError(
            f"{experiment_id}: alias summary cannot be canonical export input: "
            f"{path.name}"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Paper01ExportError(
            f"{experiment_id}: accepted summary is missing: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise Paper01ExportError(
            f"{experiment_id}: accepted summary is invalid JSON: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise Paper01ExportError(
            f"{experiment_id}: accepted summary root must be a mapping: {path}"
        )

    return payload


def _validate_summary_identity(
    consumer_record: Paper01ConsumerRecord,
    summary: Mapping[str, Any],
) -> None:
    summary_seed = _as_int(summary.get("seed"))
    if summary_seed is not None and summary_seed != consumer_record.seed:
        raise Paper01ExportError(
            f"{consumer_record.experiment_id}: summary seed={summary_seed} "
            f"does not match canonical seed={consumer_record.seed}"
        )

    summary_model = summary.get("model")
    if (
        isinstance(summary_model, str)
        and summary_model
        and summary_model != consumer_record.family
    ):
        raise Paper01ExportError(
            f"{consumer_record.experiment_id}: summary model="
            f"{summary_model!r} does not match canonical family="
            f"{consumer_record.family!r}"
        )

    summary_budget = _summary_budget(summary)
    if (
        summary_budget is not None
        and summary_budget != consumer_record.budget_per_class
    ):
        raise Paper01ExportError(
            f"{consumer_record.experiment_id}: summary budget_per_class="
            f"{summary_budget} does not match canonical budget_per_class="
            f"{consumer_record.budget_per_class}"
        )


def export_paper01_consumer_view(
    view: Paper01ConsumerView,
) -> Paper01ExportView:
    """
    Read the seven accepted timestamped summaries selected by a consumer view.

    Historical files are opened read-only. The returned records contain
    defensive summary payloads and preserve the canonical scientific identity.
    """
    records: list[Paper01ExportRecord] = []

    for consumer_record in view:
        summary = _read_summary(
            consumer_record.accepted_summary_path,
            experiment_id=consumer_record.experiment_id,
        )
        _validate_summary_identity(consumer_record, summary)

        records.append(
            Paper01ExportRecord(
                experiment_id=consumer_record.experiment_id,
                dataset=consumer_record.dataset,
                family=consumer_record.family,
                seed=consumer_record.seed,
                budget_per_class=consumer_record.budget_per_class,
                historical_run_id=consumer_record.historical_run_id,
                accepted_summary_path=consumer_record.accepted_summary_path,
                consumer_record=consumer_record,
                _summary_payload=deepcopy(summary),
            )
        )

    records.sort(key=lambda record: record.family)

    if len(records) != 7:
        raise Paper01ExportError(
            f"Paper 1 export view dataset={view.dataset!r}, seed={view.seed} "
            f"must contain exactly 7 records; found {len(records)}"
        )

    families = [record.family for record in records]
    if len(families) != len(set(families)):
        raise Paper01ExportError(
            f"Paper 1 export view dataset={view.dataset!r}, seed={view.seed} "
            "contains duplicate model families"
        )

    return Paper01ExportView(
        dataset=view.dataset,
        seed=view.seed,
        records=tuple(records),
    )


def load_paper01_export_view(
    repo_root: Path | str,
    *,
    dataset: str,
    seed: int,
) -> Paper01ExportView:
    """
    Load canonical Paper 1 authority and materialize read-only export records.

    Dataset and seed are explicit. Authority is delegated entirely to the
    canonical consumer projection layer.
    """
    consumer_view = load_paper01_consumer_view(
        repo_root,
        dataset=dataset,
        seed=seed,
    )
    return export_paper01_consumer_view(consumer_view)
