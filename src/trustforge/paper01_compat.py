"""
Paper 1 canonical-to-legacy compatibility projections.

M6.5.4B1 defines a deterministic, read-only compatibility record that can feed
the existing Paper 1 JSONL ecosystem without allowing legacy files or glob
ordering to choose scientific authority.

Authority:
    Paper01ExportRecord -> accepted timestamped historical summary

Compatibility:
    preserve the historical summary payload
    preserve nested metrics/provenance
    add legacy top-level shims used by downstream tools
    use canonical accepted_summary_path as source_path

This module writes nothing.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from trustforge.paper01_exports import (
    Paper01ExportRecord,
    Paper01ExportView,
)


class Paper01CompatibilityError(RuntimeError):
    """Raised when deterministic compatibility materialization is impossible."""


_AUDIT_SHIMS = (
    "config_path",
    "config_sha1",
    "git_commit",
    "caps",
    "budget_per_class",
)


def _dig(value: Mapping[str, Any], *path: str) -> Any:
    cur: Any = value
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _set_if_missing(record: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and record.get(key) is None:
        record[key] = value


def _timestamp_from_summary_path(path: Path) -> str:
    """
    Derive a deterministic UTC timestamp from summary_YYYYMMDD_HHMMSS.json.

    Canonical compatibility export refuses to invent a wall-clock timestamp.
    """
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[0] != "summary":
        raise Paper01CompatibilityError(
            f"cannot derive deterministic timestamp from accepted summary: {path}"
        )

    ymd = parts[-2]
    hms = parts[-1]

    try:
        parsed = datetime.strptime(
            ymd + hms,
            "%Y%m%d%H%M%S",
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise Paper01CompatibilityError(
            f"cannot derive deterministic timestamp from accepted summary: {path}"
        ) from exc

    return parsed.isoformat(timespec="seconds")


def _attach_audit_fields(record: dict[str, Any]) -> None:
    run_meta = record.get("run_meta")
    has_run_meta = isinstance(run_meta, dict) and bool(run_meta)

    if has_run_meta:
        for key in _AUDIT_SHIMS:
            _set_if_missing(record, key, run_meta.get(key))
        return

    if any(record.get(key) is not None for key in _AUDIT_SHIMS):
        record["run_meta"] = {
            key: record.get(key)
            for key in _AUDIT_SHIMS
        }


def _promote_top_level_metrics(record: dict[str, Any]) -> None:
    metrics = (
        record.get("metrics")
        if isinstance(record.get("metrics"), dict)
        else {}
    )
    counts = (
        record.get("counts")
        if isinstance(record.get("counts"), dict)
        else {}
    )
    meta = (
        record.get("metrics_meta")
        if isinstance(record.get("metrics_meta"), dict)
        else {}
    )
    downstream = _dig(metrics, "downstream")
    if not isinstance(downstream, Mapping):
        downstream = {}

    num_real = _first(
        counts.get("num_real"),
        record.get("counts.num_real"),
        record.get("num_real"),
        _dig(record, "images", "train_real"),
    )
    num_fake = _first(
        counts.get("num_fake"),
        counts.get("synthetic"),
        record.get("counts.num_fake"),
        record.get("num_fake"),
        _dig(record, "images", "synthetic"),
    )
    _set_if_missing(record, "num_real", num_real)
    _set_if_missing(record, "num_fake", num_fake)

    kid = _first(
        metrics.get("kid"),
        record.get("metrics.kid"),
        _dig(record, "generative", "kid"),
    )
    fid = _first(
        metrics.get("fid_macro"),
        metrics.get("fid"),
        record.get("metrics.fid_macro"),
        record.get("metrics.fid"),
        _dig(record, "generative", "fid_macro"),
        _dig(record, "generative", "fid"),
    )
    cfid = _first(
        metrics.get("cfid"),
        metrics.get("cfid_macro"),
        record.get("metrics.cfid"),
        record.get("metrics.cfid_macro"),
        _dig(record, "generative", "cfid_macro"),
    )
    ms_ssim = _first(
        metrics.get("ms_ssim"),
        record.get("metrics.ms_ssim"),
        _dig(record, "generative", "diversity"),
        _dig(record, "generative", "ms_ssim"),
    )

    _set_if_missing(record, "kid", kid)
    _set_if_missing(record, "fid", fid)
    _set_if_missing(record, "cfid", cfid)
    _set_if_missing(record, "ms_ssim", ms_ssim)

    macro_f1 = _first(
        downstream.get("macro_f1"),
        record.get("metrics.downstream.macro_f1"),
        _dig(record, "utility_real_plus_synth", "macro_f1"),
    )
    macro_auprc = _first(
        downstream.get("macro_auprc"),
        record.get("metrics.downstream.macro_auprc"),
        _dig(record, "utility_real_plus_synth", "macro_auprc"),
    )
    balanced_acc = _first(
        downstream.get("balanced_acc"),
        record.get("metrics.downstream.balanced_acc"),
        _dig(record, "utility_real_plus_synth", "balanced_accuracy"),
        _dig(record, "utility_real_plus_synth", "balanced_acc"),
        _dig(record, "utility_real_plus_synth", "bal_acc"),
    )
    gen_precision = _first(
        metrics.get("gen_precision"),
        record.get("metrics.gen_precision"),
        _dig(record, "utility_real_plus_synth", "macro_precision"),
        _dig(
            record,
            "utility_real_plus_synth",
            "per_class",
            "macro_precision",
        ),
    )
    gen_recall = _first(
        metrics.get("gen_recall"),
        record.get("metrics.gen_recall"),
        _dig(record, "utility_real_plus_synth", "macro_recall"),
        _dig(
            record,
            "utility_real_plus_synth",
            "per_class",
            "macro_recall",
        ),
    )

    _set_if_missing(record, "macro_f1", macro_f1)
    _set_if_missing(record, "macro_auprc", macro_auprc)
    _set_if_missing(record, "balanced_acc", balanced_acc)
    _set_if_missing(record, "gen_precision", gen_precision)
    _set_if_missing(record, "gen_recall", gen_recall)

    _set_if_missing(
        record,
        "computed_at",
        _first(meta.get("computed_at"), record.get("computed_at")),
    )
    _set_if_missing(
        record,
        "metrics_version",
        _first(meta.get("metrics_version"), record.get("metrics_version")),
    )
    _set_if_missing(
        record,
        "kid_mode",
        _first(meta.get("kid_mode"), record.get("kid_mode")),
    )


def paper01_export_record_to_legacy_jsonl(
    export_record: Paper01ExportRecord,
) -> dict[str, Any]:
    """
    Materialize one deterministic compatibility row.

    The source summary payload is defensively copied and never mutated.
    """
    record = deepcopy(export_record.summary_payload())

    record.setdefault("model", export_record.family)
    record.setdefault("seed", export_record.seed)
    record.setdefault(
        "budget_per_class",
        export_record.budget_per_class,
    )
    record["source_path"] = str(export_record.accepted_summary_path)

    if not record.get("timestamp"):
        record["timestamp"] = _timestamp_from_summary_path(
            export_record.accepted_summary_path
        )

    if not record.get("run_id"):
        record["run_id"] = export_record.historical_run_id

    _attach_audit_fields(record)
    _promote_top_level_metrics(record)

    return record


def paper01_export_view_to_legacy_jsonl(
    export_view: Paper01ExportView,
) -> tuple[dict[str, Any], ...]:
    """
    Convert exactly seven canonical export records into legacy-compatible rows.

    Output order is deterministic by model/family.
    """
    rows = [
        paper01_export_record_to_legacy_jsonl(record)
        for record in export_view
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("model", "")),
            int(row.get("seed", 0)),
        )
    )

    if len(rows) != 7:
        raise Paper01CompatibilityError(
            f"Paper 1 compatibility view dataset={export_view.dataset!r}, "
            f"seed={export_view.seed} must contain exactly 7 rows; "
            f"found {len(rows)}"
        )

    return tuple(rows)
