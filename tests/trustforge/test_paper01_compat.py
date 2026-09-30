from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from trustforge.paper01_compat import (
    Paper01CompatibilityError,
    paper01_export_record_to_legacy_jsonl,
    paper01_export_view_to_legacy_jsonl,
)


FAMILIES = (
    "autoregressive",
    "diffusion",
    "gan",
    "gaussianmixture",
    "maskedautoflow",
    "restrictedboltzmann",
    "vae",
)


class FakeExportRecord:
    def __init__(
        self,
        *,
        family: str,
        seed: int = 42,
        dataset: str = "ustc_tfc2016",
        budget: int = 2000,
        summary_name: str | None = None,
        payload: dict | None = None,
    ):
        self.family = family
        self.seed = seed
        self.dataset = dataset
        self.budget_per_class = budget
        self.experiment_id = (
            f"{dataset}_{family}_b{budget}_seed{seed}"
        )
        self.historical_run_id = f"{family}_s{seed}"
        self.accepted_summary_path = Path(
            "/history"
        ) / (
            summary_name
            or f"summary_20260404_1200{seed % 10:02d}.json"
        )
        self._payload = deepcopy(
            payload
            if payload is not None
            else {
                "model": family,
                "seed": seed,
                "budget_per_class": budget,
                "counts": {
                    "num_real": 100,
                    "num_fake": 18000,
                },
                "metrics": {
                    "kid": 0.1,
                    "cfid": 1.2,
                    "ms_ssim": 0.3,
                    "downstream": {
                        "macro_f1": 0.8,
                        "macro_auprc": 0.7,
                        "balanced_acc": 0.75,
                    },
                },
                "run_meta": {
                    "config_path": "configs/paper1.yaml",
                    "config_sha1": "abc123",
                    "git_commit": "deadbeef",
                    "caps": {"per_class_cap": 200},
                    "budget_per_class": budget,
                },
            }
        )

    def summary_payload(self) -> dict:
        return deepcopy(self._payload)


class FakeExportView:
    def __init__(
        self,
        records,
        *,
        dataset: str = "ustc_tfc2016",
        seed: int = 42,
    ):
        self.records = tuple(records)
        self.dataset = dataset
        self.seed = seed

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)


def _seven_records():
    return [
        FakeExportRecord(family=family)
        for family in reversed(FAMILIES)
    ]


def test_compatibility_row_preserves_nested_payload() -> None:
    record = FakeExportRecord(family="gan")

    row = paper01_export_record_to_legacy_jsonl(record)

    assert row["metrics"]["downstream"]["macro_f1"] == 0.8
    assert row["run_meta"]["config_sha1"] == "abc123"
    assert row["counts"]["num_fake"] == 18000


def test_compatibility_row_adds_canonical_source_path() -> None:
    record = FakeExportRecord(family="gan")

    row = paper01_export_record_to_legacy_jsonl(record)

    assert row["source_path"] == str(record.accepted_summary_path)
    assert "latest.json" not in row["source_path"]
    assert "paper1.json" not in row["source_path"]


def test_compatibility_row_promotes_audit_shims() -> None:
    row = paper01_export_record_to_legacy_jsonl(
        FakeExportRecord(family="gan")
    )

    assert row["config_path"] == "configs/paper1.yaml"
    assert row["config_sha1"] == "abc123"
    assert row["git_commit"] == "deadbeef"
    assert row["caps"] == {"per_class_cap": 200}
    assert row["budget_per_class"] == 2000


def test_compatibility_row_promotes_metric_shims() -> None:
    row = paper01_export_record_to_legacy_jsonl(
        FakeExportRecord(family="gan")
    )

    assert row["num_real"] == 100
    assert row["num_fake"] == 18000
    assert row["kid"] == 0.1
    assert row["cfid"] == 1.2
    assert row["ms_ssim"] == 0.3
    assert row["macro_f1"] == 0.8
    assert row["macro_auprc"] == 0.7
    assert row["balanced_acc"] == 0.75


def test_compatibility_row_does_not_overwrite_existing_shims() -> None:
    payload = {
        "model": "gan",
        "seed": 42,
        "budget_per_class": 2000,
        "kid": 9.9,
        "metrics": {"kid": 0.1},
        "run_meta": {
            "config_path": "canonical.yaml",
            "budget_per_class": 2000,
        },
    }

    row = paper01_export_record_to_legacy_jsonl(
        FakeExportRecord(
            family="gan",
            payload=payload,
        )
    )

    assert row["kid"] == 9.9


def test_compatibility_row_preserves_existing_timestamp_and_run_id() -> None:
    payload = {
        "model": "gan",
        "seed": 42,
        "budget_per_class": 2000,
        "timestamp": "historical-timestamp",
        "run_id": "historical-run-id",
    }

    row = paper01_export_record_to_legacy_jsonl(
        FakeExportRecord(
            family="gan",
            payload=payload,
        )
    )

    assert row["timestamp"] == "historical-timestamp"
    assert row["run_id"] == "historical-run-id"


def test_missing_timestamp_is_deterministically_derived() -> None:
    row = paper01_export_record_to_legacy_jsonl(
        FakeExportRecord(
            family="gan",
            summary_name="summary_20260404_161124.json",
            payload={
                "model": "gan",
                "seed": 42,
                "budget_per_class": 2000,
            },
        )
    )

    assert row["timestamp"] == "2026-04-04T16:11:24+00:00"


def test_missing_run_id_uses_canonical_historical_run_id() -> None:
    row = paper01_export_record_to_legacy_jsonl(
        FakeExportRecord(
            family="gan",
            payload={
                "model": "gan",
                "seed": 42,
                "budget_per_class": 2000,
            },
        )
    )

    assert row["run_id"] == "gan_s42"


@pytest.mark.parametrize(
    "summary_name",
    [
        "unexpected.json",
        "summary_bad_timestamp.json",
    ],
)
def test_nondeterministic_timestamp_fallback_is_rejected(
    summary_name: str,
) -> None:
    with pytest.raises(
        Paper01CompatibilityError,
        match="cannot derive deterministic timestamp",
    ):
        paper01_export_record_to_legacy_jsonl(
            FakeExportRecord(
                family="gan",
                summary_name=summary_name,
                payload={
                    "model": "gan",
                    "seed": 42,
                    "budget_per_class": 2000,
                },
            )
        )


def test_compatibility_conversion_does_not_mutate_export_payload() -> None:
    record = FakeExportRecord(family="gan")
    before = record.summary_payload()

    row = paper01_export_record_to_legacy_jsonl(record)
    row["model"] = "mutated"
    row["metrics"]["kid"] = 999

    assert record.summary_payload() == before


def test_view_conversion_returns_exactly_seven_rows() -> None:
    rows = paper01_export_view_to_legacy_jsonl(
        FakeExportView(_seven_records())
    )

    assert len(rows) == 7


def test_view_conversion_is_sorted_by_model() -> None:
    rows = paper01_export_view_to_legacy_jsonl(
        FakeExportView(_seven_records())
    )

    assert [row["model"] for row in rows] == sorted(FAMILIES)


def test_incomplete_view_is_rejected() -> None:
    with pytest.raises(
        Paper01CompatibilityError,
        match="must contain exactly 7 rows",
    ):
        paper01_export_view_to_legacy_jsonl(
            FakeExportView(_seven_records()[:-1])
        )
