from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts/metrics/aggregate.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate = _load(
    SCRIPT_PATH,
    "trustforge_test_metrics_aggregate",
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


def _records(
    *,
    dataset: str = "ustc_tfc2016",
    seed: int = 42,
):
    return [
        {
            "dataset": dataset,
            "model": family,
            "seed": seed,
            "budget_per_class": 2000,
            "source_path": (
                f"/history/{family}/summaries/"
                "summary_20260404_161124.json"
            ),
            "fid": float(index + 1),
            "memorization": {
                "nn_dist_mean": float(index + 1) / 100,
            },
            "deltas_RS_minus_R": {
                "accuracy": 0.01 * (index + 1),
                "macro_f1": 0.001 * (index + 1),
            },
        }
        for index, family in enumerate(FAMILIES)
    ]


def test_load_canonical_records_delegates_to_trustforge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_view = object()
    records = tuple(_records())

    class FakeExports:
        @staticmethod
        def load_paper01_export_view(
            repo_root,
            *,
            dataset,
            seed,
        ):
            assert repo_root == Path("/repo")
            assert dataset == "ustc_tfc2016"
            assert seed == 42
            return fake_view

    class FakeCompat:
        @staticmethod
        def paper01_export_view_to_legacy_jsonl(view):
            assert view is fake_view
            return records

    import sys

    monkeypatch.setitem(
        sys.modules,
        "trustforge.paper01_exports",
        FakeExports,
    )
    monkeypatch.setitem(
        sys.modules,
        "trustforge.paper01_compat",
        FakeCompat,
    )

    loaded = aggregate.load_canonical_records(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert loaded == records


def test_validate_requires_exactly_seven_records() -> None:
    with pytest.raises(
        aggregate.CanonicalMetricsError,
        match="exactly 7 records",
    ):
        aggregate.validate_canonical_records(
            _records()[:-1],
            dataset="ustc_tfc2016",
            seed=42,
        )


@pytest.mark.parametrize(
    "alias",
    ["latest.json", "paper1.json"],
)
def test_validate_rejects_alias_authority(
    alias: str,
) -> None:
    records = _records()
    records[0]["source_path"] = (
        f"/history/{alias}"
    )

    with pytest.raises(
        aggregate.CanonicalMetricsError,
        match="alias source",
    ):
        aggregate.validate_canonical_records(
            records,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_validate_rejects_seed_mismatch() -> None:
    records = _records()
    records[0]["seed"] = 43

    with pytest.raises(
        aggregate.CanonicalMetricsError,
        match="does not match requested seed",
    ):
        aggregate.validate_canonical_records(
            records,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_validate_rejects_budget_mismatch() -> None:
    records = _records()
    records[0]["budget_per_class"] = 100

    with pytest.raises(
        aggregate.CanonicalMetricsError,
        match="Paper 1 budget 2000",
    ):
        aggregate.validate_canonical_records(
            records,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_validate_rejects_duplicate_family() -> None:
    records = _records()
    records[-1]["model"] = "gan"

    with pytest.raises(
        aggregate.CanonicalMetricsError,
        match="all 7 Paper 1 families",
    ):
        aggregate.validate_canonical_records(
            records,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_canonical_metrics_uses_one_record_per_model() -> None:
    records = _records()

    metrics = aggregate.canonical_metrics(
        records
    )

    assert len(metrics) == 7
    assert metrics[
        "ConditionalAutoregressive"
    ]["fid"] == 1.0
    assert metrics[
        "ConditionalAutoregressive"
    ]["nn"] == 0.01


def test_canonical_metrics_does_not_choose_minimum_across_runs() -> None:
    record = _records()[0]

    metrics = aggregate.canonical_metrics(
        [record]
    )

    name = "ConditionalAutoregressive"
    assert metrics[name]["fid"] == 1.0

    # canonical_metrics rejects a second record for the
    # same model rather than silently selecting a minimum.
    duplicate = dict(record)
    duplicate["fid"] = 0.001

    with pytest.raises(
        aggregate.CanonicalMetricsError,
        match="duplicate canonical model",
    ):
        aggregate.canonical_metrics(
            [record, duplicate]
        )


def test_pull_metrics_supports_compatibility_fields() -> None:
    record = {
        "fid": 2.5,
        "memorization": {
            "nn_dist_mean": 0.2,
        },
        "deltas_RS_minus_R": {
            "accuracy": 0.01,
            "macro_f1": -0.02,
        },
    }

    assert aggregate.pull_metrics(
        record
    ) == (
        2.5,
        0.2,
        0.01,
        -0.02,
    )


def test_write_csv_preserves_legacy_column_contract(
    tmp_path: Path,
) -> None:
    metrics = aggregate.canonical_metrics(
        _records()
    )
    rows = aggregate.csv_rows(metrics)

    out = tmp_path / "metrics.csv"
    aggregate.write_csv(
        out,
        rows,
    )

    with out.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        body = list(reader)

    assert header == [
        "Model",
        "best_FID",
        "best_NNdist",
        "DeltaAcc_RSminusR",
        "DeltaF1_RSminusR",
    ]
    assert len(body) == 7


@pytest.mark.parametrize(
    (
        "canonical",
        "dataset",
        "seed",
        "artifacts",
        "phase1",
        "message",
    ),
    [
        (
            True,
            None,
            42,
            None,
            None,
            "--dataset is required",
        ),
        (
            True,
            "ustc_tfc2016",
            None,
            None,
            None,
            "--seed is required",
        ),
        (
            False,
            "ustc_tfc2016",
            42,
            None,
            None,
            "require --canonical",
        ),
        (
            True,
            "ustc_tfc2016",
            42,
            "/legacy",
            None,
            "--artifacts is legacy-only",
        ),
        (
            True,
            "ustc_tfc2016",
            42,
            None,
            "/legacy.jsonl",
            "--phase1 is legacy-only",
        ),
        (
            False,
            None,
            None,
            None,
            None,
            "--artifacts is required",
        ),
    ],
)
def test_mode_validation_fails_closed(
    canonical,
    dataset,
    seed,
    artifacts,
    phase1,
    message,
) -> None:
    args = SimpleNamespace(
        canonical=canonical,
        dataset=dataset,
        seed=seed,
        artifacts=artifacts,
        phase1=phase1,
    )

    with pytest.raises(
        SystemExit,
        match=message,
    ):
        aggregate._validate_args(args)


def test_legacy_collect_files_behavior_remains(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "gan"
        / "summaries"
        / "summary_20260101_000000.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        "{}",
        encoding="utf-8",
    )

    files = aggregate.collect_files(
        str(tmp_path),
        None,
    )

    assert str(path) in files


def test_canonical_path_does_not_call_legacy_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError(
            "legacy discovery must not run"
        )

    monkeypatch.setattr(
        aggregate,
        "collect_files",
        fail,
    )

    metrics = aggregate.canonical_metrics(
        _records()
    )

    assert len(metrics) == 7


def test_source_declares_canonical_contract() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "--canonical" in source
    assert "load_paper01_export_view" in source
    assert "paper01_export_view_to_legacy_jsonl" in source
    assert "duplicate canonical model" in source
