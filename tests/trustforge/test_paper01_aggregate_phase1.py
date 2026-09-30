from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = (
    REPO_ROOT
    / "tools/aggregate_phase1.py"
)


def _load(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate = _load(
    TOOL_PATH,
    "trustforge_test_aggregate_phase1",
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
            "run_id": f"{family}_run",
            "num_fake": 18000,
            "images": {
                "train_real": 10,
                "val_real": 5,
                "test_real": 5,
            },
            "generative": {
                "fid": 1.0,
                "cfid_macro": 2.0,
            },
            "utility_real_only": {
                "accuracy": 0.8,
                "macro_f1": 0.7,
                "balanced_accuracy": 0.75,
                "macro_auprc": 0.76,
                "recall_at_1pct_fpr": 0.1,
                "ece": 0.2,
                "brier": 0.3,
            },
            "utility_real_plus_synth": {
                "accuracy": 0.81,
                "macro_f1": 0.72,
                "balanced_accuracy": 0.77,
                "macro_auprc": 0.78,
                "recall_at_1pct_fpr": 0.11,
                "ece": 0.19,
                "brier": 0.29,
            },
        }
        for family in FAMILIES
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


def test_canonical_validation_requires_seven_records() -> None:
    with pytest.raises(
        aggregate.CanonicalAggregateError,
        match="exactly 7 records",
    ):
        aggregate._validate_canonical_records(
            _records()[:-1],
            dataset="ustc_tfc2016",
            seed=42,
        )


@pytest.mark.parametrize(
    "alias",
    [
        "latest.json",
        "paper1.json",
    ],
)
def test_canonical_validation_rejects_alias_sources(
    alias: str,
) -> None:
    records = _records()
    records[0]["source_path"] = (
        f"/history/gan/summaries/{alias}"
    )

    with pytest.raises(
        aggregate.CanonicalAggregateError,
        match="alias source",
    ):
        aggregate._validate_canonical_records(
            records,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_canonical_validation_rejects_seed_mismatch() -> None:
    records = _records()
    records[0]["seed"] = 43

    with pytest.raises(
        aggregate.CanonicalAggregateError,
        match="does not match requested seed",
    ):
        aggregate._validate_canonical_records(
            records,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_canonical_validation_rejects_budget_mismatch() -> None:
    records = _records()
    records[0]["budget_per_class"] = 100

    with pytest.raises(
        aggregate.CanonicalAggregateError,
        match="Paper 1 budget 2000",
    ):
        aggregate._validate_canonical_records(
            records,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_canonical_validation_rejects_duplicate_family() -> None:
    records = _records()
    records[-1]["model"] = "gan"

    with pytest.raises(
        aggregate.CanonicalAggregateError,
        match="all 7 Paper 1 families",
    ):
        aggregate._validate_canonical_records(
            records,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_canonical_rows_never_use_filesystem_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "filesystem inference must not "
            "run in canonical mode"
        )

    monkeypatch.setattr(
        aggregate,
        "infer_synth_count_from_fs",
        fail_if_called,
    )

    rows = aggregate.canonical_rows(
        _records()
    )

    assert len(rows) == 7
    assert all(
        row["_synth_fs_count"] is None
        for row in rows
    )
    assert all(
        row["synthetic"] == 18000
        for row in rows
    )


def test_canonical_rows_do_not_use_mtime_authority() -> None:
    rows = aggregate.canonical_rows(
        _records()
    )

    assert all(
        row["_summary_mtime"] is None
        for row in rows
    )


def test_canonical_rows_are_sorted_by_model() -> None:
    records = list(reversed(_records()))

    rows = aggregate.canonical_rows(
        records
    )

    assert [
        row["model"]
        for row in rows
    ] == sorted(FAMILIES)


def test_canonical_rows_do_not_mutate_compatibility_records() -> None:
    records = _records()
    before = json.dumps(
        records,
        sort_keys=True,
    )

    aggregate.canonical_rows(records)

    after = json.dumps(
        records,
        sort_keys=True,
    )
    assert after == before


def test_write_outputs_preserves_canonical_jsonl_records(
    tmp_path: Path,
) -> None:
    records = _records()
    rows = aggregate.canonical_rows(
        records
    )

    csv_path, jsonl_path = (
        aggregate.write_outputs(
            outdir=tmp_path,
            rows=rows,
            jsonl_records=records,
        )
    )

    assert csv_path.exists()
    assert jsonl_path.exists()

    loaded = [
        json.loads(line)
        for line in jsonl_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert loaded == records


@pytest.mark.parametrize(
    ("canonical", "dataset", "seed", "all_flag", "message"),
    [
        (
            True,
            None,
            42,
            False,
            "--dataset is required",
        ),
        (
            True,
            "ustc_tfc2016",
            None,
            False,
            "--seed is required",
        ),
        (
            False,
            "ustc_tfc2016",
            42,
            False,
            "require --canonical",
        ),
        (
            True,
            "ustc_tfc2016",
            42,
            True,
            "--all is legacy-only",
        ),
    ],
)
def test_mode_validation_fails_closed(
    canonical: bool,
    dataset: str | None,
    seed: int | None,
    all_flag: bool,
    message: str,
) -> None:
    args = SimpleNamespace(
        canonical=canonical,
        dataset=dataset,
        seed=seed,
        all=all_flag,
    )

    with pytest.raises(
        SystemExit,
        match=message,
    ):
        aggregate._validate_mode_args(args)


def test_legacy_newest_behavior_remains_available(
    tmp_path: Path,
) -> None:
    older = tmp_path / "older.json"
    newer = tmp_path / "newer.json"

    older.write_text(
        "{}",
        encoding="utf-8",
    )
    newer.write_text(
        "{}",
        encoding="utf-8",
    )

    older.touch()
    newer.touch()

    import os

    os.utime(
        older,
        (1, 1),
    )
    os.utime(
        newer,
        (2, 2),
    )

    assert aggregate.newest(
        [older, newer]
    ) == newer


def test_tool_declares_canonical_authority_path() -> None:
    source = TOOL_PATH.read_text(
        encoding="utf-8"
    )

    assert "--canonical" in source
    assert "load_paper01_export_view" in source
    assert "paper01_export_view_to_legacy_jsonl" in source
    assert "allow_filesystem_fallback=False" in source
    assert "summary_mtime=None" in source
