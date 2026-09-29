from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trustforge.paper01_consumer_view import (
    EXPECTED_DATASETS,
    EXPECTED_FAMILIES,
    EXPECTED_SEEDS,
    Paper01ConsumerViewError,
    load_paper01_consumer_view,
    project_paper01_consumer_view,
)
from trustforge.paper01_execution_evidence import (
    Paper01LinkageStudy,
    load_paper01_linkage_study,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dataset", sorted(EXPECTED_DATASETS))
@pytest.mark.parametrize("seed", sorted(EXPECTED_SEEDS))
def test_all_six_canonical_views_have_exactly_seven_records(
    dataset: str,
    seed: int,
) -> None:
    view = load_paper01_consumer_view(
        REPO_ROOT,
        dataset=dataset,
        seed=seed,
    )

    assert len(view) == 7
    assert view.dataset == dataset
    assert view.seed == seed
    assert set(view.by_family()) == EXPECTED_FAMILIES


@pytest.mark.parametrize("dataset", sorted(EXPECTED_DATASETS))
@pytest.mark.parametrize("seed", sorted(EXPECTED_SEEDS))
def test_each_view_has_unique_families(
    dataset: str,
    seed: int,
) -> None:
    view = load_paper01_consumer_view(
        REPO_ROOT,
        dataset=dataset,
        seed=seed,
    )

    families = [record.family for record in view.records]

    assert len(families) == len(set(families)) == 7


def test_view_records_are_deterministically_sorted_by_family() -> None:
    view = load_paper01_consumer_view(
        REPO_ROOT,
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert [record.family for record in view.records] == sorted(
        EXPECTED_FAMILIES
    )


def test_view_get_returns_expected_family_record() -> None:
    view = load_paper01_consumer_view(
        REPO_ROOT,
        dataset="cicmaldroid2020",
        seed=42,
    )

    record = view.get("gaussianmixture")

    assert record.experiment_id == (
        "cicmaldroid_gaussianmixture_b2000_seed42"
    )
    assert record.historical_run_id == "gaussianmixture_s42"
    assert record.budget_per_class == 2000


def test_view_get_rejects_unknown_family() -> None:
    view = load_paper01_consumer_view(
        REPO_ROOT,
        dataset="ustc_tfc2016",
        seed=42,
    )

    with pytest.raises(KeyError, match="Unknown Paper 1 family"):
        view.get("not-a-family")


def test_accepted_summary_path_is_timestamped_not_latest() -> None:
    view = load_paper01_consumer_view(
        REPO_ROOT,
        dataset="cicmaldroid2020",
        seed=42,
    )

    record = view.get("gaussianmixture")

    assert record.accepted_summary_path.name == (
        "summary_20260410_091359.json"
    )
    assert record.accepted_summary_path.name != "latest.json"


@pytest.mark.parametrize(
    "dataset",
    [
        "",
        "unknown",
        "CICMalDroid",
        "ustc",
    ],
)
def test_invalid_dataset_is_rejected(dataset: str) -> None:
    with pytest.raises(Paper01ConsumerViewError):
        load_paper01_consumer_view(
            REPO_ROOT,
            dataset=dataset,
            seed=42,
        )


@pytest.mark.parametrize("seed", [0, 41, 45, 999])
def test_invalid_seed_is_rejected(seed: int) -> None:
    with pytest.raises(Paper01ConsumerViewError):
        load_paper01_consumer_view(
            REPO_ROOT,
            dataset="ustc_tfc2016",
            seed=seed,
        )


def test_string_seed_is_normalized_when_valid() -> None:
    view = load_paper01_consumer_view(
        REPO_ROOT,
        dataset="ustc_tfc2016",
        seed="42",  # type: ignore[arg-type]
    )

    assert view.seed == 42


def test_dataset_is_normalized_to_lowercase() -> None:
    view = load_paper01_consumer_view(
        REPO_ROOT,
        dataset=" USTC_TFC2016 ",
        seed=42,
    )

    assert view.dataset == "ustc_tfc2016"


def test_projection_does_not_open_historical_summary_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args, **kwargs):
        text = str(self)
        if "/home/bruno.fonkeng/gencys/" in text:
            raise AssertionError(
                "consumer projection must not read historical artifact files"
            )
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    view = load_paper01_consumer_view(
        REPO_ROOT,
        dataset="cicmaldroid2020",
        seed=42,
    )

    assert len(view) == 7


def test_projection_fails_if_view_is_missing_a_family() -> None:
    study = load_paper01_linkage_study(REPO_ROOT)

    reduced = tuple(
        record
        for record in study.records
        if not (
            record.data["experiment"]["dataset"] == "ustc_tfc2016"
            and record.data["experiment"]["seed"] == 42
            and record.data["experiment"]["family"] == "gan"
        )
    )

    broken = Paper01LinkageStudy(
        index_path=study.index_path,
        index=study.index,
        records=reduced,
    )

    with pytest.raises(
        Paper01ConsumerViewError,
        match="must contain exactly 7 records",
    ):
        project_paper01_consumer_view(
            broken,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_projection_fails_on_duplicate_family() -> None:
    study = load_paper01_linkage_study(REPO_ROOT)

    target = next(
        record
        for record in study.records
        if (
            record.data["experiment"]["dataset"] == "ustc_tfc2016"
            and record.data["experiment"]["seed"] == 42
            and record.data["experiment"]["family"] == "gan"
        )
    )

    without_vae = tuple(
        record
        for record in study.records
        if not (
            record.data["experiment"]["dataset"] == "ustc_tfc2016"
            and record.data["experiment"]["seed"] == 42
            and record.data["experiment"]["family"] == "vae"
        )
    )

    broken = Paper01LinkageStudy(
        index_path=study.index_path,
        index=study.index,
        records=without_vae + (target,),
    )

    with pytest.raises(
        Paper01ConsumerViewError,
        match="duplicate model families",
    ):
        project_paper01_consumer_view(
            broken,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_projection_rejects_latest_json_if_canonical_data_is_corrupted() -> None:
    study = load_paper01_linkage_study(REPO_ROOT)

    target = next(
        record
        for record in study.records
        if (
            record.data["experiment"]["dataset"] == "ustc_tfc2016"
            and record.data["experiment"]["seed"] == 42
            and record.data["experiment"]["family"] == "gan"
        )
    )

    import copy

    corrupted_data = copy.deepcopy(dict(target.data))
    corrupted_data["accepted_evaluation"]["accepted_summary"]["path"] = (
        "/tmp/latest.json"
    )

    corrupted_record = replace(
        target,
        data=corrupted_data,
    )

    records = tuple(
        corrupted_record if record is target else record
        for record in study.records
    )

    broken = Paper01LinkageStudy(
        index_path=study.index_path,
        index=study.index,
        records=records,
    )

    with pytest.raises(
        Paper01ConsumerViewError,
        match="latest.json cannot be authoritative",
    ):
        project_paper01_consumer_view(
            broken,
            dataset="ustc_tfc2016",
            seed=42,
        )
