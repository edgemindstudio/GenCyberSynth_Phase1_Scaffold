from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from trustforge.paper01_exports import (
    Paper01ExportError,
    export_paper01_consumer_view,
)


class FakeConsumerView:
    def __init__(self, dataset: str, seed: int, records):
        self.dataset = dataset
        self.seed = seed
        self.records = tuple(records)

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)


FAMILIES = (
    "autoregressive",
    "diffusion",
    "gan",
    "gaussianmixture",
    "maskedautoflow",
    "restrictedboltzmann",
    "vae",
)


def _write_summary(
    path: Path,
    *,
    family: str,
    seed: int,
    budget: int = 2000,
) -> None:
    path.write_text(
        json.dumps(
            {
                "model": family,
                "seed": seed,
                "budget_per_class": budget,
                "counts": {"num_fake": 123},
                "metrics": {"kid": 0.1},
            }
        ),
        encoding="utf-8",
    )


def _record(
    tmp_path: Path,
    *,
    dataset: str,
    family: str,
    seed: int,
    budget: int = 2000,
):
    path = tmp_path / f"summary_{family}_{seed}.json"
    _write_summary(
        path,
        family=family,
        seed=seed,
        budget=budget,
    )

    return SimpleNamespace(
        experiment_id=f"{dataset}_{family}_b{budget}_seed{seed}",
        dataset=dataset,
        family=family,
        seed=seed,
        budget_per_class=budget,
        historical_run_id=f"{family}_s{seed}",
        accepted_summary_path=path,
    )


def _seven_record_view(
    tmp_path: Path,
    *,
    dataset: str = "ustc_tfc2016",
    seed: int = 42,
) -> FakeConsumerView:
    return FakeConsumerView(
        dataset,
        seed,
        [
            _record(
                tmp_path,
                dataset=dataset,
                family=family,
                seed=seed,
            )
            for family in FAMILIES
        ],
    )


def test_export_view_materializes_exactly_seven_records(
    tmp_path: Path,
) -> None:
    view = export_paper01_consumer_view(
        _seven_record_view(tmp_path)
    )

    assert len(view) == 7
    assert view.dataset == "ustc_tfc2016"
    assert view.seed == 42
    assert set(view.by_family()) == set(FAMILIES)


def test_export_records_are_sorted_by_family(
    tmp_path: Path,
) -> None:
    view = export_paper01_consumer_view(
        _seven_record_view(tmp_path)
    )

    assert [record.family for record in view] == sorted(FAMILIES)


def test_export_record_preserves_canonical_identity(
    tmp_path: Path,
) -> None:
    view = export_paper01_consumer_view(
        _seven_record_view(
            tmp_path,
            dataset="cicmaldroid2020",
            seed=43,
        )
    )

    record = view.get("gan")

    assert record.experiment_id == (
        "cicmaldroid2020_gan_b2000_seed43"
    )
    assert record.dataset == "cicmaldroid2020"
    assert record.family == "gan"
    assert record.seed == 43
    assert record.budget_per_class == 2000
    assert record.historical_run_id == "gan_s43"


def test_summary_payload_is_defensive_copy(
    tmp_path: Path,
) -> None:
    view = export_paper01_consumer_view(
        _seven_record_view(tmp_path)
    )

    record = view.get("gan")

    first = record.summary_payload()
    first["model"] = "mutated"
    first["metrics"]["kid"] = 999

    second = record.summary_payload()

    assert second["model"] == "gan"
    assert second["metrics"]["kid"] == 0.1


def test_export_does_not_modify_source_summary(
    tmp_path: Path,
) -> None:
    source_view = _seven_record_view(tmp_path)
    gan = next(
        record
        for record in source_view.records
        if record.family == "gan"
    )
    before = gan.accepted_summary_path.read_bytes()

    export_paper01_consumer_view(source_view)

    after = gan.accepted_summary_path.read_bytes()

    assert after == before


@pytest.mark.parametrize("alias", ["latest.json", "paper1.json"])
def test_export_rejects_alias_summary_paths(
    tmp_path: Path,
    alias: str,
) -> None:
    view = _seven_record_view(tmp_path)
    records = list(view.records)

    alias_path = tmp_path / alias
    _write_summary(
        alias_path,
        family="gan",
        seed=42,
    )

    target = next(
        record for record in records if record.family == "gan"
    )
    target.accepted_summary_path = alias_path

    with pytest.raises(
        Paper01ExportError,
        match="alias summary cannot be canonical export input",
    ):
        export_paper01_consumer_view(
            FakeConsumerView(
                view.dataset,
                view.seed,
                records,
            )
        )


def test_export_rejects_missing_accepted_summary(
    tmp_path: Path,
) -> None:
    view = _seven_record_view(tmp_path)
    records = list(view.records)

    target = next(
        record for record in records if record.family == "gan"
    )
    target.accepted_summary_path = tmp_path / "missing_summary.json"

    with pytest.raises(
        Paper01ExportError,
        match="accepted summary is missing",
    ):
        export_paper01_consumer_view(
            FakeConsumerView(
                view.dataset,
                view.seed,
                records,
            )
        )


def test_export_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    view = _seven_record_view(tmp_path)
    records = list(view.records)

    target = next(
        record for record in records if record.family == "gan"
    )
    target.accepted_summary_path.write_text(
        "{not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        Paper01ExportError,
        match="invalid JSON",
    ):
        export_paper01_consumer_view(
            FakeConsumerView(
                view.dataset,
                view.seed,
                records,
            )
        )


def test_export_rejects_summary_seed_mismatch(
    tmp_path: Path,
) -> None:
    view = _seven_record_view(tmp_path)
    target = next(
        record for record in view.records if record.family == "gan"
    )

    _write_summary(
        target.accepted_summary_path,
        family="gan",
        seed=44,
    )

    with pytest.raises(
        Paper01ExportError,
        match="summary seed=44",
    ):
        export_paper01_consumer_view(view)


def test_export_rejects_summary_model_mismatch(
    tmp_path: Path,
) -> None:
    view = _seven_record_view(tmp_path)
    target = next(
        record for record in view.records if record.family == "gan"
    )

    _write_summary(
        target.accepted_summary_path,
        family="vae",
        seed=42,
    )

    with pytest.raises(
        Paper01ExportError,
        match="summary model='vae'",
    ):
        export_paper01_consumer_view(view)


def test_export_rejects_summary_budget_mismatch(
    tmp_path: Path,
) -> None:
    view = _seven_record_view(tmp_path)
    target = next(
        record for record in view.records if record.family == "gan"
    )

    _write_summary(
        target.accepted_summary_path,
        family="gan",
        seed=42,
        budget=1000,
    )

    with pytest.raises(
        Paper01ExportError,
        match="summary budget_per_class=1000",
    ):
        export_paper01_consumer_view(view)


def test_export_rejects_incomplete_view(
    tmp_path: Path,
) -> None:
    view = _seven_record_view(tmp_path)

    with pytest.raises(
        Paper01ExportError,
        match="must contain exactly 7 records",
    ):
        export_paper01_consumer_view(
            FakeConsumerView(
                view.dataset,
                view.seed,
                view.records[:-1],
            )
        )


def test_export_get_rejects_unknown_family(
    tmp_path: Path,
) -> None:
    view = export_paper01_consumer_view(
        _seven_record_view(tmp_path)
    )

    with pytest.raises(
        KeyError,
        match="Unknown Paper 1 export family",
    ):
        view.get("not-a-family")
