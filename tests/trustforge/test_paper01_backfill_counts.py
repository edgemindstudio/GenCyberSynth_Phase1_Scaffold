from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts/utils/backfill_counts.py"
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


backfill = _load(
    SCRIPT_PATH,
    "trustforge_test_backfill_counts",
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


def _canonical_rows(
    *,
    seed: int = 42,
    dataset: str = "ustc_tfc2016",
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
            "config_path": "configs/paper1.yaml",
            "config_sha1": "abc123",
            "git_commit": "deadbeef",
            "counts": {},
        }
        for family in FAMILIES
    ]


def test_validate_canonical_input_accepts_seven_families() -> None:
    rows = _canonical_rows()

    validated = backfill.validate_canonical_input(
        rows,
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert len(validated) == 7
    assert {
        row["model"]
        for row in validated
    } == set(FAMILIES)


def test_validate_canonical_input_requires_exactly_seven_rows() -> None:
    with pytest.raises(
        backfill.CanonicalInputError,
        match="exactly 7 rows",
    ):
        backfill.validate_canonical_input(
            _canonical_rows()[:-1],
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_validate_canonical_input_rejects_seed_mismatch() -> None:
    rows = _canonical_rows()
    rows[0]["seed"] = 43

    with pytest.raises(
        backfill.CanonicalInputError,
        match="does not match requested seed",
    ):
        backfill.validate_canonical_input(
            rows,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_validate_canonical_input_rejects_budget_mismatch() -> None:
    rows = _canonical_rows()
    rows[0]["budget_per_class"] = 1000

    with pytest.raises(
        backfill.CanonicalInputError,
        match="canonical budget 2000",
    ):
        backfill.validate_canonical_input(
            rows,
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
def test_validate_canonical_input_rejects_alias_source(
    alias: str,
) -> None:
    rows = _canonical_rows()
    rows[0]["source_path"] = (
        f"/history/gan/summaries/{alias}"
    )

    with pytest.raises(
        backfill.CanonicalInputError,
        match="alias source_path",
    ):
        backfill.validate_canonical_input(
            rows,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_validate_canonical_input_rejects_non_timestamped_source() -> None:
    rows = _canonical_rows()
    rows[0]["source_path"] = (
        "/history/gan/summaries/result.json"
    )

    with pytest.raises(
        backfill.CanonicalInputError,
        match="timestamped summary",
    ):
        backfill.validate_canonical_input(
            rows,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_validate_canonical_input_rejects_duplicate_family() -> None:
    rows = _canonical_rows()
    rows[-1]["model"] = "gan"

    with pytest.raises(
        backfill.CanonicalInputError,
        match="each of the 7 model families",
    ):
        backfill.validate_canonical_input(
            rows,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_validate_canonical_input_checks_dataset_when_present() -> None:
    rows = _canonical_rows()
    rows[0]["dataset"] = "cicmaldroid2020"

    with pytest.raises(
        backfill.CanonicalInputError,
        match="does not match requested dataset",
    ):
        backfill.validate_canonical_input(
            rows,
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_enrich_rows_preserves_canonical_identity() -> None:
    rows = _canonical_rows()
    before = {
        row["model"]: {
            key: row.get(key)
            for key in backfill.CANONICAL_IDENTITY_FIELDS
        }
        for row in rows
    }

    enriched = backfill.enrich_rows(
        rows,
        real_counts={0: 10, 1: 20},
        synth_counts={0: 5, 1: 5},
        num_classes=9,
        preserve_identity=True,
    )

    for row in enriched:
        assert {
            key: row.get(key)
            for key in backfill.CANONICAL_IDENTITY_FIELDS
        } == before[row["model"]]


def test_enrich_rows_does_not_mutate_input_rows() -> None:
    rows = _canonical_rows()

    enriched = backfill.enrich_rows(
        rows,
        real_counts={0: 10},
        synth_counts={0: 5},
        num_classes=9,
        preserve_identity=True,
    )

    assert rows[0]["counts"] == {}
    assert enriched[0]["counts"] != {}


def test_enrich_rows_preserves_existing_counts() -> None:
    rows = _canonical_rows()
    rows[0]["counts"]["real_per_class"] = {
        "0": 999,
    }

    enriched = backfill.enrich_rows(
        rows,
        real_counts={0: 10},
        synth_counts={0: 5},
        num_classes=9,
        preserve_identity=True,
    )

    assert enriched[0]["counts"][
        "real_per_class"
    ] == {"0": 999}


def test_enrich_rows_adds_legacy_count_aliases() -> None:
    rows = _canonical_rows()

    enriched = backfill.enrich_rows(
        rows,
        real_counts={0: 10, 1: 20},
        synth_counts={0: 5, 1: 7},
        num_classes=9,
        preserve_identity=True,
    )

    counts = enriched[0]["counts"]

    assert counts["real_per_class"] == {
        0: 10,
        1: 20,
    }
    assert counts["synth_per_class"] == {
        0: 5,
        1: 7,
    }
    assert counts[
        "real_plus_synth_per_class"
    ] == {
        0: 15,
        1: 27,
    }


def test_num_classes_clamps_external_counts() -> None:
    rows = _canonical_rows()

    enriched = backfill.enrich_rows(
        rows,
        real_counts={0: 10, 9: 99},
        synth_counts={0: 5, 9: 99},
        num_classes=9,
        preserve_identity=True,
    )

    assert enriched[0]["counts"][
        "real_per_class"
    ] == {0: 10}
    assert enriched[0]["counts"][
        "synth_per_class"
    ] == {0: 5}


def test_write_jsonl_writes_all_rows(
    tmp_path: Path,
) -> None:
    out = tmp_path / "counts.jsonl"

    wrote = backfill.write_jsonl(
        _canonical_rows(),
        out_path=out,
    )

    assert wrote == 7
    assert len(
        out.read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 7


@pytest.mark.parametrize(
    ("canonical", "dataset", "seed", "message"),
    [
        (
            True,
            None,
            42,
            "--dataset is required",
        ),
        (
            True,
            "ustc_tfc2016",
            None,
            "--seed is required",
        ),
        (
            False,
            "ustc_tfc2016",
            42,
            "require --canonical-input",
        ),
    ],
)
def test_mode_validation_fails_closed(
    canonical: bool,
    dataset: str | None,
    seed: int | None,
    message: str,
) -> None:
    args = SimpleNamespace(
        canonical_input=canonical,
        dataset=dataset,
        seed=seed,
    )

    with pytest.raises(
        SystemExit,
        match=message,
    ):
        backfill._validate_mode_args(args)


def test_script_preserves_legacy_mode() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "--canonical-input" in source
    assert "legacy" in source
    assert "--real-root" in source
    assert "--synth-manifest" in source
