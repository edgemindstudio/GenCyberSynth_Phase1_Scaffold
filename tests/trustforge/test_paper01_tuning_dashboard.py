from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts/tuning_dashboard.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    import sys
    sys.modules[name] = module

    spec.loader.exec_module(module)
    return module


dashboard = _load(
    SCRIPT_PATH,
    "trustforge_test_tuning_dashboard",
)


class FakeRecord:
    def __init__(
        self,
        *,
        family: str,
        seed: int = 42,
        budget: int = 2000,
        source_name: str = "summary_20260404_161124.json",
    ):
        self.family = family
        self.seed = seed
        self.budget_per_class = budget
        self.experiment_id = (
            f"paper01-{family}-seed{seed}"
        )
        self.historical_run_id = (
            f"{family}-historical-{seed}"
        )
        self.accepted_summary_path = (
            f"/history/{family}/summaries/"
            f"{source_name}"
        )


def _fake_records(
    *,
    seed: int = 42,
):
    return tuple(
        FakeRecord(
            family=family,
            seed=seed,
        )
        for family in dashboard.PAPER01_FAMILIES
    )


def _install_fake_exports(
    monkeypatch: pytest.MonkeyPatch,
    records,
) -> None:
    class FakeView:
        def __init__(self, records):
            self.records = records

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
            assert seed in {42, 43}
            selected = tuple(
                FakeRecord(
                    family=record.family,
                    seed=seed,
                )
                for record in records
            )
            return FakeView(selected)

    import sys

    monkeypatch.setitem(
        sys.modules,
        "trustforge.paper01_exports",
        FakeExports,
    )


def test_load_canonical_seed_delegates_to_export_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _fake_records()
    _install_fake_exports(
        monkeypatch,
        records,
    )

    rows = dashboard.load_canonical_seed(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert len(rows) == 7
    assert all(
        row.accepted
        for row in rows
    )


def test_canonical_seed_rejects_alias_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = list(_fake_records())
    records[0] = FakeRecord(
        family=records[0].family,
        source_name="latest.json",
    )

    class FakeView:
        def __init__(self, records):
            self.records = records

    class FakeExports:
        @staticmethod
        def load_paper01_export_view(
            repo_root,
            *,
            dataset,
            seed,
        ):
            return FakeView(tuple(records))

    import sys
    monkeypatch.setitem(
        sys.modules,
        "trustforge.paper01_exports",
        FakeExports,
    )

    with pytest.raises(
        dashboard.CanonicalDashboardError,
        match="alias source",
    ):
        dashboard.load_canonical_seed(
            Path("/repo"),
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_canonical_seed_rejects_budget_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = list(_fake_records())
    records[0] = FakeRecord(
        family=records[0].family,
        budget=100,
    )

    class FakeView:
        def __init__(self, records):
            self.records = records

    class FakeExports:
        @staticmethod
        def load_paper01_export_view(
            repo_root,
            *,
            dataset,
            seed,
        ):
            return FakeView(tuple(records))

    import sys
    monkeypatch.setitem(
        sys.modules,
        "trustforge.paper01_exports",
        FakeExports,
    )

    with pytest.raises(
        dashboard.CanonicalDashboardError,
        match="budget must be 2000",
    ):
        dashboard.load_canonical_seed(
            Path("/repo"),
            dataset="ustc_tfc2016",
            seed=42,
        )


def test_canonical_status_rows_support_multiple_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _fake_records()
    _install_fake_exports(
        monkeypatch,
        records,
    )

    rows = dashboard.canonical_status_rows(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seeds=[42, 43],
        models=None,
    )

    assert len(rows) == 14
    assert {
        row.seed
        for row in rows
    } == {42, 43}


def test_canonical_status_rows_can_filter_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_exports(
        monkeypatch,
        _fake_records(),
    )

    rows = dashboard.canonical_status_rows(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seeds=[42],
        models=["gan", "vae"],
    )

    assert [
        row.model
        for row in rows
    ] == ["gan", "vae"]


def test_canonical_status_rows_reject_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        dashboard.CanonicalDashboardError,
        match="unknown canonical",
    ):
        dashboard.canonical_status_rows(
            Path("/repo"),
            dataset="ustc_tfc2016",
            seeds=[42],
            models=["not-a-family"],
        )


def test_canonical_mode_never_calls_latest_summary_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_exports(
        monkeypatch,
        _fake_records(),
    )

    def fail(*args, **kwargs):
        raise AssertionError(
            "newest-summary discovery "
            "must not run in canonical mode"
        )

    monkeypatch.setattr(
        dashboard,
        "_find_latest_summary_for_model",
        fail,
    )

    rows = dashboard.canonical_status_rows(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seeds=[42],
        models=None,
    )

    assert len(rows) == 7


def test_canonical_mode_does_not_require_manifest_or_done_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_exports(
        monkeypatch,
        _fake_records(),
    )

    rows = dashboard.canonical_status_rows(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seeds=[42],
        models=None,
    )

    assert len(rows) == 7
    assert all(
        row.accepted
        for row in rows
    )


def test_canonical_dashboard_labels_scientific_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_exports(
        monkeypatch,
        _fake_records(),
    )

    rows = dashboard.canonical_status_rows(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seeds=[42],
        models=None,
    )

    dashboard.print_canonical_dashboard(
        rows,
        dataset="ustc_tfc2016",
        seeds=[42],
        models=None,
        wide=False,
    )

    output = capsys.readouterr().out

    assert "accepted scientific evidence" in output
    assert "operational manifests/done flags" in output
    assert "canonical-paper01" in output


def test_legacy_status_rows_preserve_operational_manifest_done_behavior(
    tmp_path: Path,
) -> None:
    model = "gan"
    cfg = "A"
    seed = 42

    manifest = (
        tmp_path
        / model
        / "synthetic"
        / f"{model}_{cfg}_seed{seed}"
        / "manifest.json"
    )
    manifest.parent.mkdir(
        parents=True
    )
    manifest.write_text(
        "{}",
        encoding="utf-8",
    )

    summaries = (
        tmp_path
        / model
        / "summaries"
    )
    summaries.mkdir(
        parents=True
    )

    done = (
        summaries
        / f"done_{model}_{cfg}_seed{seed}.txt"
    )
    done.write_text(
        "done",
        encoding="utf-8",
    )

    summary = (
        summaries
        / "summary_20260101_000000.json"
    )
    summary.write_text(
        json.dumps(
            {
                "run_id": "legacy-run",
                "run_meta": {
                    "manifest_path": str(manifest),
                    "budget_per_class": 2000,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = dashboard.legacy_status_rows(
        artifacts=tmp_path,
        models=[model],
        cfgs=[cfg],
        seeds=[seed],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.has_manifest
    assert row.has_done
    assert row.has_summary
    assert row.run_id == "legacy-run"
    assert row.budget_per_class == 2000


@pytest.mark.parametrize(
    (
        "canonical",
        "dataset",
        "artifacts",
        "cfgs",
        "show_missing",
        "models",
        "message",
    ),
    [
        (
            True,
            None,
            None,
            None,
            False,
            None,
            "--dataset is required",
        ),
        (
            True,
            "ustc_tfc2016",
            "/legacy",
            None,
            False,
            None,
            "--artifacts is legacy-only",
        ),
        (
            True,
            "ustc_tfc2016",
            None,
            ["A"],
            False,
            None,
            "--cfgs is legacy-only",
        ),
        (
            True,
            "ustc_tfc2016",
            None,
            None,
            True,
            None,
            "--show-missing-only is legacy-only",
        ),
        (
            False,
            "ustc_tfc2016",
            "/legacy",
            None,
            False,
            ["gan"],
            "--dataset requires",
        ),
        (
            False,
            None,
            None,
            None,
            False,
            ["gan"],
            "--artifacts is required",
        ),
        (
            False,
            None,
            "/legacy",
            None,
            False,
            None,
            "--models is required",
        ),
    ],
)
def test_mode_validation_fails_closed(
    canonical,
    dataset,
    artifacts,
    cfgs,
    show_missing,
    models,
    message,
) -> None:
    args = SimpleNamespace(
        canonical_paper01=canonical,
        dataset=dataset,
        artifacts=artifacts,
        cfgs=cfgs,
        show_missing_only=show_missing,
        models=models,
    )

    with pytest.raises(
        SystemExit,
        match=message,
    ):
        dashboard._validate_args(args)


def test_source_separates_operational_and_scientific_authority() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "--canonical-paper01" in source
    assert "load_paper01_export_view" in source
    assert (
        "OPERATIONAL COMPLETION STATUS != ACCEPTED SCIENTIFIC EVIDENCE"
        in source
    )
    assert "_find_latest_summary_for_model" in source
