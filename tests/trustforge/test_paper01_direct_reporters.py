from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CFID_PATH = REPO_ROOT / "scripts/metrics/print_cfid_table.py"
HTML_PATH = REPO_ROOT / "scripts/phase1_html.py"
REPORT_PATH = REPO_ROOT / "scripts/phase1_report.py"


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


cfid = _load(CFID_PATH, "trustforge_test_cfid_reporter")
html_report = _load(HTML_PATH, "trustforge_test_html_reporter")
md_report = _load(REPORT_PATH, "trustforge_test_md_reporter")


def _records():
    families = (
        "autoregressive",
        "diffusion",
        "gan",
        "gaussianmixture",
        "maskedautoflow",
        "restrictedboltzmann",
        "vae",
    )

    return [
        {
            "model": family,
            "seed": 42,
            "budget_per_class": 2000,
            "source_path": (
                f"/history/{family}/summaries/"
                "summary_20260404_161124.json"
            ),
            "cfid": float(index + 1),
            "kid": float(index + 1) / 10.0,
            "ms_ssim": float(index + 1) / 100.0,
            "num_fake": 18000,
        }
        for index, family in enumerate(families)
    ]


def _install_fake_trustforge(
    monkeypatch: pytest.MonkeyPatch,
):
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

    return records


@pytest.mark.parametrize(
    "module",
    [
        cfid,
        html_report,
        md_report,
    ],
)
def test_canonical_reporters_delegate_to_trustforge(
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_trustforge(monkeypatch)

    if module is cfid:
        rows = module.canonical_table(
            Path("/repo"),
            dataset="ustc_tfc2016",
            seed=42,
        )
    else:
        rows = module.canonical_rows(
            Path("/repo"),
            dataset="ustc_tfc2016",
            seed=42,
        )

    assert len(rows) == 7


@pytest.mark.parametrize(
    "module",
    [
        cfid,
        html_report,
        md_report,
    ],
)
def test_canonical_reporters_reject_alias_authority(
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = list(_install_fake_trustforge(monkeypatch))

    records[0] = dict(records[0])
    records[0]["source_path"] = "/history/latest.json"

    class FakeCompat:
        @staticmethod
        def paper01_export_view_to_legacy_jsonl(view):
            return tuple(records)

    import sys
    monkeypatch.setitem(
        sys.modules,
        "trustforge.paper01_compat",
        FakeCompat,
    )

    with pytest.raises(
        RuntimeError,
        match="alias source",
    ):
        if module is cfid:
            module.canonical_table(
                Path("/repo"),
                dataset="ustc_tfc2016",
                seed=42,
            )
        else:
            module.canonical_rows(
                Path("/repo"),
                dataset="ustc_tfc2016",
                seed=42,
            )


@pytest.mark.parametrize(
    "module",
    [
        cfid,
        html_report,
        md_report,
    ],
)
def test_partial_identity_fails_closed(
    module,
) -> None:
    args = SimpleNamespace(
        canonical=True,
        dataset="ustc_tfc2016",
        seed=None,
    )

    with pytest.raises(
        SystemExit,
        match="--seed is required",
    ):
        module._validate_args(args)


def test_cfid_table_uses_compatibility_metric_shims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_trustforge(monkeypatch)

    rows = cfid.canonical_table(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert rows[0][1] == 1.0
    assert rows[0][2] == 0.1
    assert rows[0][3] == 0.01


def test_markdown_report_renders_seven_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_trustforge(monkeypatch)

    rows = md_report.canonical_rows(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seed=42,
    )
    text = md_report.render_markdown(rows)

    for record in _records():
        assert record["model"] in text


def test_html_report_renders_seven_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_trustforge(monkeypatch)

    rows = html_report.canonical_rows(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seed=42,
    )
    text = html_report.render_html(
        rows,
        grid_root=None,
    )

    assert text.count("<tr>") == 8
    for record in _records():
        assert record["model"] in text


def test_markdown_legacy_latest_summary_behavior_remains(
    tmp_path: Path,
) -> None:
    sdir = tmp_path / "gan" / "summaries"
    sdir.mkdir(parents=True)
    (sdir / "summary_20260101_000000.json").write_text(
        "{}",
        encoding="utf-8",
    )
    newer = sdir / "summary_20260102_000000.json"
    newer.write_text(
        "{}",
        encoding="utf-8",
    )

    assert md_report.latest_summary(
        tmp_path,
        "gan",
    ) == newer


def test_html_legacy_latest_summary_behavior_remains(
    tmp_path: Path,
) -> None:
    sdir = tmp_path / "vae" / "summaries"
    sdir.mkdir(parents=True)
    (sdir / "summary_20260101_000000.json").write_text(
        "{}",
        encoding="utf-8",
    )
    newer = sdir / "summary_20260102_000000.json"
    newer.write_text(
        "{}",
        encoding="utf-8",
    )

    assert html_report.latest_summary(
        tmp_path,
        "vae",
    ) == newer


def test_reporter_sources_expose_canonical_mode() -> None:
    for path in (
        CFID_PATH,
        HTML_PATH,
        REPORT_PATH,
    ):
        source = path.read_text(
            encoding="utf-8"
        )
        assert "--canonical" in source
        assert "load_paper01_export_view" in source
        assert "paper01_export_view_to_legacy_jsonl" in source
