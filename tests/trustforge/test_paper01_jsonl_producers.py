from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARIES_PATH = REPO_ROOT / "scripts/summaries_to_jsonl.py"
BUILD_TOOL_PATH = REPO_ROOT / "tools/build_paper1_jsonl.py"
BUILD_SH_PATH = REPO_ROOT / "scripts/build_jsonl.sh"


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


summaries = _load(
    SUMMARIES_PATH,
    "trustforge_test_summaries_to_jsonl",
)
build_tool = _load(
    BUILD_TOOL_PATH,
    "trustforge_test_build_paper1_jsonl",
)


def test_summaries_script_has_canonical_mode() -> None:
    source = SUMMARIES_PATH.read_text(
        encoding="utf-8"
    )

    assert "--canonical" in source
    assert "--dataset" in source
    assert "--seed" in source
    assert "load_paper01_export_view" in source
    assert "paper01_export_view_to_legacy_jsonl" in source


def test_canonical_rows_delegate_to_trustforge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_view = object()
    expected_rows = [{"model": "gan"}]

    class FakeExports:
        @staticmethod
        def load_paper01_export_view(
            repo_root,
            *,
            dataset,
            seed,
        ):
            assert Path(repo_root) == Path("/repo")
            assert dataset == "ustc_tfc2016"
            assert seed == 42
            return fake_view

    class FakeCompat:
        @staticmethod
        def paper01_export_view_to_legacy_jsonl(view):
            assert view is fake_view
            return tuple(expected_rows)

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

    rows = summaries.canonical_rows(
        Path("/repo"),
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert rows == expected_rows


@pytest.mark.parametrize(
    ("canonical", "dataset", "seed", "message"),
    [
        (True, None, 42, "--dataset is required"),
        (True, "ustc_tfc2016", None, "--seed is required"),
        (False, "ustc_tfc2016", 42, "require --canonical"),
    ],
)
def test_summaries_identity_validation_fails_closed(
    canonical,
    dataset,
    seed,
    message,
) -> None:
    with pytest.raises(
        SystemExit,
        match=message,
    ):
        summaries._validate_identity_args(
            canonical=canonical,
            dataset=dataset,
            seed=seed,
        )


def test_write_rows_is_idempotent_by_source_path(
    tmp_path: Path,
) -> None:
    out = tmp_path / "phase1.jsonl"
    rows = [
        {
            "model": "gan",
            "source_path": "/history/summary_1.json",
        }
    ]

    first = summaries.write_rows(
        rows,
        out_path=out,
        reset=False,
    )
    second = summaries.write_rows(
        rows,
        out_path=out,
        reset=False,
    )

    assert first == 1
    assert second == 0
    assert len(out.read_text().splitlines()) == 1


def test_write_rows_reset_replaces_existing_output(
    tmp_path: Path,
) -> None:
    out = tmp_path / "phase1.jsonl"
    out.write_text(
        json.dumps(
            {
                "model": "old",
                "source_path": "/old.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    written = summaries.write_rows(
        [
            {
                "model": "gan",
                "source_path": "/new.json",
            }
        ],
        out_path=out,
        reset=True,
    )

    assert written == 1

    payload = [
        json.loads(line)
        for line in out.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert payload == [
        {
            "model": "gan",
            "source_path": "/new.json",
        }
    ]


def test_build_tool_import_has_no_side_effect_output(
    tmp_path: Path,
) -> None:
    assert not (
        tmp_path / "phase1_summaries.jsonl"
    ).exists()


def test_build_tool_identity_requires_dataset_and_seed_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PHASE1_DATASET",
        "ustc_tfc2016",
    )
    monkeypatch.delenv(
        "PHASE1_SEED",
        raising=False,
    )

    with pytest.raises(
        SystemExit,
        match="must be provided together",
    ):
        build_tool._identity_from_env()


def test_build_tool_canonical_rows_delegate_to_trustforge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_view = object()

    class FakeExports:
        @staticmethod
        def load_paper01_export_view(
            repo_root,
            *,
            dataset,
            seed,
        ):
            assert repo_root == Path("/repo")
            assert dataset == "cicmaldroid2020"
            assert seed == 44
            return fake_view

    class FakeCompat:
        @staticmethod
        def paper01_export_view_to_legacy_jsonl(view):
            assert view is fake_view
            return (
                {"model": "gan"},
                {"model": "vae"},
            )

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

    rows = build_tool.canonical_rows(
        Path("/repo"),
        dataset="cicmaldroid2020",
        seed=44,
    )

    assert rows == [
        {"model": "gan"},
        {"model": "vae"},
    ]


def test_build_tool_write_jsonl_overwrites_deterministically(
    tmp_path: Path,
) -> None:
    out = tmp_path / "phase1.jsonl"
    out.write_text(
        "old\n",
        encoding="utf-8",
    )

    build_tool.write_jsonl(
        [
            {"model": "gan", "seed": 42},
            {"model": "vae", "seed": 42},
        ],
        out_path=out,
    )

    rows = [
        json.loads(line)
        for line in out.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert rows == [
        {"model": "gan", "seed": 42},
        {"model": "vae", "seed": 42},
    ]


def test_build_wrapper_exposes_canonical_and_legacy_paths() -> None:
    source = BUILD_SH_PATH.read_text(
        encoding="utf-8"
    )

    assert "PHASE1_DATASET" in source
    assert "PHASE1_SEED" in source
    assert "--canonical" in source
    assert "legacy compatibility mode" in source
    assert "phase1-artifacts-raw" in source


def test_build_wrapper_partial_identity_fails_closed(
    tmp_path: Path,
) -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "PHASE1_DATASET": "ustc_tfc2016",
    }

    result = subprocess.run(
        ["bash", str(BUILD_SH_PATH)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "PHASE1_SEED is required" in result.stderr


def test_canonical_sources_do_not_use_latest_alias_as_authority() -> None:
    summaries_source = SUMMARIES_PATH.read_text(
        encoding="utf-8"
    )
    wrapper_source = BUILD_SH_PATH.read_text(
        encoding="utf-8"
    )
    tool_source = BUILD_TOOL_PATH.read_text(
        encoding="utf-8"
    )

    forbidden_executable_patterns = (
        'glob("*/summaries/latest.json")',
        "glob('*/summaries/latest.json')",
        'glob(f"*/summaries/latest.json")',
        "glob(f'*/summaries/latest.json')",
        'Path("artifacts").glob("*/summaries/latest.json")',
        "Path('artifacts').glob('*/summaries/latest.json')",
        'os.getenv("PHASE1_SUMMARY_NAME", "latest.json")',
        "os.getenv('PHASE1_SUMMARY_NAME', 'latest.json')",
        'SUMMARY_NAME = "latest.json"',
        "SUMMARY_NAME = 'latest.json'",
    )

    combined = (
        summaries_source
        + wrapper_source
        + tool_source
    )

    for pattern in forbidden_executable_patterns:
        assert pattern not in combined

    assert "load_paper01_export_view" in summaries_source
    assert "load_paper01_export_view" in tool_source


def test_build_tool_preserves_legacy_paper1_snapshot_path() -> None:
    source = BUILD_TOOL_PATH.read_text(
        encoding="utf-8"
    )

    assert '"paper1.json"' in source
    assert "mode=legacy_snapshot" in source


def test_legacy_summaries_mode_remains_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    path = (
        tmp_path
        / "artifacts"
        / "gan"
        / "summaries"
        / "summary_20260404_161124.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "model": "gan",
                "seed": 42,
                "metrics": {"kid": 0.1},
            }
        ),
        encoding="utf-8",
    )

    rows = summaries.legacy_rows(
        "artifacts/*/summaries/summary_*.json"
    )

    assert len(rows) == 1
    assert rows[0]["model"] == "gan"
    assert rows[0]["seed"] == 42
    assert rows[0]["source_path"].endswith(
        "summary_20260404_161124.json"
    )
