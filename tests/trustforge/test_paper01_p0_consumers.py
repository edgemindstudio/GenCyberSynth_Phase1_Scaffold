from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PATH = REPO_ROOT / "tools/build_phase1_scores.py"
CHECK_PATH = REPO_ROOT / "tools/check_phase1_integrity.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = _load(BUILD_PATH, "trustforge_test_build_phase1_scores")
check = _load(CHECK_PATH, "trustforge_test_check_phase1_integrity")


def _summary(
    path: Path,
    *,
    family: str,
    seed: int,
    budget: int,
    num_fake: int,
) -> Path:
    payload = {
        "model": family,
        "seed": seed,
        "budget_per_class": budget,
        "counts": {
            "num_fake": num_fake,
        },
        "run_meta": {
            "config_path": f"configs/{family}.yaml",
            "config_sha1": "abc123",
            "git_commit": "deadbeef",
        },
        "metrics": {
            "kid": 0.1,
            "cfid": 1.2,
            "ms_ssim": 0.3,
            "fid": 4.5,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(
    *,
    tmp_path: Path,
    dataset: str,
    family: str,
    seed: int,
    budget: int = 2000,
    num_fake: int,
):
    summary_path = _summary(
        tmp_path / f"summary_{family}_{seed}.json",
        family=family,
        seed=seed,
        budget=budget,
        num_fake=num_fake,
    )
    return SimpleNamespace(
        experiment_id=f"{dataset}_{family}_b{budget}_seed{seed}",
        dataset=dataset,
        family=family,
        seed=seed,
        budget_per_class=budget,
        historical_run_id=f"{family}_s{seed}",
        accepted_summary_path=summary_path,
    )


def _view(dataset: str, seed: int, records):
    return SimpleNamespace(
        dataset=dataset,
        seed=seed,
        records=tuple(records),
        __iter__=lambda self: iter(self.records),
    )


class FakeView:
    def __init__(self, dataset: str, seed: int, records):
        self.dataset = dataset
        self.seed = seed
        self.records = tuple(records)

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)


def test_build_requires_explicit_dataset() -> None:
    with pytest.raises(SystemExit):
        build.parse_args(
            ["--seed", "42"],
            environ={},
        )


def test_build_requires_explicit_seed() -> None:
    with pytest.raises(SystemExit):
        build.parse_args(
            ["--dataset", "ustc_tfc2016"],
            environ={},
        )


def test_check_requires_explicit_dataset() -> None:
    with pytest.raises(SystemExit):
        check.parse_args(
            ["--seed", "42"],
            environ={},
        )


def test_check_requires_explicit_seed() -> None:
    with pytest.raises(SystemExit):
        check.parse_args(
            ["--dataset", "ustc_tfc2016"],
            environ={},
        )


def test_environment_identity_is_supported() -> None:
    args = build.parse_args(
        [],
        environ={
            "PHASE1_DATASET": "ustc_tfc2016",
            "PHASE1_SEED": "43",
        },
    )

    assert args.dataset == "ustc_tfc2016"
    assert args.seed == 43


def test_cli_identity_overrides_environment() -> None:
    args = build.parse_args(
        [
            "--dataset",
            "cicmaldroid2020",
            "--seed",
            "44",
        ],
        environ={
            "PHASE1_DATASET": "ustc_tfc2016",
            "PHASE1_SEED": "42",
        },
    )

    assert args.dataset == "cicmaldroid2020"
    assert args.seed == 44


def test_trustforge_artifacts_root_is_read_from_environment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trustforge_artifacts"

    args = build.parse_args(
        [],
        environ={
            "PHASE1_DATASET": "ustc_tfc2016",
            "PHASE1_SEED": "42",
            "TRUSTFORGE_ARTIFACTS_ROOT": str(root),
        },
    )

    assert args.trustforge_artifacts_root == root


def test_no_implicit_legacy_artifact_root_when_trustforge_root_missing() -> None:
    args = build.parse_args(
        [],
        environ={
            "PHASE1_DATASET": "ustc_tfc2016",
            "PHASE1_SEED": "42",
        },
    )

    assert args.trustforge_artifacts_root is None


def test_default_output_is_namespaced_by_dataset_and_seed(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "trustforge_artifacts"

    out = build.default_output_path(
        artifacts_root,
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert out == (
        artifacts_root
        / "paper01/ustc_tfc2016/seed42/phase1_scores.csv"
    )


def test_default_output_does_not_use_legacy_shared_filename_root(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "trustforge_artifacts"

    out = build.default_output_path(
        artifacts_root,
        dataset="cicmaldroid2020",
        seed=43,
    )

    assert "artifacts/phase1_scores.csv" not in str(out)
    assert out.is_relative_to(artifacts_root)


def test_build_row_uses_canonical_identity(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path=tmp_path,
        dataset="ustc_tfc2016",
        family="gan",
        seed=42,
        num_fake=18000,
    )

    rows = build.build_rows(
        FakeView("ustc_tfc2016", 42, [record])
    )

    assert rows == [
        {
            "experiment_id": "ustc_tfc2016_gan_b2000_seed42",
            "dataset": "ustc_tfc2016",
            "model": "gan",
            "seed": 42,
            "budget_per_class": 2000,
            "num_fake": 18000,
            "config_path": "configs/gan.yaml",
            "config_sha1": "abc123",
            "git_commit": "deadbeef",
            "kid": 0.1,
            "cfid": 1.2,
            "ms_ssim": 0.3,
            "fid": 4.5,
            "accepted_summary_path": str(record.accepted_summary_path),
        }
    ]


def test_build_rejects_summary_seed_mismatch(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path=tmp_path,
        dataset="ustc_tfc2016",
        family="gan",
        seed=42,
        num_fake=18000,
    )
    data = json.loads(record.accepted_summary_path.read_text())
    data["seed"] = 43
    record.accepted_summary_path.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match="summary seed=43"):
        build.build_rows(
            FakeView("ustc_tfc2016", 42, [record])
        )


def test_build_rejects_summary_family_mismatch(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path=tmp_path,
        dataset="ustc_tfc2016",
        family="gan",
        seed=42,
        num_fake=18000,
    )
    data = json.loads(record.accepted_summary_path.read_text())
    data["model"] = "vae"
    record.accepted_summary_path.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match="summary model='vae'"):
        build.build_rows(
            FakeView("ustc_tfc2016", 42, [record])
        )


def test_build_rejects_summary_budget_mismatch(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path=tmp_path,
        dataset="ustc_tfc2016",
        family="gan",
        seed=42,
        num_fake=18000,
    )
    data = json.loads(record.accepted_summary_path.read_text())
    data["budget_per_class"] = 1000
    record.accepted_summary_path.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match="summary budget_per_class=1000"):
        build.build_rows(
            FakeView("ustc_tfc2016", 42, [record])
        )


def test_build_refuses_latest_alias(
    tmp_path: Path,
) -> None:
    path = tmp_path / "latest.json"
    path.write_text("{}")
    record = SimpleNamespace(
        experiment_id="x",
        dataset="ustc_tfc2016",
        family="gan",
        seed=42,
        budget_per_class=2000,
        accepted_summary_path=path,
    )

    with pytest.raises(RuntimeError, match="latest.json"):
        build._load_accepted_summary(record)


def test_build_csv_is_lf_only(
    tmp_path: Path,
) -> None:
    out = tmp_path / "scores.csv"
    build.write_csv(
        out,
        [
            {
                "experiment_id": "x",
                "dataset": "ustc_tfc2016",
                "model": "gan",
            }
        ],
    )

    raw = out.read_bytes()

    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 2


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("ustc_tfc2016", 9),
        ("cicmaldroid2020", 5),
    ],
)
def test_integrity_class_count_contract(
    dataset: str,
    expected: int,
) -> None:
    assert check.NUM_CLASSES_BY_DATASET[dataset] == expected


@pytest.mark.parametrize(
    ("dataset", "num_fake"),
    [
        ("ustc_tfc2016", 18000),
        ("cicmaldroid2020", 10000),
    ],
)
def test_integrity_accepts_dataset_specific_counts(
    tmp_path: Path,
    dataset: str,
    num_fake: int,
) -> None:
    record = _record(
        tmp_path=tmp_path,
        dataset=dataset,
        family="gan",
        seed=42,
        num_fake=num_fake,
    )

    issues = check.check_view(
        FakeView(dataset, 42, [record])
    )

    assert issues == []


@pytest.mark.parametrize(
    ("dataset", "wrong_num_fake", "expected_text"),
    [
        ("ustc_tfc2016", 10000, "expected=18000"),
        ("cicmaldroid2020", 18000, "expected=10000"),
    ],
)
def test_integrity_rejects_wrong_dataset_specific_counts(
    tmp_path: Path,
    dataset: str,
    wrong_num_fake: int,
    expected_text: str,
) -> None:
    record = _record(
        tmp_path=tmp_path,
        dataset=dataset,
        family="gan",
        seed=42,
        num_fake=wrong_num_fake,
    )

    issues = check.check_view(
        FakeView(dataset, 42, [record])
    )

    assert len(issues) == 1
    assert expected_text in issues[0]


def test_integrity_rejects_missing_num_fake(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path=tmp_path,
        dataset="ustc_tfc2016",
        family="gan",
        seed=42,
        num_fake=18000,
    )
    data = json.loads(record.accepted_summary_path.read_text())
    del data["counts"]["num_fake"]
    record.accepted_summary_path.write_text(json.dumps(data))

    issues = check.check_view(
        FakeView("ustc_tfc2016", 42, [record])
    )

    assert any("missing summary num_fake" in issue for issue in issues)


def test_integrity_rejects_missing_budget(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path=tmp_path,
        dataset="ustc_tfc2016",
        family="gan",
        seed=42,
        num_fake=18000,
    )
    data = json.loads(record.accepted_summary_path.read_text())
    del data["budget_per_class"]
    record.accepted_summary_path.write_text(json.dumps(data))

    issues = check.check_view(
        FakeView("ustc_tfc2016", 42, [record])
    )

    assert any(
        "missing summary budget_per_class" in issue
        for issue in issues
    )


def test_integrity_rejects_latest_alias(
    tmp_path: Path,
) -> None:
    path = tmp_path / "latest.json"
    path.write_text("{}")
    record = SimpleNamespace(
        experiment_id="x",
        dataset="ustc_tfc2016",
        family="gan",
        seed=42,
        budget_per_class=2000,
        accepted_summary_path=path,
    )

    issues = check._check_record(
        record,
        num_classes=9,
    )

    assert len(issues) == 1
    assert "latest.json" in issues[0]


def test_integrity_rejects_unknown_dataset() -> None:
    with pytest.raises(RuntimeError, match="no Paper 1 class-count contract"):
        check.check_view(
            FakeView("unknown", 42, [])
        )


def test_p0_sources_do_not_reference_legacy_summary_selector() -> None:
    build_source = BUILD_PATH.read_text()
    check_source = CHECK_PATH.read_text()

    assert "PHASE1_SUMMARY_NAME" not in build_source
    assert "PHASE1_SUMMARY_NAME" not in check_source


def test_p0_sources_do_not_glob_local_summary_aliases() -> None:
    build_source = BUILD_PATH.read_text()
    check_source = CHECK_PATH.read_text()

    forbidden_code_tokens = (
        'Path("artifacts").glob',
        'glob("*/summaries/',
        "glob('*/summaries/",
        'SUMMARY_NAME =',
        'os.environ.get("PHASE1_SUMMARY_NAME"',
        "os.environ.get('PHASE1_SUMMARY_NAME'",
    )

    for token in forbidden_code_tokens:
        assert token not in build_source
        assert token not in check_source


def test_p0_sources_use_canonical_consumer_view() -> None:
    build_source = BUILD_PATH.read_text()
    check_source = CHECK_PATH.read_text()

    needle = "load_paper01_consumer_view"

    assert needle in build_source
    assert needle in check_source
