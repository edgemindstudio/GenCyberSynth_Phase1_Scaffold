from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDITOR_PATH = REPO_ROOT / "scripts" / "inventory_paper01_execution_evidence.py"
AUDIT_CSV = (
    REPO_ROOT
    / "studies"
    / "paper01_benchmark"
    / "audits"
    / "m6_4_1"
    / "paper01_execution_evidence_inventory.csv"
)


def _load_auditor_module():
    spec = importlib.util.spec_from_file_location(
        "inventory_paper01_execution_evidence",
        AUDITOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load auditor module: {AUDITOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


auditor = _load_auditor_module()


def _accepted_metrics() -> dict[str, float]:
    return {
        "kid": 0.11659309234180974,
        "ms_ssim": 0.5581168315600425,
        "balanced_accuracy": 0.681933740234518,
        "macro_f1": 0.7052081458641456,
        "macro_auprc": 0.75538738708267,
        "generative_precision": 0.7672466571242247,
        "generative_recall": 0.681933740234518,
    }


def _summary_payload(config_sha1: str) -> dict[str, object]:
    return {
        "model": "gaussianmixture",
        "seed": 42,
        "run_id": "gaussianmixture_s42",
        "config_sha1": config_sha1,
        "git_commit": "a9d6317a34e63df1e4ee445b618c29169e7a26e4",
        "generative": {
            "kid": 0.11659309234180974,
            "ms_ssim": 0.5581168315600425,
        },
        "utility_real_only": {
            "macro_f1": 0.7201326336544642,
            "macro_auprc": 0.7655298513722542,
            "bal_acc": 0.6972212486055602,
            "macro_precision": 0.768802671694024,
            "macro_recall": 0.6972212486055602,
        },
        "utility_real_plus_synth": {
            "macro_f1": 0.7052081458641456,
            "macro_auprc": 0.75538738708267,
            "bal_acc": 0.681933740234518,
            "balanced_acc": 0.681933740234518,
            "macro_precision": 0.7672466571242247,
            "macro_recall": 0.681933740234518,
        },
        "metrics.kid": 0.11659309234180974,
        "metrics.ms_ssim": 0.5581168315600425,
        "metrics.downstream.macro_f1": 0.7052081458641456,
        "metrics.downstream.macro_auprc": 0.75538738708267,
        "metrics.downstream.bal_acc": 0.681933740234518,
        "metrics.downstream.precision": 0.7672466571242247,
        "metrics.downstream.recall": 0.681933740234518,
        "metrics.gen_precision": 0.7672466571242247,
        "metrics.gen_recall": 0.681933740234518,
    }


def test_summary_metric_semantics_use_real_plus_synthetic_values():
    flat = auditor.flatten(_summary_payload("0" * 40))

    observed = auditor.summary_metrics(flat)

    assert observed == _accepted_metrics()

    # Guard against the historical bug where leaf-only matching selected
    # utility_real_only values because they appeared first in the JSON.
    assert observed["macro_f1"] != pytest.approx(0.7201326336544642)
    assert observed["balanced_accuracy"] != pytest.approx(0.6972212486055602)


def test_latest_json_is_inventoried_as_alias_not_historical_summary(tmp_path: Path):
    model_root = tmp_path / "gaussianmixture"
    summaries_root = model_root / "summaries"
    summaries_root.mkdir(parents=True)

    config_sha1 = "0d8e51e3e3a54b2816ac72bed5ee001973498cbb"
    payload = _summary_payload(config_sha1)

    timestamped = summaries_root / "summary_20260410_091359.json"
    latest = summaries_root / "latest.json"

    timestamped.write_text(json.dumps(payload), encoding="utf-8")
    latest.write_text(json.dumps(payload), encoding="utf-8")

    log = auditor.LogEvidence(
        path="papers/paper1_phase1_benchmark/logs/slurm_raw/slurm-paper1.246846_15.out",
        mtime_utc="2026-04-10T14:15:39+00:00",
        dataset="cicmaldroid2020",
        family="gaussianmixture",
        seed=42,
        job_id="246862",
        task_id="15",
        host="talon35",
        config_path="configs/paper1_cicmaldroid.yaml",
        artifact_root="/historical/artifacts",
        config_sha1=config_sha1,
        git_commit="a9d6317a34e63df1e4ee445b618c29169e7a26e4",
        do_train=False,
        do_synth=False,
        do_eval=True,
        stages=["eval"],
        summary_paths_reported=[str(timestamped.resolve())],
        identity_source="explicit",
    )

    historical, aliases = auditor.discover_summaries_for_experiment(
        model_root=model_root,
        accepted_metrics=_accepted_metrics(),
        experiment_logs=[log],
        repo_root=tmp_path,
        abs_tol=1e-12,
        rel_tol=1e-9,
    )

    assert len(historical) == 1
    assert Path(historical[0].path).name == "summary_20260410_091359.json"
    assert historical[0].metric_match_status == "exact_all_accepted_metrics"

    assert len(aliases) == 1
    assert Path(aliases[0].path).name == "latest.json"
    assert aliases[0].metric_match_status == "exact_all_accepted_metrics"

    # The compatibility alias must never inflate independent historical
    # evidence count.
    assert len(historical) == 1


def test_synthetic_layouts_preserve_historical_coexistence(tmp_path: Path):
    model_root = tmp_path / "gaussianmixture"
    synthetic_root = model_root / "synthetic"

    seed_directory_file = synthetic_root / "seed42" / "0" / "sample_a.png"
    class_then_seed_file = synthetic_root / "0" / "42" / "sample_b.png"

    seed_directory_file.parent.mkdir(parents=True)
    class_then_seed_file.parent.mkdir(parents=True)

    seed_directory_file.write_bytes(b"seed-layout")
    class_then_seed_file.write_bytes(b"class-layout")

    evidence = auditor.discover_synthetic(
        model_root=model_root,
        seed=42,
        repo_root=tmp_path,
    )

    assert evidence.file_count == 2
    assert evidence.layout == "mixed:class_then_seed,seed_directory"
    assert evidence.layout_counts == {
        "class_then_seed": 1,
        "seed_directory": 1,
    }


def test_output_location_must_stay_inside_repo_and_outside_historical_roots(
    tmp_path: Path,
):
    repo_root = tmp_path / "repo"
    historical_root = tmp_path / "historical"
    valid_output = repo_root / "studies" / "paper01_benchmark" / "audits" / "m6_4_1"

    valid_output.mkdir(parents=True)
    historical_root.mkdir(parents=True)

    auditor.validate_output_location(
        valid_output,
        repo_root,
        [historical_root],
    )

    with pytest.raises(RuntimeError, match="inside the TrustForge repository"):
        auditor.validate_output_location(
            tmp_path / "outside",
            repo_root,
            [historical_root],
        )

    # Simulate a historical root located inside the repository. Even though
    # the path is repository-local, audit output must not be written there.
    repo_historical_root = repo_root / "historical_artifacts"
    forbidden_output = repo_historical_root / "audit"
    forbidden_output.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="must not be inside historical root"):
        auditor.validate_output_location(
            forbidden_output,
            repo_root,
            [repo_historical_root],
        )


def test_committed_m6_4_1_inventory_invariants():
    assert AUDIT_CSV.is_file(), f"Missing M6.4.1 evidence inventory: {AUDIT_CSV}"

    with AUDIT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 42

    assert Counter(
        row["exact_accepted_summary_count"] for row in rows
    ) == {"1": 42}

    assert Counter(
        row["seed_manifest_present"] for row in rows
    ) == {
        "True": 39,
        "False": 3,
    }

    assert Counter(
        row["synthetic_layout"] for row in rows
    ) == {
        "seed_directory": 36,
        "class_then_seed": 3,
        "mixed:class_then_seed,seed_directory": 3,
    }

    missing_seed_manifests = {
        row["experiment_id"]
        for row in rows
        if row["seed_manifest_present"] == "False"
    }

    assert missing_seed_manifests == {
        "cicmaldroid2020_gaussianmixture_b2000_seed42",
        "cicmaldroid2020_gaussianmixture_b2000_seed43",
        "cicmaldroid2020_gaussianmixture_b2000_seed44",
    }

    for row in rows:
        assert row["accepted_score_row_present"] == "True"
        assert int(row["synthetic_file_count"]) > 0
        assert int(row["exact_accepted_summary_count"]) == 1

    # latest.json may survive as an observed compatibility alias, but must
    # not create a second independent accepted historical summary.
    assert Counter(
        row["latest_alias_count"] for row in rows
    ) == {
        "0": 28,
        "1": 14,
    }
