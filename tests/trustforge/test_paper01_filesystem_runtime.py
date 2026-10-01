from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "tools/audit_paper01_filesystem_runtime.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = _load(
    SCRIPT_PATH,
    "trustforge_test_m72b_filesystem_runtime",
)


def _write_manifest(
    tmp_path: Path,
) -> Path:
    image_root = tmp_path / "images"

    records = []
    for class_id in ("0", "1"):
        class_dir = image_root / class_id
        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index in range(2):
            image = class_dir / f"image_{index}.png"
            image.write_bytes(b"fake")
            records.append(
                {
                    "path": str(image),
                    "class_id": class_id,
                }
            )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"images": records}),
        encoding="utf-8",
    )
    return manifest


def test_manifest_records_support_images_schema(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    records = audit._manifest_records(manifest)
    assert len(records) == 4


def test_class_labels_are_extracted(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    assert audit._class_labels(manifest) == {"0", "1"}


def test_top_level_manifest_path_is_required() -> None:
    assert (
        audit._top_level_manifest_path(
            {
                "run_meta": {
                    "manifest_path": "/history/manifest.json"
                }
            }
        )
        is None
    )

    assert audit._top_level_manifest_path(
        {"manifest_path": "/history/manifest.json"}
    ) == Path("/history/manifest.json")


def test_source_does_not_invoke_fallback_helpers() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "_fallback_synth_glob(" not in source
    assert "_fallback_synth_by_class(" not in source

    normalized = source.lower()
    assert "fallback" in normalized
    assert "not scientific authority" in normalized


def test_run_audit_adds_repo_root_for_consumer_imports() -> None:
    source = SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "repo_root_text = str(repo_root)" in source
    assert "sys.path.insert(0, repo_root_text)" in source


def test_direct_cli_import_environment_reaches_audit_logic() -> None:
    # Reproduce the same import environment as:
    #   PYTHONPATH=src python tools/audit_paper01_filesystem_runtime.py ...
    #
    # Use an intentionally invalid dataset so the command exits quickly after
    # canonical loading begins. The important regression check is that it must
    # not fail with "No module named 'scripts'".
    env = {
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PATH": __import__("os").environ.get("PATH", ""),
        "HOME": __import__("os").environ.get("HOME", ""),
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(REPO_ROOT),
            "--dataset",
            "__m72b_invalid_dataset__",
            "--seed",
            "42",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    combined = proc.stdout + proc.stderr
    assert "No module named 'scripts'" not in combined


def test_real_ustc_seed42_runtime_audit() -> None:
    report = audit.run_audit(
        REPO_ROOT,
        dataset="ustc_tfc2016",
        seed=42,
    )

    assert report["summary"]["rows"] == 7
    assert report["summary"]["models"] == 7
    assert report["status"] in {"PASS", "FAIL"}


def test_real_cic_seed42_runtime_audit() -> None:
    report = audit.run_audit(
        REPO_ROOT,
        dataset="cicmaldroid2020",
        seed=42,
    )

    assert report["summary"]["rows"] == 7
    assert report["summary"]["models"] == 7
    assert report["status"] in {"PASS", "FAIL"}


def test_reports_write_to_requested_directory(
    tmp_path: Path,
) -> None:
    report = {
        "audit": "M7.2B",
        "dataset": "ustc_tfc2016",
        "seed": 42,
        "status": "PASS",
        "summary": {
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "rows": 7,
            "models": 7,
        },
        "checks": [],
        "models": [],
        "authority_statement": "test",
    }

    json_path, md_path = audit.write_reports(
        report,
        tmp_path,
    )

    assert json_path.parent == tmp_path
    assert md_path.parent == tmp_path
    assert json_path.is_file()
    assert md_path.is_file()
