#!/usr/bin/env python3
"""
M7.2B — Filesystem-coupled runtime compatibility audit.

Read-only audit for the two P2 consumers that cross from canonical-derived
JSONL into historical manifest/filesystem state:

- scripts/plots/imbalance/simple_stats_sanity.py
- scripts/plots/qual/class_triptychs.py

The audit intentionally does NOT invoke either consumer's filesystem fallback.
Canonical authority must come from the accepted Paper 1 row's manifest_path.

Checks for one explicit dataset+seed:
- exactly seven canonical-derived rows;
- each row has a top-level manifest_path field, because both consumers require
  a dataframe column literally named "manifest_path";
- manifest_path exists and is a readable JSON file;
- the simple-stats manifest parser yields at least one existing synthetic image;
- the class-triptych manifest parser yields at least one existing synthetic
  image for every class represented in the manifest;
- the consumer's own _manifest_from_jsonl helper resolves the exact canonical
  manifest when given one canonical row per model;
- no fallback glob is needed or invoked by this audit.

This audit never mutates historical manifests or images.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

from trustforge.paper01_compat import (
    paper01_export_view_to_legacy_jsonl,
)
from trustforge.paper01_exports import (
    load_paper01_export_view,
)


PAPER01_FAMILIES = {
    "autoregressive",
    "diffusion",
    "gan",
    "gaussianmixture",
    "maskedautoflow",
    "restrictedboltzmann",
    "vae",
}

SIMPLE_STATS_REL = Path(
    "scripts/plots/imbalance/simple_stats_sanity.py"
)
CLASS_TRIPTYCHS_REL = Path(
    "scripts/plots/qual/class_triptychs.py"
)


class FilesystemCompatibilityAuditError(RuntimeError):
    """Raised when the runtime compatibility audit cannot be performed."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _load_module(
    path: Path,
    name: str,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    if spec is None or spec.loader is None:
        raise FilesystemCompatibilityAuditError(
            f"cannot import module from {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_records(
    manifest_path: Path,
) -> list[Any]:
    try:
        data = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except FileNotFoundError as exc:
        raise FilesystemCompatibilityAuditError(
            f"manifest missing: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FilesystemCompatibilityAuditError(
            f"manifest invalid JSON: {manifest_path}"
        ) from exc

    if isinstance(data, list):
        return list(data)

    if isinstance(data, dict):
        for key in (
            "images",
            "paths",
            "samples",
        ):
            value = data.get(key)
            if isinstance(value, list):
                return list(value)

    return []


def _class_labels(
    manifest_path: Path,
) -> set[str]:
    labels: set[str] = set()

    for record in _manifest_records(
        manifest_path
    ):
        if isinstance(record, Mapping):
            for key in (
                "class",
                "class_id",
                "label",
            ):
                if (
                    key in record
                    and record[key] is not None
                ):
                    labels.add(
                        str(record[key]).strip()
                    )
                    break
            continue

        if isinstance(record, str):
            p = Path(record)
            if p.parent.name:
                labels.add(
                    p.parent.name
                )

    return {
        label
        for label in labels
        if label
    }


def _top_level_manifest_path(
    row: Mapping[str, Any],
) -> Path | None:
    value = row.get("manifest_path")
    if not isinstance(value, str):
        return None
    if not value.strip():
        return None
    return Path(value)


def load_rows(
    repo_root: Path,
    *,
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    export_view = (
        load_paper01_export_view(
            repo_root,
            dataset=dataset,
            seed=seed,
        )
    )
    return tuple(
        paper01_export_view_to_legacy_jsonl(
            export_view
        )
    )


def run_audit(
    repo_root: Path,
    *,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    rows = load_rows(
        repo_root,
        dataset=dataset,
        seed=seed,
    )

    # Direct invocation commonly uses:
    #   PYTHONPATH=src python tools/audit_paper01_filesystem_runtime.py
    # In that environment, src/ is importable but the repository-level
    # scripts/ namespace is not. Add the explicit repo root before importing
    # the two audited consumer modules. This changes import visibility only;
    # it does not alter scientific authority or filesystem selection.
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)

    simple_stats = _load_module(
        repo_root / SIMPLE_STATS_REL,
        "trustforge_m72b_simple_stats",
    )
    class_triptychs = _load_module(
        repo_root / CLASS_TRIPTYCHS_REL,
        "trustforge_m72b_class_triptychs",
    )

    checks: list[CheckResult] = []

    checks.append(
        CheckResult(
            name="exactly_seven_rows",
            passed=len(rows) == 7,
            detail=f"rows={len(rows)}",
        )
    )

    families = [
        str(row.get("model", ""))
        for row in rows
    ]
    checks.append(
        CheckResult(
            name="exactly_seven_families",
            passed=(
                len(families) == 7
                and len(set(families)) == 7
                and set(families)
                == PAPER01_FAMILIES
            ),
            detail=(
                f"families={sorted(families)}"
            ),
        )
    )

    # Reproduce the dataframe shape created by read_jsonl/json_normalize.
    dataframe = pd.json_normalize(
        list(rows)
    )

    per_model: list[dict[str, Any]] = []

    for row in rows:
        family = str(row.get("model"))
        manifest = _top_level_manifest_path(
            row
        )

        has_top_level = (
            manifest is not None
        )
        checks.append(
            CheckResult(
                name=(
                    f"top_level_manifest_path:"
                    f"{family}"
                ),
                passed=has_top_level,
                detail=(
                    str(manifest)
                    if manifest is not None
                    else (
                        "missing top-level "
                        "manifest_path"
                    )
                ),
            )
        )

        if manifest is None:
            per_model.append(
                {
                    "model": family,
                    "manifest_path": None,
                    "manifest_exists": False,
                    "simple_stats_images": 0,
                    "class_labels": [],
                    "class_triptych_counts": {},
                    "consumer_lookup_simple_stats": None,
                    "consumer_lookup_class_triptychs": None,
                }
            )
            continue

        manifest_exists = (
            manifest.is_file()
        )
        checks.append(
            CheckResult(
                name=f"manifest_exists:{family}",
                passed=manifest_exists,
                detail=str(manifest),
            )
        )

        if not manifest_exists:
            per_model.append(
                {
                    "model": family,
                    "manifest_path": str(
                        manifest
                    ),
                    "manifest_exists": False,
                    "simple_stats_images": 0,
                    "class_labels": [],
                    "class_triptych_counts": {},
                    "consumer_lookup_simple_stats": None,
                    "consumer_lookup_class_triptychs": None,
                }
            )
            continue

        try:
            records = _manifest_records(
                manifest
            )
            manifest_readable = bool(
                records
            )
        except FilesystemCompatibilityAuditError:
            records = []
            manifest_readable = False

        checks.append(
            CheckResult(
                name=f"manifest_records:{family}",
                passed=manifest_readable,
                detail=(
                    f"records={len(records)}"
                ),
            )
        )

        simple_paths = (
            simple_stats
            ._synth_paths_from_manifest(
                manifest,
                limit=64,
            )
        )

        checks.append(
            CheckResult(
                name=(
                    f"simple_stats_manifest_images:"
                    f"{family}"
                ),
                passed=bool(simple_paths),
                detail=(
                    f"existing_images="
                    f"{len(simple_paths)}"
                ),
            )
        )

        labels = sorted(
            _class_labels(manifest)
        )

        checks.append(
            CheckResult(
                name=f"class_labels:{family}",
                passed=bool(labels),
                detail=f"classes={labels}",
            )
        )

        class_counts: dict[str, int] = {}
        for class_id in labels:
            paths = (
                class_triptychs
                ._synth_by_class_from_manifest(
                    manifest,
                    class_id=class_id,
                    limit=32,
                )
            )
            class_counts[
                class_id
            ] = len(paths)

        all_classes_resolve = (
            bool(class_counts)
            and all(
                count > 0
                for count in class_counts.values()
            )
        )

        checks.append(
            CheckResult(
                name=(
                    f"class_triptych_manifest_images:"
                    f"{family}"
                ),
                passed=all_classes_resolve,
                detail=(
                    f"class_counts={class_counts}"
                ),
            )
        )

        simple_lookup = (
            simple_stats
            ._manifest_from_jsonl(
                dataframe,
                family,
            )
        )
        triptych_lookup = (
            class_triptychs
            ._manifest_from_jsonl(
                dataframe,
                family,
            )
        )

        checks.append(
            CheckResult(
                name=(
                    f"consumer_lookup_simple_stats:"
                    f"{family}"
                ),
                passed=(
                    simple_lookup == manifest
                ),
                detail=(
                    f"resolved={simple_lookup} "
                    f"expected={manifest}"
                ),
            )
        )

        checks.append(
            CheckResult(
                name=(
                    f"consumer_lookup_class_triptychs:"
                    f"{family}"
                ),
                passed=(
                    triptych_lookup == manifest
                ),
                detail=(
                    f"resolved={triptych_lookup} "
                    f"expected={manifest}"
                ),
            )
        )

        per_model.append(
            {
                "model": family,
                "manifest_path": str(
                    manifest
                ),
                "manifest_exists": True,
                "manifest_records": len(
                    records
                ),
                "simple_stats_images": len(
                    simple_paths
                ),
                "class_labels": labels,
                "class_triptych_counts": (
                    class_counts
                ),
                "consumer_lookup_simple_stats": (
                    str(simple_lookup)
                    if simple_lookup
                    is not None
                    else None
                ),
                "consumer_lookup_class_triptychs": (
                    str(triptych_lookup)
                    if triptych_lookup
                    is not None
                    else None
                ),
            }
        )

    passed = all(
        check.passed
        for check in checks
    )

    return {
        "audit": "M7.2B",
        "dataset": dataset,
        "seed": seed,
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "summary": {
            "checks_total": len(
                checks
            ),
            "checks_passed": sum(
                check.passed
                for check in checks
            ),
            "checks_failed": sum(
                not check.passed
                for check in checks
            ),
            "rows": len(rows),
            "models": len(per_model),
        },
        "checks": [
            asdict(check)
            for check in checks
        ],
        "models": per_model,
        "authority_statement": (
            "Runtime filesystem compatibility was "
            "validated only through manifest_path "
            "carried by canonical-derived rows. "
            "Fallback synthetic discovery was not "
            "invoked and is not scientific authority."
        ),
    }


def _render_markdown(
    report: Mapping[str, Any],
) -> str:
    lines = [
        "# M7.2B — Filesystem-coupled Runtime Compatibility Audit",
        "",
        f"**Status:** {report['status']}",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Seed: `{report['seed']}`",
        (
            f"- Checks: "
            f"{report['summary']['checks_passed']}/"
            f"{report['summary']['checks_total']} passed"
        ),
        "",
        "## Per-model runtime compatibility",
        "",
        (
            "| Model | Manifest exists | Manifest records | "
            "Simple-stats images | Classes | "
            "Class-triptych counts |"
        ),
        "|---|---:|---:|---:|---|---|",
    ]

    for model in report["models"]:
        counts = ", ".join(
            f"{key}:{value}"
            for key, value
            in model.get(
                "class_triptych_counts",
                {},
            ).items()
        )
        lines.append(
            f"| `{model['model']}` | "
            f"{'yes' if model.get('manifest_exists') else 'no'} | "
            f"{model.get('manifest_records', 0)} | "
            f"{model.get('simple_stats_images', 0)} | "
            f"{', '.join(model.get('class_labels', [])) or '-'} | "
            f"{counts or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Authority statement",
            "",
            report["authority_statement"],
            "",
            (
                "> A PASS means the exact canonical "
                "manifest paths resolve and both "
                "consumers can parse historical "
                "synthetic images from them directly. "
                "It does not authorize fallback globs "
                "to select scientific evidence."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_reports(
    report: Mapping[str, Any],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"paper01_filesystem_runtime_"
        f"{report['dataset']}_seed{report['seed']}"
    )
    json_path = (
        out_dir / f"{stem}.json"
    )
    md_path = (
        out_dir / f"{stem}.md"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        _render_markdown(report),
        encoding="utf-8",
    )

    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "studies/paper01_benchmark/"
            "audits/m7_2b"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        args.repo_root
        .expanduser()
        .resolve()
    )
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    report = run_audit(
        repo_root,
        dataset=args.dataset,
        seed=args.seed,
    )
    json_path, md_path = (
        write_reports(
            report,
            out_dir,
        )
    )

    print(
        f"[audit] status={report['status']}"
    )
    print(
        "[audit] dataset="
        f"{report['dataset']} "
        f"seed={report['seed']} "
        f"models={report['summary']['models']}"
    )
    print(
        "[audit] checks="
        f"{report['summary']['checks_passed']}/"
        f"{report['summary']['checks_total']}"
    )

    for model in report["models"]:
        print(
            "[audit] "
            f"{model['model']}: "
            f"manifest="
            f"{'Y' if model.get('manifest_exists') else '-'} "
            f"records={model.get('manifest_records', 0)} "
            f"simple_images="
            f"{model.get('simple_stats_images', 0)} "
            f"classes="
            f"{len(model.get('class_labels', []))}"
        )

    print(f"[audit] wrote {json_path}")
    print(f"[audit] wrote {md_path}")

    return (
        0
        if report["status"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
