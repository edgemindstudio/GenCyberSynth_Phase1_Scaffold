#!/usr/bin/env python3
"""Validate one canonical Paper 1 dataset/seed consumer view.

M6.5.3B migration rules:
- dataset + seed are required;
- canonical linkage selects the accepted timestamped summaries;
- local snapshot aliases do not select authority;
- integrity expectations are dataset-specific:
    USTC-TFC2016      -> 9 classes
    CICMalDroid2020   -> 5 classes

Historical summaries are opened read-only. No artifact is modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from trustforge.paper01_consumer_view import (
    Paper01ConsumerRecord,
    Paper01ConsumerView,
    load_paper01_consumer_view,
)


NUM_CLASSES_BY_DATASET = {
    "ustc_tfc2016": 9,
    "cicmaldroid2020": 5,
}


def _dig(value: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = value
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_budget(summary: Mapping[str, Any]) -> int | None:
    run_meta = summary.get("run_meta")
    if not isinstance(run_meta, Mapping):
        run_meta = {}

    return _as_int(
        _first(
            summary.get("budget_per_class"),
            run_meta.get("budget_per_class"),
        )
    )


def _summary_num_fake(summary: Mapping[str, Any]) -> int | None:
    return _as_int(
        _first(
            summary.get("counts.num_fake"),
            _dig(summary, "counts", "num_fake"),
            _dig(summary, "counts", "synthetic"),
            _dig(summary, "counts", "synthetic_count"),
            summary.get("num_fake"),
        )
    )


def _load_summary(
    record: Paper01ConsumerRecord,
) -> dict[str, Any]:
    path = record.accepted_summary_path

    if path.name == "latest.json":
        raise RuntimeError(
            f"{record.experiment_id}: latest.json cannot be authoritative"
        )

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{record.experiment_id}: accepted summary is missing: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{record.experiment_id}: accepted summary is invalid JSON: {path}"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError(
            f"{record.experiment_id}: accepted summary root must be a mapping: "
            f"{path}"
        )

    return value


def _check_record(
    record: Paper01ConsumerRecord,
    *,
    num_classes: int,
) -> list[str]:
    issues: list[str] = []

    try:
        summary = _load_summary(record)
    except RuntimeError as exc:
        return [str(exc)]

    summary_seed = _as_int(summary.get("seed"))
    if summary_seed is not None and summary_seed != record.seed:
        issues.append(
            f"{record.experiment_id}: summary seed={summary_seed} "
            f"!= canonical seed={record.seed}"
        )

    summary_model = summary.get("model")
    if (
        isinstance(summary_model, str)
        and summary_model
        and summary_model != record.family
    ):
        issues.append(
            f"{record.experiment_id}: summary model={summary_model!r} "
            f"!= canonical family={record.family!r}"
        )

    summary_budget = _summary_budget(summary)
    if summary_budget is None:
        issues.append(
            f"{record.experiment_id}: missing summary budget_per_class"
        )
    elif summary_budget != record.budget_per_class:
        issues.append(
            f"{record.experiment_id}: summary budget_per_class="
            f"{summary_budget} != canonical budget_per_class="
            f"{record.budget_per_class}"
        )

    num_fake = _summary_num_fake(summary)
    expected_num_fake = record.budget_per_class * num_classes

    if num_fake is None:
        issues.append(
            f"{record.experiment_id}: missing summary num_fake"
        )
    elif num_fake != expected_num_fake:
        issues.append(
            f"{record.experiment_id}: num_fake={num_fake} "
            f"!= expected={expected_num_fake} "
            f"({record.budget_per_class} x {num_classes} classes)"
        )

    return issues


def check_view(view: Paper01ConsumerView) -> list[str]:
    try:
        num_classes = NUM_CLASSES_BY_DATASET[view.dataset]
    except KeyError as exc:
        raise RuntimeError(
            f"no Paper 1 class-count contract for dataset={view.dataset!r}"
        ) from exc

    issues: list[str] = []

    for record in view:
        issues.extend(
            _check_record(
                record,
                num_classes=num_classes,
            )
        )

    return issues


def _parse_seed(value: str, parser: argparse.ArgumentParser) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        parser.error(f"seed must be an integer; got {value!r}")
        raise AssertionError("unreachable")


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    env = os.environ if environ is None else environ

    parser = argparse.ArgumentParser(
        description=(
            "Validate one canonical Paper 1 dataset/seed consumer view."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--dataset",
        default=env.get("PHASE1_DATASET"),
    )
    parser.add_argument(
        "--seed",
        default=env.get("PHASE1_SEED"),
    )

    args = parser.parse_args(argv)

    if not args.dataset:
        parser.error(
            "dataset is required; pass --dataset or set PHASE1_DATASET"
        )
    if args.seed is None or str(args.seed).strip() == "":
        parser.error(
            "seed is required; pass --seed or set PHASE1_SEED"
        )

    args.seed = _parse_seed(str(args.seed), parser)
    args.dataset = str(args.dataset).strip().lower()
    args.repo_root = args.repo_root.resolve()

    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    view = load_paper01_consumer_view(
        args.repo_root,
        dataset=args.dataset,
        seed=args.seed,
    )

    issues = check_view(view)

    print(f"[paper01_integrity] dataset={view.dataset}")
    print(f"[paper01_integrity] seed={view.seed}")
    print(
        f"[paper01_integrity] num_classes="
        f"{NUM_CLASSES_BY_DATASET[view.dataset]}"
    )
    print(f"[paper01_integrity] records={len(view)}")
    print(f"[paper01_integrity] issues={len(issues)}")
    print("[paper01_integrity] authority=canonical_linkage")
    print("[paper01_integrity] latest.json_authoritative=NO")

    for issue in issues:
        print(f" - {issue}")

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
