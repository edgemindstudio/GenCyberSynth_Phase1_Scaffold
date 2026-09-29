#!/usr/bin/env python3
"""Build a canonical Paper 1 score view for one explicit dataset/seed.

M6.5.3B migration rules:
- scientific identity is explicit: dataset + seed;
- canonical linkage selects the accepted historical summary;
- latest.json and paper1.json aliases are never used for authority;
- no "newest summary" globbing is permitted;
- the legacy shared artifacts/phase1_scores.csv path is not overwritten;
- canonical outputs live under TRUSTFORGE_ARTIFACTS_ROOT unless an explicit --output is supplied.

The selected accepted timestamped summaries are opened read-only to obtain the
metric/count payload. Authority selection remains exclusively in TrustForge's
canonical Paper 1 linkage/projection layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from trustforge.paper01_consumer_view import (
    Paper01ConsumerRecord,
    Paper01ConsumerView,
    load_paper01_consumer_view,
)


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


def _load_accepted_summary(record: Paper01ConsumerRecord) -> dict[str, Any]:
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


def _validate_identity(
    record: Paper01ConsumerRecord,
    summary: Mapping[str, Any],
) -> None:
    summary_seed = _as_int(summary.get("seed"))
    if summary_seed is not None and summary_seed != record.seed:
        raise RuntimeError(
            f"{record.experiment_id}: summary seed={summary_seed} does not "
            f"match canonical seed={record.seed}"
        )

    summary_model = summary.get("model")
    if (
        isinstance(summary_model, str)
        and summary_model
        and summary_model != record.family
    ):
        raise RuntimeError(
            f"{record.experiment_id}: summary model={summary_model!r} does not "
            f"match canonical family={record.family!r}"
        )

    summary_budget = _summary_budget(summary)
    if (
        summary_budget is not None
        and summary_budget != record.budget_per_class
    ):
        raise RuntimeError(
            f"{record.experiment_id}: summary budget_per_class="
            f"{summary_budget} does not match canonical budget_per_class="
            f"{record.budget_per_class}"
        )


def _row_for_record(
    record: Paper01ConsumerRecord,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_identity(record, summary)

    run_meta = summary.get("run_meta")
    if not isinstance(run_meta, Mapping):
        run_meta = {}

    return {
        "experiment_id": record.experiment_id,
        "dataset": record.dataset,
        "model": record.family,
        "seed": record.seed,
        "budget_per_class": record.budget_per_class,
        "num_fake": _summary_num_fake(summary),
        "config_path": _first(
            summary.get("config_path"),
            run_meta.get("config_path"),
        ),
        "config_sha1": _first(
            summary.get("config_sha1"),
            run_meta.get("config_sha1"),
        ),
        "git_commit": _first(
            summary.get("git_commit"),
            run_meta.get("git_commit"),
        ),
        "kid": _first(
            summary.get("metrics.kid"),
            _dig(summary, "metrics", "kid"),
            _dig(summary, "generative", "kid"),
        ),
        "cfid": _first(
            summary.get("metrics.cfid"),
            summary.get("metrics.cfid_macro"),
            _dig(summary, "metrics", "cfid"),
            _dig(summary, "metrics", "cfid_macro"),
            _dig(summary, "generative", "cfid_macro"),
        ),
        "ms_ssim": _first(
            summary.get("metrics.ms_ssim"),
            _dig(summary, "metrics", "ms_ssim"),
            _dig(summary, "generative", "ms_ssim"),
        ),
        "fid": _first(
            summary.get("metrics.fid"),
            summary.get("metrics.fid_macro"),
            _dig(summary, "metrics", "fid"),
            _dig(summary, "metrics", "fid_macro"),
            _dig(summary, "generative", "fid"),
            _dig(summary, "generative", "fid_macro"),
        ),
        "accepted_summary_path": str(record.accepted_summary_path),
    }


def build_rows(view: Paper01ConsumerView) -> list[dict[str, Any]]:
    rows = [
        _row_for_record(record, _load_accepted_summary(record))
        for record in view
    ]
    rows.sort(key=lambda row: row["model"])
    return rows


def default_output_path(
    artifacts_root: Path,
    *,
    dataset: str,
    seed: int,
) -> Path:
    return (
        artifacts_root
        / "paper01"
        / dataset
        / f"seed{seed}"
        / "phase1_scores.csv"
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty Paper 1 score view")

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


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
            "Build a canonical Paper 1 score view for one explicit "
            "dataset/seed."
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--trustforge-artifacts-root",
        type=Path,
        default=(
            Path(env["TRUSTFORGE_ARTIFACTS_ROOT"])
            if env.get("TRUSTFORGE_ARTIFACTS_ROOT")
            else None
        ),
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

    rows = build_rows(view)

    output = args.output

    if output is None:
        artifacts_root = args.trustforge_artifacts_root
        if artifacts_root is None:
            raise SystemExit(
                "[paper01_scores] TRUSTFORGE_ARTIFACTS_ROOT is required "
                "when --output is not provided"
            )
        output = default_output_path(
            artifacts_root.expanduser().resolve(),
            dataset=view.dataset,
            seed=view.seed,
        )
    elif not output.is_absolute():
        output = args.repo_root / output

    output = output.expanduser().resolve()
    write_csv(output, rows)

    print(f"[paper01_scores] dataset={view.dataset}")
    print(f"[paper01_scores] seed={view.seed}")
    print(f"[paper01_scores] rows={len(rows)}")
    print(f"[paper01_scores] output={output}")
    print("[paper01_scores] authority=canonical_linkage")
    print("[paper01_scores] latest.json_authoritative=NO")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
