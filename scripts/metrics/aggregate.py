#!/usr/bin/env python3
"""
Aggregate Phase-1 metrics into a compatibility CSV.

TrustForge canonical mode
=========================
Canonical mode requires an explicit dataset + seed:

    python scripts/metrics/aggregate.py \
      --canonical \
      --dataset <dataset> \
      --seed <seed> \
      --repo-root <repo> \
      --out_csv <path>

Canonical mode reads exactly the seven adjudicated Paper 1 compatibility
records selected by TrustForge. It does not glob summaries, merge multiple
executions, or choose "best" values across historical runs.

The historical CSV column names are retained for downstream compatibility:
"best_FID" and "best_NNdist" are compatibility labels only in canonical mode;
their values come from the single accepted execution for each model.

Legacy mode
===========
Without --canonical, preserve the historical behavior:
- glob per-model summary_*.json files,
- optionally ingest a Phase-1 JSONL,
- select minima for FID/NN distance across discovered records,
- retain the last numeric RS-R deltas encountered.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PRETTY = {
    "gan": "ConditionalDCGAN",
    "vae": "ConditionalVAE",
    "autoregressive": "ConditionalAutoregressive",
    "diffusion": "ConditionalDiffusion",
    "gaussianmixture": "GaussianMixture",
    "restrictedboltzmann": "RestrictedBoltzmann",
    "maskedautoflow": "MaskedAutoflow",
}
for value in list(PRETTY.values()):
    PRETTY[value] = value


PAPER01_FAMILIES = {
    "gan",
    "vae",
    "autoregressive",
    "diffusion",
    "gaussianmixture",
    "restrictedboltzmann",
    "maskedautoflow",
}


class CanonicalMetricsError(RuntimeError):
    """Raised when canonical reporting invariants are violated."""


def normalize_name(
    model: Optional[str],
    path_hint: Optional[str] = None,
) -> str:
    if not model and path_hint:
        parts = path_hint.split("/")
        for key in PRETTY:
            if key in parts:
                model = key
                break

    if model in PRETTY:
        return PRETTY[model]

    lowered = (model or "").lower()
    if lowered in PRETTY:
        return PRETTY[lowered]

    return model or "unknown"


def dict_get(
    data: Dict[str, Any],
    path: Iterable[str],
) -> Any:
    current: Any = data
    for key in path:
        if (
            not isinstance(current, dict)
            or key not in current
        ):
            return None
        current = current[key]
    return current


def first_num(*values: Any) -> float | None:
    for value in values:
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            value = float(value)
            if math.isfinite(value):
                return value
    return None


def mean_or_none(value: Any) -> float | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None

    numbers = [
        float(item)
        for item in value
        if (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
        )
    ]
    return (
        sum(numbers) / len(numbers)
        if numbers
        else None
    )


def pull_metrics(
    record: Dict[str, Any],
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
]:
    fid = first_num(
        record.get("fid"),
        dict_get(
            record,
            ("generative", "fid"),
        ),
        dict_get(
            record,
            ("metrics", "fid"),
        ),
        dict_get(
            record,
            ("generative", "fid_macro"),
        ),
        dict_get(
            record,
            ("metrics", "fid_macro"),
        ),
    )

    memorization = (
        record.get("memorization")
        if isinstance(
            record.get("memorization"),
            dict,
        )
        else {}
    )
    metrics = (
        record.get("metrics")
        if isinstance(
            record.get("metrics"),
            dict,
        )
        else {}
    )

    nn_dist = first_num(
        memorization.get("nn_dist_mean"),
        metrics.get("nn_dist_mean"),
        memorization.get("nn_mean"),
        metrics.get("nn_mean"),
        record.get("nn_dist_mean"),
    )

    if nn_dist is None:
        nn_dist = (
            mean_or_none(
                memorization.get("nn_dists")
            )
            or mean_or_none(
                metrics.get("nn_dists")
            )
        )

    deltas = (
        record.get("deltas_RS_minus_R")
        if isinstance(
            record.get("deltas_RS_minus_R"),
            dict,
        )
        else {}
    )

    delta_acc = first_num(
        deltas.get("accuracy"),
        record.get("delta_accuracy"),
        record.get("DeltaAcc_RSminusR"),
    )
    delta_f1 = first_num(
        deltas.get("macro_f1"),
        record.get("delta_macro_f1"),
        record.get("DeltaF1_RSminusR"),
    )

    return (
        fid,
        nn_dist,
        delta_acc,
        delta_f1,
    )


def collect_files(
    artifacts_root: str,
    phase1_jsonl: Optional[str],
) -> list[str]:
    files: list[str] = []

    for model in [
        "autoregressive",
        "vae",
        "gaussianmixture",
        "restrictedboltzmann",
        "maskedautoflow",
        "diffusion",
        "gan",
    ]:
        files += glob.glob(
            f"{artifacts_root}/{model}/summaries/summary_*.json"
        )

    if (
        phase1_jsonl
        and os.path.exists(phase1_jsonl)
    ):
        files.append(phase1_jsonl)

    return files


def parse_file(
    path: str,
    best: Dict[str, Dict[str, Any]],
) -> None:
    def update(
        name: str,
        fid: float | None,
        nn_dist: float | None,
        delta_acc: float | None,
        delta_f1: float | None,
    ) -> None:
        if not name:
            return

        metrics = best.setdefault(
            name,
            {
                "fid": math.inf,
                "nn": math.inf,
                "da": None,
                "df": None,
            },
        )

        if (
            isinstance(fid, (int, float))
            and fid < metrics["fid"]
        ):
            metrics["fid"] = fid

        if (
            isinstance(nn_dist, (int, float))
            and nn_dist < metrics["nn"]
        ):
            metrics["nn"] = nn_dist

        if isinstance(
            delta_acc,
            (int, float),
        ):
            metrics["da"] = delta_acc

        if isinstance(
            delta_f1,
            (int, float),
        ):
            metrics["df"] = delta_f1

    if path.endswith(".jsonl"):
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                raw_name = (
                    record.get("model")
                    or record.get("model_name")
                    or record.get("adapter")
                    or record.get("repo")
                )
                name = normalize_name(
                    raw_name,
                    path_hint=path,
                )
                update(
                    name,
                    *pull_metrics(record),
                )
        return

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as handle:
            record = json.load(handle)
    except Exception:
        return

    raw_name = (
        record.get("model")
        or record.get("adapter")
        or record.get("repo")
    )
    name = normalize_name(
        raw_name,
        path_hint=path,
    )
    update(
        name,
        *pull_metrics(record),
    )


def load_canonical_records(
    repo_root: Path,
    *,
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    from trustforge.paper01_compat import (
        paper01_export_view_to_legacy_jsonl,
    )
    from trustforge.paper01_exports import (
        load_paper01_export_view,
    )

    export_view = load_paper01_export_view(
        repo_root,
        dataset=dataset,
        seed=seed,
    )
    records = tuple(
        paper01_export_view_to_legacy_jsonl(
            export_view
        )
    )

    return validate_canonical_records(
        records,
        dataset=dataset,
        seed=seed,
    )


def validate_canonical_records(
    records: Iterable[dict[str, Any]],
    *,
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    rows = tuple(records)

    if len(rows) != 7:
        raise CanonicalMetricsError(
            "canonical Paper 1 metrics report "
            f"requires exactly 7 records; found {len(rows)}"
        )

    families: list[str] = []

    for index, record in enumerate(
        rows,
        start=1,
    ):
        family = record.get("model")
        if family not in PAPER01_FAMILIES:
            raise CanonicalMetricsError(
                f"row {index}: unexpected model {family!r}"
            )
        families.append(str(family))

        try:
            row_seed = int(record.get("seed"))
        except (TypeError, ValueError) as exc:
            raise CanonicalMetricsError(
                f"row {index}: invalid seed"
            ) from exc

        if row_seed != seed:
            raise CanonicalMetricsError(
                f"row {index}: seed={row_seed} "
                f"does not match requested seed={seed}"
            )

        try:
            budget = int(
                record.get("budget_per_class")
            )
        except (TypeError, ValueError) as exc:
            raise CanonicalMetricsError(
                f"row {index}: invalid budget_per_class"
            ) from exc

        if budget != 2000:
            raise CanonicalMetricsError(
                f"row {index}: budget_per_class={budget} "
                "does not match Paper 1 budget 2000"
            )

        source_path = record.get(
            "source_path"
        )
        if not isinstance(
            source_path,
            str,
        ):
            raise CanonicalMetricsError(
                f"row {index}: source_path is required"
            )

        source_name = Path(
            source_path
        ).name

        if source_name in {
            "latest.json",
            "paper1.json",
        }:
            raise CanonicalMetricsError(
                f"row {index}: alias source "
                f"{source_name!r} is not canonical authority"
            )

        if not source_name.startswith(
            "summary_"
        ):
            raise CanonicalMetricsError(
                f"row {index}: accepted source must "
                "be summary_*.json"
            )

        row_dataset = record.get(
            "dataset"
        )
        if (
            row_dataset is not None
            and row_dataset != dataset
        ):
            raise CanonicalMetricsError(
                f"row {index}: dataset={row_dataset!r} "
                f"does not match requested dataset={dataset!r}"
            )

    if set(families) != PAPER01_FAMILIES:
        raise CanonicalMetricsError(
            "canonical metrics report must contain "
            "all 7 Paper 1 families exactly once"
        )

    if len(families) != len(
        set(families)
    ):
        raise CanonicalMetricsError(
            "canonical metrics report contains "
            "duplicate model families"
        )

    return rows


def canonical_metrics(
    records: Iterable[dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for record in records:
        raw_name = str(record["model"])
        pretty_name = normalize_name(
            raw_name
        )

        if pretty_name in result:
            raise CanonicalMetricsError(
                f"duplicate canonical model {pretty_name}"
            )

        (
            fid,
            nn_dist,
            delta_acc,
            delta_f1,
        ) = pull_metrics(record)

        result[pretty_name] = {
            "fid": (
                math.inf
                if fid is None
                else fid
            ),
            "nn": (
                math.inf
                if nn_dist is None
                else nn_dist
            ),
            "da": delta_acc,
            "df": delta_f1,
            "source_path": record["source_path"],
        }

    return result


def csv_rows(
    metrics_by_model: Dict[
        str,
        Dict[str, Any],
    ],
) -> list[list[str]]:
    rows: list[list[str]] = []

    for model in sorted(
        metrics_by_model
    ):
        metrics = metrics_by_model[
            model
        ]

        rows.append(
            [
                model,
                (
                    ""
                    if math.isinf(
                        metrics["fid"]
                    )
                    else f"{metrics['fid']:.6f}"
                ),
                (
                    ""
                    if math.isinf(
                        metrics["nn"]
                    )
                    else f"{metrics['nn']:.6f}"
                ),
                (
                    ""
                    if metrics["da"] is None
                    else f"{metrics['da']:.6f}"
                ),
                (
                    ""
                    if metrics["df"] is None
                    else f"{metrics['df']:.6f}"
                ),
            ]
        )

    return rows


def write_csv(
    out_path: Path,
    rows: list[list[str]],
) -> None:
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Model",
                "best_FID",
                "best_NNdist",
                "DeltaAcc_RSminusR",
                "DeltaF1_RSminusR",
            ]
        )
        writer.writerows(rows)


def print_table(
    metrics_by_model: Dict[
        str,
        Dict[str, Any],
    ],
    *,
    canonical: bool,
) -> None:
    heading = (
        "Model                      "
        "FID        NNdist      "
        "ΔAcc(RS−R)   ΔF1(RS−R)"
        if canonical
        else
        "Model                      "
        "best_FID   best_NNdist   "
        "ΔAcc(RS−R)   ΔF1(RS−R)"
    )
    print(heading)

    for model in sorted(
        metrics_by_model
    ):
        metrics = metrics_by_model[
            model
        ]
        fid = (
            "-"
            if math.isinf(
                metrics["fid"]
            )
            else f"{metrics['fid']:.3f}"
        )
        nn_dist = (
            "-"
            if math.isinf(
                metrics["nn"]
            )
            else f"{metrics['nn']:.4f}"
        )
        delta_acc = (
            "-"
            if metrics["da"] is None
            else f"{metrics['da']:+.6f}"
        )
        delta_f1 = (
            "-"
            if metrics["df"] is None
            else f"{metrics['df']:+.6f}"
        )

        print(
            f"{model:26s} "
            f"{fid:>9}   "
            f"{nn_dist:>10}   "
            f"{delta_acc:>10}   "
            f"{delta_f1:>10}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Phase-1 metrics "
            "across models."
        )
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help=(
            "Use exactly seven canonical "
            "TrustForge Paper 1 records."
        ),
    )
    parser.add_argument("--dataset")
    parser.add_argument(
        "--seed",
        type=int,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--artifacts",
        default=None,
        help=(
            "Legacy mode: artifacts root "
            "with per-model subdirectories."
        ),
    )
    parser.add_argument(
        "--phase1",
        default=None,
        help=(
            "Legacy mode: optional Phase-1 "
            "summaries JSONL."
        ),
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts_summary.csv",
    )

    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace,
) -> None:
    if args.canonical:
        if not args.dataset:
            raise SystemExit(
                "--dataset is required with --canonical"
            )
        if args.seed is None:
            raise SystemExit(
                "--seed is required with --canonical"
            )
        if args.artifacts is not None:
            raise SystemExit(
                "--artifacts is legacy-only and "
                "cannot be used with --canonical"
            )
        if args.phase1 is not None:
            raise SystemExit(
                "--phase1 is legacy-only and "
                "cannot be used with --canonical"
            )
        return

    if (
        args.dataset is not None
        or args.seed is not None
    ):
        raise SystemExit(
            "--dataset/--seed require --canonical"
        )

    if args.artifacts is None:
        raise SystemExit(
            "--artifacts is required in legacy mode"
        )


def main() -> int:
    args = parse_args()
    _validate_args(args)

    if args.canonical:
        records = load_canonical_records(
            args.repo_root.resolve(),
            dataset=args.dataset,
            seed=args.seed,
        )
        metrics_by_model = (
            canonical_metrics(records)
        )
    else:
        files = collect_files(
            args.artifacts,
            args.phase1,
        )

        if not files:
            print(
                "No summary files found. "
                "Check logs for "
                "'Saved evaluation summary'."
            )
            return 1

        metrics_by_model: Dict[
            str,
            Dict[str, Any],
        ] = {}
        for path in files:
            parse_file(
                path,
                metrics_by_model,
            )

    print_table(
        metrics_by_model,
        canonical=args.canonical,
    )

    out_path = Path(
        args.out_csv
    )
    write_csv(
        out_path,
        csv_rows(metrics_by_model),
    )

    print(
        "\nWrote CSV → "
        f"{out_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
