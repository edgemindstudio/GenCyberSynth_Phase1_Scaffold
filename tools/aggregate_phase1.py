#!/usr/bin/env python3
"""
Aggregate Phase-1 evaluation summaries into CSV + JSONL.

TrustForge migration behavior
=============================

Canonical mode
--------------
Use:

    python tools/aggregate_phase1.py \
      --canonical \
      --dataset <dataset> \
      --seed <seed> \
      --repo-root <repo> \
      --outdir <dir>

Canonical mode obtains authority only through:

    trustforge.paper01_exports
        -> trustforge.paper01_compat

It does NOT:
- discover summary files by glob,
- select a newest summary,
- use file modification time as authority,
- infer synthetic counts from sibling filesystem artifacts.

It only transforms the already-adjudicated seven canonical Paper 1 records into
the historical aggregate CSV schema. The JSONL output remains the canonical
legacy-compatible records so downstream compatibility consumers retain the
accepted source_path and provenance fields.

Legacy mode
-----------
When --canonical is absent, the historical cross-repository discovery behavior
is preserved for compatibility during migration.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_BASE = Path.home() / "PycharmProjects"
DEFAULT_REPOS = [
    "GAN",
    "VAEs",
    "AUTOREGRESSIVE",
    "MASKEDAUTOFLOW",
    "RESTRICTEDBOLTZMANN",
    "GAUSSIANMIXTURE",
    "DIFFUSION",
]

SUMMARY_PATTERNS: Tuple[str, ...] = (
    "artifacts/**/summaries/*_eval_summary_seed*.json",
    "artifacts/*/summaries/*_eval_summary_seed*.json",
)

CSV_NAME = "phase1_table.csv"
JSONL_NAME = "phase1_summaries.jsonl"

PAPER01_FAMILIES = {
    "gan",
    "vae",
    "diffusion",
    "autoregressive",
    "maskedautoflow",
    "gaussianmixture",
    "restrictedboltzmann",
}


class CanonicalAggregateError(RuntimeError):
    """Raised when canonical aggregation invariants are violated."""


def pick(
    d: Optional[Dict[str, Any]],
    *keys: str,
    default: Any = None,
) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def is_num(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def as_float_or_none(
    value: Any,
) -> Optional[float]:
    if is_num(value):
        return float(value)

    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except Exception:
            return None
        return parsed if math.isfinite(parsed) else None

    return None


def coerce_numeric_inplace(obj: Any) -> Any:
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = coerce_numeric_inplace(value)
        return obj

    if isinstance(obj, list):
        for index, value in enumerate(obj):
            obj[index] = coerce_numeric_inplace(value)
        return obj

    parsed = as_float_or_none(obj)
    return parsed if parsed is not None else obj


def find_summaries(
    repo_dir: Path,
    patterns: Iterable[str],
) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()

    for pattern in patterns:
        for path in repo_dir.glob(pattern):
            absolute = (
                path
                if path.is_absolute()
                else repo_dir / path
            )
            if absolute not in seen:
                seen.add(absolute)
                out.append(absolute)

    return out


def newest(
    paths: List[Path],
) -> Optional[Path]:
    return (
        max(
            paths,
            key=lambda path: path.stat().st_mtime,
        )
        if paths
        else None
    )


def recompute_deltas(
    row: Dict[str, Any],
) -> None:
    pairs = [
        ("ΔAcc", "Acc_RS", "Acc_R"),
        ("ΔMacroF1", "MacroF1_RS", "MacroF1_R"),
        ("ΔBalAcc", "BalAcc_RS", "BalAcc_R"),
        ("ΔmAUPRC", "mAUPRC_RS", "mAUPRC_R"),
        ("ΔR@1%FPR", "R@1%FPR_RS", "R@1%FPR_R"),
        ("ΔECE", "ECE_RS", "ECE_R"),
        ("ΔBrier", "Brier_RS", "Brier_R"),
    ]

    for delta_key, rs_key, real_key in pairs:
        if is_num(row.get(delta_key)):
            continue

        rs_value = as_float_or_none(
            row.get(rs_key)
        )
        real_value = as_float_or_none(
            row.get(real_key)
        )

        row[delta_key] = (
            rs_value - real_value
            if (
                rs_value is not None
                and real_value is not None
            )
            else None
        )


def rs_block(
    data: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], str]:
    candidates = [
        "utility_real_plus_synth",
        "utility_real_plus_synthetic",
        "utility_RS",
        "utility_real_and_synth",
        "util_real_plus_synth",
    ]

    for name in candidates:
        value = data.get(name)
        if isinstance(value, dict):
            return value, name

    utility = data.get("utility")
    if isinstance(utility, dict):
        for name in ("real_plus_synth", "RS"):
            value = utility.get(name)
            if isinstance(value, dict):
                return value, f"utility.{name}"

    return None, "missing"


def rs_status(
    utility_rs: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not isinstance(utility_rs, dict):
        return False, "util_RS missing"

    keys = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "macro_auprc",
        "recall_at_1pct_fpr",
        "ece",
        "brier",
    ]

    missing = [
        key
        for key in keys
        if key not in utility_rs
    ]
    if missing:
        return (
            False,
            "util_RS missing keys: "
            + ", ".join(missing),
        )

    non_numeric = [
        key
        for key in keys
        if as_float_or_none(
            utility_rs.get(key)
        )
        is None
    ]
    if len(non_numeric) == len(keys):
        return (
            False,
            "util_RS present but empty/non-numeric "
            "(REAL-only?)",
        )

    return True, ""


def infer_synth_count_from_fs(
    summary_path: Path,
) -> Optional[int]:
    """
    Historical legacy fallback only.

    Canonical mode never calls this function.
    """
    if "summaries" not in summary_path.parts:
        return None

    parts = list(summary_path.parts)
    try:
        index = parts.index("summaries")
    except ValueError:
        return None

    synth_dir = Path(
        *parts[:index],
        "synthetic",
    )
    if not synth_dir.exists():
        return None

    x_all = synth_dir / "x_synth.npy"
    if x_all.exists():
        try:
            import numpy as np

            values = np.load(
                x_all,
                mmap_mode="r",
            )
            return int(values.shape[0])
        except Exception:
            pass

    per_class = list(
        synth_dir.glob("gen_class_*.npy")
    )
    if per_class:
        total = 0
        try:
            import numpy as np

            for path in per_class:
                values = np.load(
                    path,
                    mmap_mode="r",
                )
                total += int(
                    values.shape[0]
                )
            return (
                total
                if total > 0
                else None
            )
        except Exception:
            return None

    return None


def columns() -> List[str]:
    return [
        "repo",
        "model",
        "seed",
        "train_real",
        "val_real",
        "test_real",
        "synthetic",
        "FID",
        "cFID_macro",
        "JS",
        "KL",
        "Diversity",
        "Acc_R",
        "MacroF1_R",
        "BalAcc_R",
        "mAUPRC_R",
        "R@1%FPR_R",
        "ECE_R",
        "Brier_R",
        "Acc_RS",
        "MacroF1_RS",
        "BalAcc_RS",
        "mAUPRC_RS",
        "R@1%FPR_RS",
        "ECE_RS",
        "Brier_RS",
        "ΔAcc",
        "ΔMacroF1",
        "ΔBalAcc",
        "ΔmAUPRC",
        "ΔR@1%FPR",
        "ΔECE",
        "ΔBrier",
        "_has_RS_metrics",
        "_rs_reason",
        "_rs_schema",
        "_has_FID",
        "_synth_fs_count",
        "_summary_relpath",
        "_summary_path",
        "_summary_mtime",
    ]


def _synthetic_count_from_payload(
    data: Dict[str, Any],
) -> Any:
    images = (
        data.get("images")
        if isinstance(
            data.get("images"),
            dict,
        )
        else {}
    )

    synthetic = images.get("synthetic")
    if synthetic not in (None, 0):
        return synthetic

    counts = (
        data.get("counts")
        if isinstance(
            data.get("counts"),
            dict,
        )
        else {}
    )

    for candidate in (
        counts.get("num_fake"),
        counts.get("synthetic"),
        data.get("num_fake"),
    ):
        if candidate not in (None, 0):
            return candidate

    return synthetic


def row_from_payload(
    repo: str,
    source_path: Path,
    raw: Dict[str, Any],
    *,
    repo_dir: Path | None,
    allow_filesystem_fallback: bool,
    summary_mtime: float | None,
) -> Dict[str, Any]:
    data = coerce_numeric_inplace(
        deepcopy(raw)
    )

    model = (
        pick(data, "model")
        or repo
    )
    seed = pick(data, "seed")

    images = (
        pick(
            data,
            "images",
            default={},
        )
        or {}
    )
    generative = (
        pick(
            data,
            "generative",
            default={},
        )
        or {}
    )
    utility_real = (
        pick(
            data,
            "utility_real_only",
            default={},
        )
        or {}
    )
    utility_rs, rs_schema = rs_block(data)
    deltas = (
        pick(
            data,
            "deltas_RS_minus_R",
            default={},
        )
        or {}
    )

    if repo_dir is None:
        rel = source_path
    else:
        try:
            rel = source_path.relative_to(
                repo_dir
            )
        except Exception:
            rel = source_path.name

    has_rs, rs_reason = rs_status(
        utility_rs
    )

    synthetic_json = (
        images.get("synthetic")
    )

    synth_fs = None
    if (
        allow_filesystem_fallback
        and synthetic_json in (None, 0)
    ):
        synth_fs = infer_synth_count_from_fs(
            source_path
        )

    if allow_filesystem_fallback:
        synthetic = (
            synthetic_json
            if synthetic_json not in (None, 0)
            else synth_fs
        )
    else:
        synthetic = (
            _synthetic_count_from_payload(
                data
            )
        )

    row = {
        "repo": repo,
        "model": model,
        "seed": seed,
        "train_real": images.get("train_real"),
        "val_real": images.get("val_real"),
        "test_real": images.get("test_real"),
        "synthetic": synthetic,
        "FID": generative.get("fid"),
        "cFID_macro": generative.get(
            "cfid_macro"
        ),
        "JS": generative.get("js"),
        "KL": generative.get("kl"),
        "Diversity": generative.get(
            "diversity"
        ),
        "Acc_R": utility_real.get(
            "accuracy"
        ),
        "MacroF1_R": utility_real.get(
            "macro_f1"
        ),
        "BalAcc_R": utility_real.get(
            "balanced_accuracy"
        ),
        "mAUPRC_R": utility_real.get(
            "macro_auprc"
        ),
        "R@1%FPR_R": utility_real.get(
            "recall_at_1pct_fpr"
        ),
        "ECE_R": utility_real.get("ece"),
        "Brier_R": utility_real.get(
            "brier"
        ),
        "Acc_RS": (
            None
            if utility_rs is None
            else utility_rs.get(
                "accuracy"
            )
        ),
        "MacroF1_RS": (
            None
            if utility_rs is None
            else utility_rs.get(
                "macro_f1"
            )
        ),
        "BalAcc_RS": (
            None
            if utility_rs is None
            else utility_rs.get(
                "balanced_accuracy"
            )
        ),
        "mAUPRC_RS": (
            None
            if utility_rs is None
            else utility_rs.get(
                "macro_auprc"
            )
        ),
        "R@1%FPR_RS": (
            None
            if utility_rs is None
            else utility_rs.get(
                "recall_at_1pct_fpr"
            )
        ),
        "ECE_RS": (
            None
            if utility_rs is None
            else utility_rs.get("ece")
        ),
        "Brier_RS": (
            None
            if utility_rs is None
            else utility_rs.get("brier")
        ),
        "ΔAcc": pick(
            deltas,
            "accuracy",
        ),
        "ΔMacroF1": pick(
            deltas,
            "macro_f1",
        ),
        "ΔBalAcc": pick(
            deltas,
            "balanced_accuracy",
        ),
        "ΔmAUPRC": pick(
            deltas,
            "macro_auprc",
        ),
        "ΔR@1%FPR": pick(
            deltas,
            "recall_at_1pct_fpr",
        ),
        "ΔECE": pick(
            deltas,
            "ece",
        ),
        "ΔBrier": pick(
            deltas,
            "brier",
        ),
        "_has_RS_metrics": has_rs,
        "_rs_reason": rs_reason,
        "_rs_schema": rs_schema,
        "_has_FID": is_num(
            generative.get("fid")
        ),
        "_synth_fs_count": synth_fs,
        "_summary_relpath": str(rel),
        "_summary_path": str(source_path),
        "_summary_mtime": summary_mtime,
    }

    recompute_deltas(row)
    return row


def row_from_summary(
    repo: str,
    summary_path: Path,
    repo_dir: Path,
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    return row_from_payload(
        repo,
        summary_path,
        raw,
        repo_dir=repo_dir,
        allow_filesystem_fallback=True,
        summary_mtime=summary_path.stat().st_mtime,
    )


def _validate_canonical_records(
    records: Iterable[dict[str, Any]],
    *,
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    rows = tuple(records)

    if len(rows) != 7:
        raise CanonicalAggregateError(
            "canonical Paper 1 aggregate requires "
            f"exactly 7 records; found {len(rows)}"
        )

    families: list[str] = []

    for index, record in enumerate(
        rows,
        start=1,
    ):
        family = record.get("model")
        if family not in PAPER01_FAMILIES:
            raise CanonicalAggregateError(
                f"row {index}: unexpected model "
                f"{family!r}"
            )
        families.append(family)

        row_seed = record.get("seed")
        try:
            row_seed = int(row_seed)
        except (TypeError, ValueError) as exc:
            raise CanonicalAggregateError(
                f"row {index}: invalid seed"
            ) from exc

        if row_seed != seed:
            raise CanonicalAggregateError(
                f"row {index}: seed={row_seed} "
                f"does not match requested seed={seed}"
            )

        budget = record.get(
            "budget_per_class"
        )
        try:
            budget = int(budget)
        except (TypeError, ValueError) as exc:
            raise CanonicalAggregateError(
                f"row {index}: invalid "
                "budget_per_class"
            ) from exc

        if budget != 2000:
            raise CanonicalAggregateError(
                f"row {index}: "
                f"budget_per_class={budget} "
                "does not match Paper 1 budget 2000"
            )

        source_path = record.get(
            "source_path"
        )
        if not isinstance(
            source_path,
            str,
        ):
            raise CanonicalAggregateError(
                f"row {index}: source_path "
                "is required"
            )

        source_name = Path(
            source_path
        ).name

        if source_name in {
            "latest.json",
            "paper1.json",
        }:
            raise CanonicalAggregateError(
                f"row {index}: alias source "
                f"{source_name!r} is not "
                "canonical authority"
            )

        if not source_name.startswith(
            "summary_"
        ):
            raise CanonicalAggregateError(
                f"row {index}: accepted "
                "source must be summary_*.json"
            )

        row_dataset = record.get(
            "dataset"
        )
        if (
            row_dataset is not None
            and row_dataset != dataset
        ):
            raise CanonicalAggregateError(
                f"row {index}: dataset="
                f"{row_dataset!r} does not match "
                f"requested dataset={dataset!r}"
            )

    if set(families) != PAPER01_FAMILIES:
        raise CanonicalAggregateError(
            "canonical aggregate must contain "
            "all 7 Paper 1 families exactly once"
        )

    if len(families) != len(
        set(families)
    ):
        raise CanonicalAggregateError(
            "canonical aggregate contains "
            "duplicate model families"
        )

    return rows


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
    records = (
        paper01_export_view_to_legacy_jsonl(
            export_view
        )
    )

    return _validate_canonical_records(
        records,
        dataset=dataset,
        seed=seed,
    )


def canonical_rows(
    records: Iterable[dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for record in records:
        source_path = Path(
            record["source_path"]
        )
        model = str(record["model"])

        row = row_from_payload(
            model,
            source_path,
            record,
            repo_dir=None,
            allow_filesystem_fallback=False,
            summary_mtime=None,
        )
        rows.append(row)

    rows.sort(
        key=lambda row: str(
            row.get("model", "")
        )
    )
    return rows


def legacy_records_and_rows(
    *,
    base: Path,
    include_all: bool,
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    for repo in DEFAULT_REPOS:
        repo_dir = base / repo
        paths = find_summaries(
            repo_dir,
            SUMMARY_PATTERNS,
        )

        if not paths:
            print(
                "[skip] No summaries found "
                f"in {repo_dir}"
            )
            continue

        use_paths = (
            paths
            if include_all
            else [newest(paths)]
        )

        for summary_path in use_paths:
            if summary_path is None:
                continue

            try:
                raw = json.loads(
                    Path(
                        summary_path
                    ).read_text(
                        encoding="utf-8"
                    )
                )
            except Exception as exc:
                print(
                    "[warn] Failed to read "
                    f"{summary_path}: {exc}"
                )
                continue

            rows.append(
                row_from_summary(
                    repo,
                    summary_path,
                    repo_dir,
                    raw,
                )
            )
            records.append(
                {
                    "repo": repo,
                    **raw,
                }
            )

    return records, rows


def write_outputs(
    *,
    outdir: Path,
    rows: List[Dict[str, Any]],
    jsonl_records: Iterable[
        Dict[str, Any]
    ],
) -> tuple[Path, Path]:
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = outdir / CSV_NAME
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns(),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = outdir / JSONL_NAME
    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        first = True
        for record in jsonl_records:
            if not first:
                handle.write("\n")
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            first = False

    return csv_path, jsonl_path


def fmt_delta(value: Any) -> str:
    parsed = as_float_or_none(value)
    return (
        "n/a"
        if parsed is None
        else f"{parsed:+.4f}"
    )


def print_topline(
    rows: List[Dict[str, Any]],
) -> None:
    winners = [
        row
        for row in rows
        if as_float_or_none(
            row.get("ΔMacroF1")
        )
        is not None
    ]

    winners.sort(
        key=lambda row: (
            float(row["ΔMacroF1"]),
            float(
                as_float_or_none(
                    row.get("ΔBalAcc")
                )
                or -1e12
            ),
        ),
        reverse=True,
    )

    print("\nTop-line (by ΔMacroF1):")
    if not winners:
        print(
            "  (no rows had a numeric "
            "ΔMacroF1)"
        )
    else:
        for row in winners[:5]:
            print(
                f"  {row['model']:>28s}  "
                f"ΔF1={fmt_delta(row.get('ΔMacroF1'))}  "
                f"ΔBalAcc={fmt_delta(row.get('ΔBalAcc'))}  "
                f"ΔAUPRC={fmt_delta(row.get('ΔmAUPRC'))}  "
                f"ΔECE={fmt_delta(row.get('ΔECE'))}  "
                "FID="
                f"{row.get('FID') if is_num(row.get('FID')) else 'n/a'}"
            )

    missing = [
        row["model"]
        for row in rows
        if as_float_or_none(
            row.get("ΔMacroF1")
        )
        is None
    ]
    if missing:
        print(
            "Note: no synthetic delta for -> "
            + ", ".join(
                sorted(
                    set(missing)
                )
            )
        )


def print_diagnostics(
    rows: List[Dict[str, Any]],
) -> None:
    print("\nDiagnostics:")
    header = (
        f"{'repo/model':34s} "
        f"{'synthetic':>9s} "
        f"{'has_RS':>7s} "
        f"{'has_FID':>8s} "
        f"{'rs_key':>18s}  "
        "reason / summary"
    )
    print(header)

    for row in rows:
        synth = row.get("synthetic")
        has_rs = bool(
            row.get("_has_RS_metrics")
        )
        has_fid = bool(
            row.get("_has_FID")
        )
        rs_key = (
            row.get("_rs_schema")
            or "missing"
        )[:18]
        reason = (
            row.get("_rs_reason")
            or ""
        )

        print(
            f"{(row['repo'] + ' / ' + row['model'])[:34].ljust(34)}"
            f"{str(synth).rjust(9)}"
            f"{str(has_rs).rjust(7)}"
            f"{str(has_fid).rjust(8)} "
            f"{rs_key.rjust(18)}  "
            f"{reason}  "
            f"[{row['_summary_relpath']}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Phase-1 evaluation "
            "results."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help=(
            "Legacy mode: base directory "
            "containing historical repos"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd(),
        help="Directory to write CSV/JSONL",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Legacy mode: include all "
            "summaries instead of newest"
        ),
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print per-row diagnostics",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help=(
            "Use canonical TrustForge "
            "Paper 1 authority."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Paper 1 dataset identity; "
            "required with --canonical."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Paper 1 seed identity; "
            "required with --canonical."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "TrustForge repository root "
            "for canonical mode."
        ),
    )
    return parser.parse_args()


def _validate_mode_args(
    args: argparse.Namespace,
) -> None:
    if args.canonical:
        if not args.dataset:
            raise SystemExit(
                "--dataset is required with "
                "--canonical"
            )
        if args.seed is None:
            raise SystemExit(
                "--seed is required with "
                "--canonical"
            )
        if args.all:
            raise SystemExit(
                "--all is legacy-only and "
                "cannot be used with "
                "--canonical"
            )
        return

    if (
        args.dataset is not None
        or args.seed is not None
    ):
        raise SystemExit(
            "--dataset/--seed require "
            "--canonical"
        )


def main() -> int:
    args = parse_args()
    _validate_mode_args(args)

    if args.canonical:
        records = load_canonical_records(
            args.repo_root.resolve(),
            dataset=args.dataset,
            seed=args.seed,
        )
        rows = canonical_rows(
            records
        )
        jsonl_records = list(records)
        mode = (
            "canonical "
            f"dataset={args.dataset} "
            f"seed={args.seed}"
        )
    else:
        (
            jsonl_records,
            rows,
        ) = legacy_records_and_rows(
            base=args.base,
            include_all=args.all,
        )
        mode = "legacy"

        def sort_key(
            row: Dict[str, Any],
        ) -> Tuple[float, float]:
            delta_f1 = (
                as_float_or_none(
                    row.get("ΔMacroF1")
                )
                or -1e12
            )
            delta_bal = (
                as_float_or_none(
                    row.get("ΔBalAcc")
                )
                or -1e12
            )
            return (
                delta_f1,
                delta_bal,
            )

        rows.sort(
            key=sort_key,
            reverse=True,
        )

    if not rows:
        print(
            "[warn] No summaries available "
            f"for mode={mode}."
        )
        return 1

    csv_path, jsonl_path = write_outputs(
        outdir=args.outdir,
        rows=rows,
        jsonl_records=jsonl_records,
    )

    print(
        f"[ok] wrote {csv_path} and "
        f"{jsonl_path} [{mode}]"
    )

    print_topline(rows)
    if args.diagnose:
        print_diagnostics(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
