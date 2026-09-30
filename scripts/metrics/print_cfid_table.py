#!/usr/bin/env python3
"""
Print a Phase-1 cFID/KID/MS-SSIM table.

Canonical mode requires explicit dataset + seed and reads only TrustForge
canonical Paper 1 compatibility records.

Legacy mode preserves the historical ~/gencys/artifacts newest-summary
behavior during migration.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any


MODELS = [
    "gan",
    "diffusion",
    "autoregressive",
    "vae",
    "gaussianmixture",
    "restrictedboltzmann",
    "maskedautoflow",
]


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _metric(
    row: dict[str, Any],
    top_level: str,
    nested: tuple[str, ...],
) -> float | None:
    direct = _num(row.get(top_level))
    if direct is not None:
        return direct

    current: Any = row
    for key in nested:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return _num(current)


def canonical_records(
    repo_root: Path,
    *,
    dataset: str,
    seed: int,
) -> list[dict[str, Any]]:
    from trustforge.paper01_compat import (
        paper01_export_view_to_legacy_jsonl,
    )
    from trustforge.paper01_exports import (
        load_paper01_export_view,
    )

    view = load_paper01_export_view(
        repo_root,
        dataset=dataset,
        seed=seed,
    )
    rows = list(
        paper01_export_view_to_legacy_jsonl(view)
    )

    if len(rows) != 7:
        raise RuntimeError(
            f"expected 7 canonical Paper 1 rows; found {len(rows)}"
        )

    return rows


def canonical_table(
    repo_root: Path,
    *,
    dataset: str,
    seed: int,
) -> list[tuple[str, float | None, float | None, float | None]]:
    rows = canonical_records(
        repo_root,
        dataset=dataset,
        seed=seed,
    )

    table = []
    for row in rows:
        source_name = Path(
            str(row["source_path"])
        ).name
        if source_name in {"latest.json", "paper1.json"}:
            raise RuntimeError(
                f"alias source is not canonical authority: {source_name}"
            )

        model = str(row["model"])
        cfid = _metric(
            row,
            "cfid",
            ("generative", "cfid_macro"),
        )
        kid = _metric(
            row,
            "kid",
            ("generative", "kid"),
        )
        ms_ssim = _metric(
            row,
            "ms_ssim",
            ("metrics", "ms_ssim"),
        )
        if ms_ssim is None:
            ms_ssim = _metric(
                row,
                "ms_ssim",
                ("generative", "diversity"),
            )

        table.append(
            (model, cfid, kid, ms_ssim)
        )

    return sorted(
        table,
        key=lambda row: (
            math.inf if row[1] is None else row[1],
            row[0],
        ),
    )


def legacy_table() -> list[
    tuple[str, float | None, float | None, float | None]
]:
    rows = []

    for model in MODELS:
        pattern = os.path.expanduser(
            f"~/gencys/artifacts/{model}/summaries/summary_*.json"
        )
        files = sorted(glob.glob(pattern))
        data = (
            json.load(open(files[-1], encoding="utf-8"))
            if files
            else {}
        )
        generative = data.get("generative") or {}

        rows.append(
            (
                model,
                _num(generative.get("cfid_macro")),
                _num(generative.get("kid")),
                _num(generative.get("ms_ssim")),
            )
        )

    return sorted(
        rows,
        key=lambda row: (
            math.inf if row[1] is None else row[1],
            row[0],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical",
        action="store_true",
    )
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.canonical:
        if not args.dataset:
            raise SystemExit(
                "--dataset is required with --canonical"
            )
        if args.seed is None:
            raise SystemExit(
                "--seed is required with --canonical"
            )
        return

    if args.dataset is not None or args.seed is not None:
        raise SystemExit(
            "--dataset/--seed require --canonical"
        )


def main() -> int:
    args = parse_args()
    _validate_args(args)

    rows = (
        canonical_table(
            args.repo_root.resolve(),
            dataset=args.dataset,
            seed=args.seed,
        )
        if args.canonical
        else legacy_table()
    )

    print(
        f"{'model':20s} {'cFID':>12} "
        f"{'KID':>12} {'MS-SSIM':>12}"
    )

    def fmt(value: float | None) -> str:
        return (
            "-"
            if value is None
            else f"{value:.6f}"
        )

    for model, cfid, kid, ms_ssim in rows:
        print(
            f"{model:20s} "
            f"{fmt(cfid):>12} "
            f"{fmt(kid):>12} "
            f"{fmt(ms_ssim):>12}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
