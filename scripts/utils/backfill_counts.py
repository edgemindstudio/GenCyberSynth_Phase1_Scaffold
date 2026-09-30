#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill per-class counts into a consolidated JSONL.

TrustForge migration behavior
-----------------------------
Legacy mode preserves the historical transformer behavior.

Canonical-input mode is enabled with:

    --canonical-input --dataset <dataset> --seed <seed>

In canonical-input mode the input JSONL must already be a canonical-derived
Paper 1 compatibility export. This script does not select scientific authority;
it validates the incoming authority/provenance shape before applying count
enrichment.

Count enrichment remains auxiliary transformation data. It does not replace,
rewrite, or select the canonical summary source.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from glob import glob
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

PAPER01_FAMILIES = {
    "gan",
    "vae",
    "diffusion",
    "autoregressive",
    "maskedautoflow",
    "gaussianmixture",
    "restrictedboltzmann",
}

CANONICAL_IDENTITY_FIELDS = (
    "model",
    "seed",
    "budget_per_class",
    "source_path",
    "run_id",
    "config_path",
    "config_sha1",
    "git_commit",
)


class CanonicalInputError(RuntimeError):
    """Raised when canonical-derived JSONL fails identity validation."""


def count_real_per_class(real_root: Path) -> Dict[int, int]:
    if not real_root or not real_root.exists():
        return {}

    counts: Dict[int, int] = {}
    for sub in sorted(
        p for p in real_root.iterdir()
        if p.is_dir()
    ):
        if not re.fullmatch(r"\d+", sub.name):
            continue

        total = 0
        for ext in IMG_EXTS:
            total += len(
                glob(
                    str(sub / f"**/*{ext}"),
                    recursive=True,
                )
            )
        counts[int(sub.name)] = total

    return counts


def count_synth_from_manifest(
    manifest_path: Path,
) -> Dict[int, int]:
    if not manifest_path or not manifest_path.exists():
        return {}

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    images = data.get("images", [])
    counter = Counter()

    for item in images:
        cls = item.get(
            "class",
            item.get(
                "label",
                item.get("y"),
            ),
        )

        try:
            counter[int(cls)] += 1
            continue
        except Exception:
            pass

        path = Path(item.get("path", ""))
        try:
            counter[int(path.parent.name)] += 1
        except Exception:
            pass

    return dict(counter)


def merge_sum(
    a: Dict[int, int],
    b: Dict[int, int],
) -> Dict[int, int]:
    out = defaultdict(int)

    for key, value in a.items():
        out[int(key)] += int(value)

    for key, value in b.items():
        out[int(key)] += int(value)

    return dict(out)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CanonicalInputError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc

            if not isinstance(row, dict):
                raise CanonicalInputError(
                    f"{path}:{line_number}: JSONL row "
                    "must be an object"
                )

            rows.append(row)

    return rows


def validate_canonical_input(
    rows: Iterable[dict[str, Any]],
    *,
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    materialized = tuple(rows)

    if len(materialized) != 7:
        raise CanonicalInputError(
            "canonical Paper 1 compatibility input "
            f"must contain exactly 7 rows; found "
            f"{len(materialized)}"
        )

    models: list[str] = []

    for index, row in enumerate(
        materialized,
        start=1,
    ):
        model = row.get("model")
        if model not in PAPER01_FAMILIES:
            raise CanonicalInputError(
                f"row {index}: unexpected model {model!r}"
            )
        models.append(model)

        row_seed = row.get("seed")
        try:
            row_seed_int = int(row_seed)
        except (TypeError, ValueError) as exc:
            raise CanonicalInputError(
                f"row {index}: invalid seed {row_seed!r}"
            ) from exc

        if row_seed_int != seed:
            raise CanonicalInputError(
                f"row {index}: seed={row_seed_int} "
                f"does not match requested seed={seed}"
            )

        budget = row.get("budget_per_class")
        try:
            budget_int = int(budget)
        except (TypeError, ValueError) as exc:
            raise CanonicalInputError(
                f"row {index}: invalid "
                f"budget_per_class={budget!r}"
            ) from exc

        if budget_int != 2000:
            raise CanonicalInputError(
                f"row {index}: budget_per_class="
                f"{budget_int} does not match Paper 1 "
                "canonical budget 2000"
            )

        source_path = row.get("source_path")
        if not isinstance(source_path, str):
            raise CanonicalInputError(
                f"row {index}: source_path is required"
            )

        source_name = Path(source_path).name
        if source_name in {
            "latest.json",
            "paper1.json",
        }:
            raise CanonicalInputError(
                f"row {index}: alias source_path "
                f"is not canonical authority: "
                f"{source_name}"
            )

        if not source_name.startswith("summary_"):
            raise CanonicalInputError(
                f"row {index}: canonical source_path "
                "must be a timestamped summary_*.json"
            )

        row_dataset = row.get("dataset")
        if (
            row_dataset is not None
            and row_dataset != dataset
        ):
            raise CanonicalInputError(
                f"row {index}: dataset={row_dataset!r} "
                f"does not match requested "
                f"dataset={dataset!r}"
            )

    if set(models) != PAPER01_FAMILIES:
        raise CanonicalInputError(
            "canonical Paper 1 compatibility input "
            "must contain each of the 7 model "
            "families exactly once"
        )

    if len(models) != len(set(models)):
        raise CanonicalInputError(
            "canonical Paper 1 compatibility input "
            "contains duplicate model families"
        )

    return materialized


def _identity_snapshot(
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: deepcopy(row.get(key))
        for key in CANONICAL_IDENTITY_FIELDS
    }


def enrich_rows(
    rows: Iterable[dict[str, Any]],
    *,
    real_counts: Dict[int, int],
    synth_counts: Dict[int, int],
    num_classes: int | None,
    preserve_identity: bool,
) -> tuple[dict[str, Any], ...]:
    if num_classes is not None:
        def clamp(
            values: Dict[int, int],
        ) -> Dict[int, int]:
            return {
                int(key): int(value)
                for key, value in values.items()
                if 0 <= int(key) < num_classes
            }

        real_counts = clamp(real_counts)
        synth_counts = clamp(synth_counts)

    real_plus = merge_sum(
        real_counts,
        synth_counts,
    )

    output: list[dict[str, Any]] = []

    for original in rows:
        row = deepcopy(original)
        before = (
            _identity_snapshot(row)
            if preserve_identity
            else None
        )

        counts = row.setdefault("counts", {})
        if not isinstance(counts, dict):
            raise CanonicalInputError(
                "counts must be an object when present"
            )

        if (
            "real_per_class" not in counts
            and real_counts
        ):
            counts["real_per_class"] = real_counts

        if (
            "synth_per_class" not in counts
            and synth_counts
        ):
            counts["synth_per_class"] = synth_counts

        if (
            "real_plus_synth_per_class"
            not in counts
            and (
                real_plus
                or (
                    real_counts
                    and synth_counts
                )
            )
        ):
            counts[
                "real_plus_synth_per_class"
            ] = real_plus

        if preserve_identity:
            after = _identity_snapshot(row)
            if before != after:
                raise CanonicalInputError(
                    "count enrichment changed canonical "
                    "identity/provenance fields"
                )

        output.append(row)

    return tuple(output)


def write_jsonl(
    rows: Iterable[dict[str, Any]],
    *,
    out_path: Path,
) -> int:
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0
    with out_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill per-class counts into "
            "summaries JSONL."
        )
    )
    parser.add_argument(
        "--in",
        dest="inp",
        required=True,
        help=(
            "Input JSONL, e.g. "
            "artifacts/summaries/"
            "phase1_summaries.jsonl"
        ),
    )
    parser.add_argument(
        "--out",
        dest="out",
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--real-root",
        type=str,
        default="",
        help=(
            "Real images root "
            "(<root>/<class_id>/**)"
        ),
    )
    parser.add_argument(
        "--synth-manifest",
        type=str,
        default="",
        help=(
            "Synth manifest JSON with "
            "images[].class"
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Optional clamp to [0..C-1]",
    )
    parser.add_argument(
        "--canonical-input",
        action="store_true",
        help=(
            "Validate input as a canonical-derived "
            "Paper 1 compatibility JSONL before "
            "enrichment."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Paper 1 dataset identity; required "
            "with --canonical-input."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Paper 1 seed identity; required "
            "with --canonical-input."
        ),
    )
    return parser.parse_args()


def _validate_mode_args(
    args: argparse.Namespace,
) -> None:
    if args.canonical_input:
        if not args.dataset:
            raise SystemExit(
                "--dataset is required with "
                "--canonical-input"
            )
        if args.seed is None:
            raise SystemExit(
                "--seed is required with "
                "--canonical-input"
            )
        return

    if args.dataset is not None or args.seed is not None:
        raise SystemExit(
            "--dataset/--seed require "
            "--canonical-input"
        )


def main() -> int:
    args = parse_args()
    _validate_mode_args(args)

    input_path = Path(args.inp)
    rows = load_jsonl(input_path)

    if args.canonical_input:
        rows = list(
            validate_canonical_input(
                rows,
                dataset=args.dataset,
                seed=args.seed,
            )
        )

    real_counts = (
        count_real_per_class(
            Path(args.real_root)
        )
        if args.real_root
        else {}
    )
    synth_counts = (
        count_synth_from_manifest(
            Path(args.synth_manifest)
        )
        if args.synth_manifest
        else {}
    )

    enriched = enrich_rows(
        rows,
        real_counts=real_counts,
        synth_counts=synth_counts,
        num_classes=args.num_classes,
        preserve_identity=args.canonical_input,
    )

    wrote = write_jsonl(
        enriched,
        out_path=Path(args.out),
    )

    mode = (
        (
            "canonical-input "
            f"dataset={args.dataset} "
            f"seed={args.seed}"
        )
        if args.canonical_input
        else "legacy"
    )

    print(
        f"[ok] wrote {wrote} line(s) → "
        f"{args.out} [{mode}]"
    )

    if not real_counts and not synth_counts:
        print(
            "[warn] No counts derived. Provide "
            "--real-root and/or --synth-manifest."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
