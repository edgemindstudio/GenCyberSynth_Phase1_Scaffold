#!/usr/bin/env python3
"""
Build the Phase-1 Markdown snapshot report.

Canonical mode requires explicit dataset + seed and reads only TrustForge
canonical Paper 1 compatibility records.

Legacy mode preserves historical newest-summary behavior because the existing
Makefile still invokes this script inside the historical paper1_build chain.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MODELS = [
    "gan",
    "vae",
    "gaussianmixture",
    "diffusion",
    "autoregressive",
    "restrictedboltzmann",
    "maskedautoflow",
]


def _metric(
    row: dict[str, Any],
    key: str,
) -> Any:
    value = row.get(key)
    if value is not None:
        return value

    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get(key)
        if value is not None:
            return value

    if key == "ms_ssim":
        generative = row.get("generative")
        if isinstance(generative, dict):
            return generative.get("diversity")

    if key == "num_fake":
        counts = row.get("counts")
        if isinstance(counts, dict):
            return counts.get("num_fake")

    return None


def canonical_rows(
    repo_root: Path,
    *,
    dataset: str,
    seed: int,
) -> list[tuple[str, Any, Any]]:
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
    records = list(
        paper01_export_view_to_legacy_jsonl(view)
    )

    if len(records) != 7:
        raise RuntimeError(
            f"expected 7 canonical Paper 1 rows; found {len(records)}"
        )

    rows = []
    for record in records:
        source_name = Path(
            str(record["source_path"])
        ).name
        if source_name in {"latest.json", "paper1.json"}:
            raise RuntimeError(
                f"alias source is not canonical authority: {source_name}"
            )

        rows.append(
            (
                str(record["model"]),
                _metric(record, "ms_ssim"),
                _metric(record, "num_fake"),
            )
        )

    return sorted(
        rows,
        key=lambda row: (
            math.inf if row[1] is None else row[1],
            row[0],
        ),
    )


def latest_summary(
    artifacts_root: Path,
    model: str,
) -> Path | None:
    files = sorted(
        (artifacts_root / model / "summaries").glob(
            "summary_*.json"
        )
    )
    return files[-1] if files else None


def legacy_rows(
    artifacts_root: Path,
) -> list[tuple[str, Any, Any]]:
    rows = []

    for model in MODELS:
        path = latest_summary(
            artifacts_root,
            model,
        )
        if path is None:
            rows.append((model, None, None))
            continue

        data = json.loads(
            path.read_text(encoding="utf-8")
        )
        rows.append(
            (
                model,
                _metric(data, "ms_ssim"),
                _metric(data, "num_fake"),
            )
        )

    return sorted(
        rows,
        key=lambda row: (
            math.inf if row[1] is None else row[1],
            row[0],
        ),
    )


def render_markdown(
    rows: list[tuple[str, Any, Any]],
) -> str:
    lines = [
        "# Phase 1 – Synth Evaluation (Snapshot)",
        "",
        "| Model | MS-SSIM ↓ | #Fake |",
        "|---|---:|---:|",
    ]

    for model, ms_ssim, num_fake in rows:
        lines.append(
            f"| {model} | "
            f"{ms_ssim if ms_ssim is not None else '—'} | "
            f"{num_fake if num_fake is not None else '—'} |"
        )

    lines.extend(
        [
            "",
            "**Notes**",
            "- Lower MS-SSIM indicates higher intra-class diversity.",
            "- Extremely low values can correspond to noisy samples; grids were visually checked.",
            "- Metrics requiring `gcs_core` backends (KID/CFID) are pending environment setup.",
            "",
        ]
    )

    return "\n".join(lines)


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
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
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

    if args.canonical:
        rows = canonical_rows(
            args.repo_root.resolve(),
            dataset=args.dataset,
            seed=args.seed,
        )
        out_path = (
            args.out
            if args.out is not None
            else Path(
                f"phase1_report_{args.dataset}_seed{args.seed}.md"
            )
        )
    else:
        rows = legacy_rows(
            args.artifacts_root
        )
        out_path = (
            args.out
            if args.out is not None
            else args.artifacts_root / "phase1_report.md"
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_path.write_text(
        render_markdown(rows),
        encoding="utf-8",
    )

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
