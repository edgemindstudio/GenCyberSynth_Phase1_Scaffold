#!/usr/bin/env python3
"""
tools/build_paper1_jsonl.py

Transitional Paper 1 JSONL producer.

Canonical mode is selected when PHASE1_DATASET and PHASE1_SEED are both
provided. It reads canonical authority through TrustForge and writes seven
legacy-compatible JSONL rows.

When neither identity variable is present, the historical paper1.json
snapshot behavior is preserved for the still-unmigrated Paper 1 build chain.

A partial scientific identity fails closed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any


def _identity_from_env() -> tuple[str | None, int | None]:
    dataset = os.getenv("PHASE1_DATASET")
    seed_text = os.getenv("PHASE1_SEED")

    if bool(dataset) != bool(seed_text):
        raise SystemExit(
            "PHASE1_DATASET and PHASE1_SEED must be provided together"
        )

    if not dataset:
        return None, None

    try:
        seed = int(seed_text)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"PHASE1_SEED must be an integer: {seed_text!r}"
        ) from exc

    return dataset, seed


def canonical_rows(
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

    export_view = load_paper01_export_view(
        repo_root,
        dataset=dataset,
        seed=seed,
    )
    return list(
        paper01_export_view_to_legacy_jsonl(
            export_view
        )
    )


def legacy_rows(
    *,
    summary_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for path in Path("artifacts").glob(
        f"*/summaries/{summary_name}"
    ):
        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        data.setdefault(
            "source_path",
            str(path),
        )
        rows.append(data)

    rows.sort(
        key=lambda row: (
            row.get("model", ""),
            row.get("seed", 0),
        )
    )
    return rows


def write_jsonl(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main() -> int:
    dataset, seed = _identity_from_env()

    out_path = Path(
        os.getenv(
            "OUT_JSONL",
            "artifacts/summaries/phase1_summaries.jsonl",
        )
    )

    if dataset is not None and seed is not None:
        repo_root = Path(
            os.getenv(
                "TRUSTFORGE_REPO_ROOT",
                ".",
            )
        ).resolve()

        rows = canonical_rows(
            repo_root,
            dataset=dataset,
            seed=seed,
        )
        write_jsonl(
            rows,
            out_path=out_path,
        )

        print(
            f"wrote {out_path} rows={len(rows)} "
            f"authority=canonical_linkage "
            f"dataset={dataset} seed={seed}"
        )
        return 0

    summary_name = os.getenv(
        "PHASE1_SUMMARY_NAME",
        "paper1.json",
    )
    rows = legacy_rows(
        summary_name=summary_name,
    )
    write_jsonl(
        rows,
        out_path=out_path,
    )

    print(
        f"wrote {out_path} rows={len(rows)} "
        f"mode=legacy_snapshot "
        f"from={summary_name}",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
