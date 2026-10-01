#!/usr/bin/env python3
"""
Status dashboard for GenCyberSynth tuning and canonical Paper 1 evidence.

Two modes are intentionally separate.

Canonical Paper 1 mode
----------------------
Use:

    python scripts/tuning_dashboard.py \
      --canonical-paper01 \
      --dataset <dataset> \
      --seeds 42 43 44 \
      --repo-root <repo>

Canonical mode reports accepted scientific evidence selected by TrustForge.
It does NOT use:
- tuning manifests as scientific authority,
- done flags as scientific authority,
- newest-summary discovery,
- file modification time,
- manifest-path matching.

Legacy tuning mode
------------------
Without --canonical-paper01, preserve the historical tuning-lite operational
dashboard. It reports whether manifests, matching summaries, and done flags
exist for requested MODEL × CFG × SEED combinations.

Important:
    OPERATIONAL COMPLETION STATUS != ACCEPTED SCIENTIFIC EVIDENCE
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PAPER01_FAMILIES = (
    "autoregressive",
    "diffusion",
    "gan",
    "gaussianmixture",
    "maskedautoflow",
    "restrictedboltzmann",
    "vae",
)


@dataclass(frozen=True)
class RunStatus:
    model: str
    cfg: str
    seed: int
    manifest_path: Path
    done_flag: Path
    has_manifest: bool
    has_done: bool
    latest_summary: Optional[Path]
    has_summary: bool
    run_id: Optional[str]
    budget_per_class: Optional[int]


@dataclass(frozen=True)
class CanonicalEvidenceStatus:
    model: str
    seed: int
    experiment_id: str
    historical_run_id: str
    accepted_summary_path: Path
    budget_per_class: int
    accepted: bool = True


class CanonicalDashboardError(RuntimeError):
    """Raised when canonical Paper 1 dashboard invariants fail."""


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _find_latest_summary_for_model(
    model_summaries_dir: Path,
    expected_manifest: Path,
) -> Optional[Path]:
    """
    Legacy operational helper only.

    Among summary_*.json files, return the newest file whose manifest reference
    matches expected_manifest. Canonical Paper 1 mode never calls this helper.
    """
    candidates = sorted(
        model_summaries_dir.glob("summary_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    expected = str(expected_manifest)

    for path in candidates[:200]:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                obj = json.load(handle)

            manifest_path = obj.get("manifest_path")
            run_manifest_path = None
            run_meta = obj.get("run_meta")
            if isinstance(run_meta, dict):
                run_manifest_path = run_meta.get(
                    "manifest_path"
                )

            if (
                manifest_path == expected
                or run_manifest_path == expected
            ):
                return path
        except Exception:
            continue

    return None


def _read_summary_fields(
    path: Path,
) -> Tuple[Optional[str], Optional[int]]:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            obj = json.load(handle)

        run_id = obj.get("run_id")
        budget = obj.get("budget_per_class")

        run_meta = obj.get("run_meta")
        if budget is None and isinstance(
            run_meta,
            dict,
        ):
            budget = run_meta.get(
                "budget_per_class"
            )

        return (
            (
                run_id
                if isinstance(run_id, str)
                else None
            ),
            _safe_int(budget),
        )
    except Exception:
        return (None, None)


def legacy_status_rows(
    *,
    artifacts: Path,
    models: List[str],
    cfgs: List[str],
    seeds: List[int],
) -> List[RunStatus]:
    rows: List[RunStatus] = []

    for model in models:
        model_root = artifacts / model
        synth_root = model_root / "synthetic"
        summaries_dir = (
            model_root / "summaries"
        )

        for cfg in cfgs:
            for seed in seeds:
                run_dir = (
                    synth_root
                    / f"{model}_{cfg}_seed{seed}"
                )
                manifest = (
                    run_dir / "manifest.json"
                )
                done_flag = (
                    summaries_dir
                    / f"done_{model}_{cfg}_seed{seed}.txt"
                )

                has_manifest = (
                    manifest.exists()
                )
                has_done = (
                    done_flag.exists()
                )

                latest_summary = None
                has_summary = False
                run_id = None
                budget = None

                if (
                    summaries_dir.exists()
                    and has_manifest
                ):
                    latest_summary = (
                        _find_latest_summary_for_model(
                            summaries_dir,
                            manifest,
                        )
                    )

                    if (
                        latest_summary is not None
                        and latest_summary.exists()
                    ):
                        has_summary = True
                        (
                            run_id,
                            budget,
                        ) = _read_summary_fields(
                            latest_summary
                        )

                rows.append(
                    RunStatus(
                        model=model,
                        cfg=cfg,
                        seed=seed,
                        manifest_path=manifest,
                        done_flag=done_flag,
                        has_manifest=has_manifest,
                        has_done=has_done,
                        latest_summary=latest_summary,
                        has_summary=has_summary,
                        run_id=run_id,
                        budget_per_class=budget,
                    )
                )

    return rows


def load_canonical_seed(
    repo_root: Path,
    *,
    dataset: str,
    seed: int,
) -> tuple[CanonicalEvidenceStatus, ...]:
    from trustforge.paper01_exports import (
        load_paper01_export_view,
    )

    view = load_paper01_export_view(
        repo_root,
        dataset=dataset,
        seed=seed,
    )

    if len(view.records) != 7:
        raise CanonicalDashboardError(
            "canonical Paper 1 dashboard "
            f"requires exactly 7 records per seed; "
            f"found {len(view.records)} for seed={seed}"
        )

    statuses: list[
        CanonicalEvidenceStatus
    ] = []
    families: list[str] = []

    for record in view.records:
        family = record.family
        families.append(family)

        if family not in PAPER01_FAMILIES:
            raise CanonicalDashboardError(
                f"unexpected Paper 1 family "
                f"{family!r}"
            )

        if record.seed != seed:
            raise CanonicalDashboardError(
                f"record seed={record.seed} "
                f"does not match requested "
                f"seed={seed}"
            )

        if record.budget_per_class != 2000:
            raise CanonicalDashboardError(
                "canonical Paper 1 budget must "
                "be 2000 per class"
            )

        source_path = Path(
            record.accepted_summary_path
        )
        source_name = source_path.name

        if source_name in {
            "latest.json",
            "paper1.json",
        }:
            raise CanonicalDashboardError(
                f"alias source {source_name!r} "
                "is not accepted authority"
            )

        if not source_name.startswith(
            "summary_"
        ):
            raise CanonicalDashboardError(
                "accepted source must be "
                "summary_*.json"
            )

        statuses.append(
            CanonicalEvidenceStatus(
                model=family,
                seed=seed,
                experiment_id=(
                    record.experiment_id
                ),
                historical_run_id=(
                    record.historical_run_id
                ),
                accepted_summary_path=(
                    source_path
                ),
                budget_per_class=(
                    record.budget_per_class
                ),
            )
        )

    if set(families) != set(
        PAPER01_FAMILIES
    ):
        raise CanonicalDashboardError(
            "canonical seed view must contain "
            "all 7 Paper 1 families"
        )

    if len(families) != len(
        set(families)
    ):
        raise CanonicalDashboardError(
            "canonical seed view contains "
            "duplicate model families"
        )

    return tuple(
        sorted(
            statuses,
            key=lambda status: status.model,
        )
    )


def canonical_status_rows(
    repo_root: Path,
    *,
    dataset: str,
    seeds: List[int],
    models: List[str] | None,
) -> List[CanonicalEvidenceStatus]:
    requested_models = (
        set(models)
        if models
        else set(PAPER01_FAMILIES)
    )

    unknown = (
        requested_models
        - set(PAPER01_FAMILIES)
    )
    if unknown:
        raise CanonicalDashboardError(
            "unknown canonical Paper 1 "
            "model(s): "
            + ", ".join(sorted(unknown))
        )

    rows: list[
        CanonicalEvidenceStatus
    ] = []

    for seed in seeds:
        for status in load_canonical_seed(
            repo_root,
            dataset=dataset,
            seed=seed,
        ):
            if (
                status.model
                in requested_models
            ):
                rows.append(status)

    return sorted(
        rows,
        key=lambda status: (
            status.seed,
            status.model,
        ),
    )


def print_canonical_dashboard(
    rows: List[CanonicalEvidenceStatus],
    *,
    dataset: str,
    seeds: List[int],
    models: List[str] | None,
    wide: bool,
) -> None:
    expected_models = (
        len(models)
        if models
        else len(PAPER01_FAMILIES)
    )
    expected = (
        expected_models
        * len(seeds)
    )

    accepted = sum(
        row.accepted
        for row in rows
    )

    print("=" * 104)
    print(
        "[dashboard] mode=canonical-paper01 "
        f"dataset={dataset}"
    )
    print(
        "[dashboard] accepted scientific "
        f"evidence: {accepted}/{expected}"
    )
    print(
        "[dashboard] operational manifests/"
        "done flags are intentionally not "
        "used as scientific authority"
    )
    print("=" * 104)

    if wide:
        print(
            f"{'MODEL':<20} "
            f"{'SEED':<5} "
            f"{'ACC':<3} "
            f"{'BPC':<5} "
            f"{'HISTORICAL_RUN_ID':<32} "
            f"{'SUMMARY_FILE':<32} "
            "EXPERIMENT_ID"
        )
    else:
        print(
            f"{'MODEL':<20} "
            f"{'SEED':<5} "
            f"{'ACC':<3} "
            f"{'BPC':<5} "
            f"{'HISTORICAL_RUN_ID'}"
        )

    for row in rows:
        mark = "Y" if row.accepted else "-"
        if wide:
            print(
                f"{row.model:<20} "
                f"{row.seed:<5} "
                f"{mark:<3} "
                f"{row.budget_per_class:<5} "
                f"{row.historical_run_id:<32} "
                f"{row.accepted_summary_path.name:<32} "
                f"{row.experiment_id}"
            )
        else:
            print(
                f"{row.model:<20} "
                f"{row.seed:<5} "
                f"{mark:<3} "
                f"{row.budget_per_class:<5} "
                f"{row.historical_run_id}"
            )

    print("=" * 104)


def print_legacy_dashboard(
    rows: List[RunStatus],
    *,
    artifacts: Path,
    models: List[str],
    cfgs: List[str],
    seeds: List[int],
    show_missing_only: bool,
    wide: bool,
) -> None:
    total = len(rows)
    n_manifest = sum(
        row.has_manifest
        for row in rows
    )
    n_summary = sum(
        row.has_summary
        for row in rows
    )
    n_done = sum(
        row.has_done
        for row in rows
    )

    print("=" * 88)
    print(
        f"[dashboard] artifacts={artifacts}"
    )
    print(
        "[dashboard] mode=legacy-operational"
    )
    print(
        "[dashboard] grid = "
        f"{len(models)} models × "
        f"{len(cfgs)} cfgs × "
        f"{len(seeds)} seeds = "
        f"{total} runs"
    )
    print(
        "[dashboard] manifest: "
        f"{n_manifest}/{total} | "
        f"summary: {n_summary}/{total} | "
        f"done: {n_done}/{total}"
    )
    print("=" * 88)

    if wide:
        print(
            f"{'MODEL':<18} "
            f"{'CFG':<3} "
            f"{'SEED':<5} "
            f"{'MAN':<3} "
            f"{'SUM':<3} "
            f"{'DONE':<4} "
            f"{'BPC':<4} "
            f"{'RUN_ID':<30} "
            f"{'SUMMARY_FILE'}"
        )
    else:
        print(
            f"{'MODEL':<18} "
            f"{'CFG':<3} "
            f"{'SEED':<5} "
            f"{'MAN':<3} "
            f"{'SUM':<3} "
            f"{'DONE':<4} "
            f"{'BPC':<4} "
            f"{'RUN_ID'}"
        )

    def mark(value: bool) -> str:
        return "Y" if value else "-"

    shown = 0
    for row in rows:
        incomplete = not (
            row.has_manifest
            and row.has_summary
            and row.has_done
        )
        if (
            show_missing_only
            and not incomplete
        ):
            continue

        budget = (
            str(row.budget_per_class)
            if row.budget_per_class
            is not None
            else "-"
        )
        run_id = (
            row.run_id
            or "-"
        )

        if wide:
            summary_file = (
                str(row.latest_summary)
                if row.latest_summary
                else "-"
            )
            print(
                f"{row.model:<18} "
                f"{row.cfg:<3} "
                f"{row.seed:<5} "
                f"{mark(row.has_manifest):<3} "
                f"{mark(row.has_summary):<3} "
                f"{mark(row.has_done):<4} "
                f"{budget:<4} "
                f"{run_id:<30} "
                f"{summary_file}"
            )
        else:
            print(
                f"{row.model:<18} "
                f"{row.cfg:<3} "
                f"{row.seed:<5} "
                f"{mark(row.has_manifest):<3} "
                f"{mark(row.has_summary):<3} "
                f"{mark(row.has_done):<4} "
                f"{budget:<4} "
                f"{run_id}"
            )

        shown += 1

    if show_missing_only:
        print("-" * 88)
        print(
            "[dashboard] showing incomplete "
            f"only: {shown}/{total} rows "
            "displayed"
        )

    print("=" * 88)
    print(
        "[dashboard] per-model completion "
        "(done flags; operational only):"
    )
    for model in models:
        model_rows = [
            row
            for row in rows
            if row.model == model
        ]
        done = sum(
            row.has_done
            for row in model_rows
        )
        print(
            f"  - {model:<18}: "
            f"{done:>2}/{len(model_rows)} done"
        )
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--canonical-paper01",
        action="store_true",
        help=(
            "Report accepted Paper 1 "
            "scientific evidence."
        ),
    )
    parser.add_argument("--dataset")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--artifacts",
        default=None,
        help=(
            "Legacy operational mode: "
            "artifact root."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--cfgs",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--show-missing-only",
        action="store_true",
        help=(
            "Legacy operational mode only: "
            "show incomplete runs."
        ),
    )
    parser.add_argument(
        "--wide",
        action="store_true",
    )

    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace,
) -> None:
    if args.canonical_paper01:
        if not args.dataset:
            raise SystemExit(
                "--dataset is required with "
                "--canonical-paper01"
            )
        if args.artifacts is not None:
            raise SystemExit(
                "--artifacts is legacy-only and "
                "cannot be used with "
                "--canonical-paper01"
            )
        if args.cfgs is not None:
            raise SystemExit(
                "--cfgs is legacy-only and "
                "cannot be used with "
                "--canonical-paper01"
            )
        if args.show_missing_only:
            raise SystemExit(
                "--show-missing-only is "
                "legacy-only and cannot be "
                "used with --canonical-paper01"
            )
        return

    if args.dataset is not None:
        raise SystemExit(
            "--dataset requires "
            "--canonical-paper01"
        )

    if args.artifacts is None:
        raise SystemExit(
            "--artifacts is required in "
            "legacy operational mode"
        )

    if not args.models:
        raise SystemExit(
            "--models is required in "
            "legacy operational mode"
        )


def main() -> int:
    args = parse_args()
    _validate_args(args)

    if args.canonical_paper01:
        rows = canonical_status_rows(
            args.repo_root.resolve(),
            dataset=args.dataset,
            seeds=args.seeds,
            models=args.models,
        )

        print_canonical_dashboard(
            rows,
            dataset=args.dataset,
            seeds=args.seeds,
            models=args.models,
            wide=args.wide,
        )
        return 0

    artifacts = Path(
        args.artifacts
    ).expanduser().resolve()
    cfgs = (
        args.cfgs
        if args.cfgs is not None
        else ["A", "B"]
    )

    rows = legacy_status_rows(
        artifacts=artifacts,
        models=args.models,
        cfgs=cfgs,
        seeds=args.seeds,
    )

    print_legacy_dashboard(
        rows,
        artifacts=artifacts,
        models=args.models,
        cfgs=cfgs,
        seeds=args.seeds,
        show_missing_only=(
            args.show_missing_only
        ),
        wide=args.wide,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
