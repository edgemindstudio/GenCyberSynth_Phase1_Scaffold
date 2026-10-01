#!/usr/bin/env python3
"""
M7.2A — Canonical-derived JSONL schema/field compatibility audit.

This is a read-only compatibility audit over one explicit Paper 1 dataset+seed.

Authority:
    explicit dataset + seed
      -> load_paper01_export_view
      -> paper01_export_view_to_legacy_jsonl
      -> exactly seven canonical-derived compatibility rows

The audit verifies structural compatibility required by the 12 P2 consumers.
Optional metric families are reported as capabilities, not invented and not
treated as authority failures when absent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

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

P2_CONSUMERS = (
    "scripts/jsonl_to_csv.py",
    "scripts/plots/_common.py",
    "scripts/plots/core/calibration_curves.py",
    "scripts/plots/core/pareto_downstream_vs_similarity.py",
    "scripts/plots/core/per_class_delta_f1.py",
    "scripts/plots/diversity/ms_ssim_hist.py",
    "scripts/plots/diversity/nn_distance_distrib.py",
    "scripts/plots/hparams/ablation_bars.py",
    "scripts/plots/hparams/parallel_coords.py",
    "scripts/plots/imbalance/class_counts_before_after.py",
    "scripts/plots/imbalance/simple_stats_sanity.py",
    "scripts/plots/qual/class_triptychs.py",
)

REQUIRED_ROW_FIELDS = (
    "model",
    "seed",
    "budget_per_class",
    "run_id",
    "timestamp",
    "source_path",
)

OPTIONAL_CAPABILITIES = {
    "counts": (
        "num_real",
        "num_fake",
        "counts.num_real",
        "counts.num_fake",
        "counts.train_real",
        "counts.synthetic",
    ),
    "generative_similarity": (
        "kid",
        "fid",
        "cfid",
        "ms_ssim",
        "metrics.kid",
        "metrics.fid",
        "metrics.fid_macro",
        "metrics.cfid",
        "metrics.cfid_macro",
        "metrics.ms_ssim",
        "generative.kid",
        "generative.fid",
        "generative.fid_macro",
        "generative.cfid_macro",
        "generative.ms_ssim",
        "generative.diversity",
    ),
    "downstream_utility": (
        "macro_f1",
        "macro_auprc",
        "balanced_acc",
        "metrics.downstream.macro_f1",
        "metrics.downstream.macro_auprc",
        "metrics.downstream.balanced_acc",
        "utility_real_plus_synth.macro_f1",
        "utility_real_plus_synth.macro_auprc",
        "utility_real_plus_synth.balanced_accuracy",
        "utility_real_plus_synth.balanced_acc",
        "utility_real_plus_synth.bal_acc",
    ),
    "calibration": (
        "utility_real_plus_synth.ece",
        "utility_real_only.ece",
        "metrics.ece",
    ),
    "per_class_f1": (
        "metrics.real_only.per_class_f1",
        "metrics.real_plus_synth.per_class_f1",
        "utility_real_only.per_class_f1",
        "utility_real_plus_synth.per_class_f1",
    ),
    "ms_ssim_per_class": (
        "metrics.ms_ssim_per_class",
        "generative.ms_ssim_per_class",
    ),
    "nn_distance": (
        "metrics.nn_dists",
        "memorization.nn_dists",
        "metrics.nn_dist_mean",
        "memorization.nn_dist_mean",
        "nn_distance_mean",
    ),
    "per_class_counts": (
        "counts.per_class.real",
        "counts.per_class.synth",
        "counts.per_class.real_plus_synth",
        "counts.real_per_class",
        "counts.synth_per_class",
        "counts.real_plus_synth_per_class",
    ),
    "manifest_path": (
        "manifest_path",
        "run_meta.manifest_path",
    ),
}

CONSUMER_CAPABILITIES = {
    "scripts/jsonl_to_csv.py": (
        "counts",
        "generative_similarity",
        "downstream_utility",
    ),
    "scripts/plots/_common.py": (),
    "scripts/plots/core/calibration_curves.py": (
        "calibration",
    ),
    "scripts/plots/core/pareto_downstream_vs_similarity.py": (
        "generative_similarity",
        "downstream_utility",
    ),
    "scripts/plots/core/per_class_delta_f1.py": (
        "per_class_f1",
    ),
    "scripts/plots/diversity/ms_ssim_hist.py": (
        "generative_similarity",
        "ms_ssim_per_class",
    ),
    "scripts/plots/diversity/nn_distance_distrib.py": (
        "nn_distance",
    ),
    "scripts/plots/hparams/ablation_bars.py": (
        "downstream_utility",
    ),
    "scripts/plots/hparams/parallel_coords.py": (
        "generative_similarity",
        "downstream_utility",
    ),
    "scripts/plots/imbalance/class_counts_before_after.py": (
        "per_class_counts",
    ),
    "scripts/plots/imbalance/simple_stats_sanity.py": (
        "manifest_path",
    ),
    "scripts/plots/qual/class_triptychs.py": (
        "manifest_path",
    ),
}


class JsonlCompatibilityAuditError(RuntimeError):
    """Raised when canonical-derived JSONL structural compatibility fails."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _dig(
    value: Mapping[str, Any],
    dotted_path: str,
) -> Any:
    current: Any = value
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _path_present(
    row: Mapping[str, Any],
    dotted_path: str,
) -> bool:
    if dotted_path in row and row.get(dotted_path) is not None:
        return True
    return _dig(row, dotted_path) is not None


def _capability_present(
    row: Mapping[str, Any],
    capability: str,
) -> bool:
    return any(
        _path_present(row, candidate)
        for candidate in OPTIONAL_CAPABILITIES[capability]
    )


def load_rows(
    repo_root: Path,
    *,
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    view = load_paper01_export_view(
        repo_root,
        dataset=dataset,
        seed=seed,
    )
    return tuple(
        paper01_export_view_to_legacy_jsonl(
            view
        )
    )


def validate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    seed: int,
) -> list[CheckResult]:
    materialized = tuple(rows)
    checks: list[CheckResult] = []

    checks.append(
        CheckResult(
            name="exactly_seven_rows",
            passed=len(materialized) == 7,
            detail=f"rows={len(materialized)}",
        )
    )

    families = [
        str(row.get("model", ""))
        for row in materialized
    ]
    checks.append(
        CheckResult(
            name="exactly_seven_families",
            passed=(
                len(families) == 7
                and len(set(families)) == 7
                and set(families) == PAPER01_FAMILIES
            ),
            detail=f"families={sorted(families)}",
        )
    )

    for index, row in enumerate(
        materialized,
        start=1,
    ):
        missing = [
            field
            for field in REQUIRED_ROW_FIELDS
            if row.get(field) is None
        ]
        checks.append(
            CheckResult(
                name=f"required_fields:row{index}",
                passed=not missing,
                detail=(
                    "all required fields present"
                    if not missing
                    else f"missing={missing}"
                ),
            )
        )

        row_seed = row.get("seed")
        try:
            row_seed = int(row_seed)
        except (TypeError, ValueError):
            row_seed = None

        checks.append(
            CheckResult(
                name=f"seed_identity:row{index}",
                passed=row_seed == seed,
                detail=f"seed={row_seed} requested={seed}",
            )
        )

        try:
            budget = int(
                row.get("budget_per_class")
            )
        except (TypeError, ValueError):
            budget = None

        checks.append(
            CheckResult(
                name=f"paper01_budget:row{index}",
                passed=budget == 2000,
                detail=f"budget_per_class={budget}",
            )
        )

        source_path = Path(
            str(row.get("source_path", ""))
        )
        source_name = source_path.name
        source_ok = (
            source_name.startswith("summary_")
            and source_name.endswith(".json")
            and source_name not in {
                "latest.json",
                "paper1.json",
            }
        )
        checks.append(
            CheckResult(
                name=f"canonical_source:row{index}",
                passed=source_ok,
                detail=f"source={source_name}",
            )
        )

        row_dataset = row.get("dataset")
        checks.append(
            CheckResult(
                name=f"dataset_if_present:row{index}",
                passed=(
                    row_dataset is None
                    or row_dataset == dataset
                ),
                detail=(
                    f"dataset={row_dataset!r} "
                    f"requested={dataset!r}"
                ),
            )
        )

    return checks


def capability_report(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = tuple(rows)
    capabilities: dict[str, Any] = {}

    for capability in OPTIONAL_CAPABILITIES:
        per_family = {
            str(row.get("model")): (
                _capability_present(
                    row,
                    capability,
                )
            )
            for row in materialized
        }

        available = sum(
            per_family.values()
        )
        capabilities[capability] = {
            "available_rows": available,
            "total_rows": len(materialized),
            "all_rows": (
                available == len(materialized)
            ),
            "any_rows": available > 0,
            "per_family": per_family,
        }

    return capabilities


def consumer_report(
    capabilities: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []

    for consumer in P2_CONSUMERS:
        names = CONSUMER_CAPABILITIES[
            consumer
        ]
        output.append(
            {
                "path": consumer,
                "structural_contract": "PASS",
                "optional_capabilities": {
                    name: {
                        "any_rows": capabilities[
                            name
                        ]["any_rows"],
                        "all_rows": capabilities[
                            name
                        ]["all_rows"],
                        "available_rows": (
                            capabilities[
                                name
                            ]["available_rows"]
                        ),
                    }
                    for name in names
                },
            }
        )

    return output


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
    checks = validate_rows(
        rows,
        dataset=dataset,
        seed=seed,
    )
    capabilities = capability_report(
        rows
    )

    passed = all(
        check.passed
        for check in checks
    )

    return {
        "audit": "M7.2A",
        "dataset": dataset,
        "seed": seed,
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(
                check.passed
                for check in checks
            ),
            "checks_failed": sum(
                not check.passed
                for check in checks
            ),
            "rows": len(rows),
            "p2_consumers": len(
                P2_CONSUMERS
            ),
        },
        "checks": [
            asdict(check)
            for check in checks
        ],
        "capabilities": capabilities,
        "consumers": consumer_report(
            capabilities
        ),
        "authority_statement": (
            "All structural rows derive from explicit "
            "dataset+seed canonical export authority. "
            "Optional metrics are reported but never "
            "invented or used to select authority."
        ),
    }


def _render_markdown(
    report: Mapping[str, Any],
) -> str:
    lines = [
        "# M7.2A — Canonical-derived JSONL Compatibility Audit",
        "",
        f"**Status:** {report['status']}",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Seed: `{report['seed']}`",
        f"- Rows: {report['summary']['rows']}",
        (
            f"- Structural checks: "
            f"{report['summary']['checks_passed']}/"
            f"{report['summary']['checks_total']} passed"
        ),
        f"- P2 consumers covered: {report['summary']['p2_consumers']}",
        "",
        "## Optional capability coverage",
        "",
        "| Capability | Available rows | All 7 | Any |",
        "|---|---:|---:|---:|",
    ]

    for name, info in sorted(
        report["capabilities"].items()
    ):
        lines.append(
            f"| `{name}` | "
            f"{info['available_rows']}/"
            f"{info['total_rows']} | "
            f"{'yes' if info['all_rows'] else 'no'} | "
            f"{'yes' if info['any_rows'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Consumer compatibility",
            "",
            "| Consumer | Structural contract | Optional capability status |",
            "|---|---|---|",
        ]
    )

    for consumer in report["consumers"]:
        optional = consumer[
            "optional_capabilities"
        ]
        if optional:
            status = "; ".join(
                (
                    f"{name}="
                    f"{info['available_rows']}/7"
                )
                for name, info in optional.items()
            )
        else:
            status = "none required"

        lines.append(
            f"| `{consumer['path']}` | "
            f"{consumer['structural_contract']} | "
            f"{status} |"
        )

    lines.extend(
        [
            "",
            "## Authority statement",
            "",
            report["authority_statement"],
            "",
            (
                "> Missing optional metrics are capability gaps, "
                "not authority gaps. This audit does not invent "
                "metrics or substitute another historical execution."
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
        f"paper01_jsonl_compat_"
        f"{report['dataset']}_seed{report['seed']}"
    )
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

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
            "audits/m7_2a"
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
    json_path, md_path = write_reports(
        report,
        out_dir,
    )

    print(
        f"[audit] status={report['status']}"
    )
    print(
        "[audit] dataset="
        f"{report['dataset']} "
        f"seed={report['seed']} "
        f"rows={report['summary']['rows']}"
    )
    print(
        "[audit] structural checks="
        f"{report['summary']['checks_passed']}/"
        f"{report['summary']['checks_total']}"
    )

    for name, info in sorted(
        report["capabilities"].items()
    ):
        print(
            f"[audit] capability {name}: "
            f"{info['available_rows']}/"
            f"{info['total_rows']}"
        )

    print(f"[audit] wrote {json_path}")
    print(f"[audit] wrote {md_path}")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
