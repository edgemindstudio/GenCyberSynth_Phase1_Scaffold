#!/usr/bin/env python3
"""Generate the M6.5.2 Paper 1 consumer migration policy.

This is a read-only policy generator. It consumes the committed M6.5.1
downstream-consumer inventory and writes only repository-owned M6.5.2 audit
artifacts when not run with --dry-run.

It does not modify historical artifacts, consumer code, configs, or canonical
execution-evidence records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

AUDIT_VERSION = "M6.5.2-v1.0"
SOURCE_AUDIT_VERSION = "M6.5.1-v1.0"
STUDY_ID = "paper01_benchmark"

SOURCE_REL = Path(
    "studies/paper01_benchmark/audits/m6_5_1/"
    "paper01_downstream_consumer_inventory.json"
)
OUTPUT_DIR_REL = Path("studies/paper01_benchmark/audits/m6_5_2")
JSON_NAME = "paper01_consumer_migration_policy.json"
CSV_NAME = "paper01_consumer_migration_policy.csv"
MD_NAME = "paper01_consumer_migration_policy.md"

PRIMARY_CLASSIFICATIONS = {
    "canonical_consumer",
    "configuration_dependency",
    "governance_auditor",
    "historical_evidence_mutator_or_producer",
    "historical_pipeline",
    "legacy_or_direct_consumer",
}

TREATMENTS = {
    "ALREADY_CANONICAL",
    "CONFIGURATION_ONLY",
    "GOVERNANCE_TOOL",
    "MIGRATE_TO_CANONICAL_INTERFACE",
    "PRESERVE_HISTORICAL",
    "REVIEW_BEFORE_MIGRATION",
    "USE_CANONICAL_DERIVED_EXPORT",
    "WRAP_WITH_COMPATIBILITY_LAYER",
}

CANONICAL_LOADER = (
    "trustforge.paper01_execution_evidence.load_paper01_linkage_study"
)
PROPOSED_EXPORT = "proposed:trustforge.paper01_exports"
NO_DIRECT_TARGET = "none"

# Explicit policy is required for every mutable/legacy executable. This avoids
# silently changing treatment when the M6.5.1 inventory changes.
PATH_TREATMENT: dict[str, str] = {
    # Historical evidence mutators/producers.
    "Makefile": "REVIEW_BEFORE_MIGRATION",
    "scripts/backfill_kid_and_downstream.py": "PRESERVE_HISTORICAL",
    "scripts/backfill_ms_ssim_per_class.py": "PRESERVE_HISTORICAL",
    "scripts/eval_write_summary.py": "PRESERVE_HISTORICAL",
    "scripts/metrics/local_fid_kid.py": "PRESERVE_HISTORICAL",
    "scripts/normalize_summaries.py": "PRESERVE_HISTORICAL",
    "scripts/phase1_html.py": "MIGRATE_TO_CANONICAL_INTERFACE",
    "scripts/phase1_report.py": "MIGRATE_TO_CANONICAL_INTERFACE",
    "scripts/summaries_to_jsonl.py": "WRAP_WITH_COMPATIBILITY_LAYER",
    "scripts/utils/backfill_counts.py": "WRAP_WITH_COMPATIBILITY_LAYER",
    "tools/aggregate_phase1.py": "WRAP_WITH_COMPATIBILITY_LAYER",
    "tools/build_paper1_jsonl.py": "WRAP_WITH_COMPATIBILITY_LAYER",
    "tools/build_phase1_scores.py": "MIGRATE_TO_CANONICAL_INTERFACE",
    "tools/freeze_phase1_snapshots.py": "PRESERVE_HISTORICAL",

    # Legacy/direct consumers.
    "scripts/build_jsonl.sh": "WRAP_WITH_COMPATIBILITY_LAYER",
    "scripts/jsonl_to_csv.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/metrics/aggregate.py": "MIGRATE_TO_CANONICAL_INTERFACE",
    "scripts/metrics/print_cfid_table.py": "MIGRATE_TO_CANONICAL_INTERFACE",
    "scripts/plots/_common.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/core/calibration_curves.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/core/pareto_downstream_vs_similarity.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/core/per_class_delta_f1.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/diversity/ms_ssim_hist.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/diversity/nn_distance_distrib.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/hparams/ablation_bars.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/hparams/parallel_coords.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/imbalance/class_counts_before_after.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/imbalance/simple_stats_sanity.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/plots/qual/class_triptychs.py": "USE_CANONICAL_DERIVED_EXPORT",
    "scripts/tuning_dashboard.py": "MIGRATE_TO_CANONICAL_INTERFACE",
    "tools/check_phase1_integrity.py": "MIGRATE_TO_CANONICAL_INTERFACE",
}

EXPECTED_PATH_TREATMENT_COUNT = 31


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _load_source(repo_root: Path) -> tuple[dict[str, Any], bytes]:
    path = repo_root / SOURCE_REL
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))

    if data.get("audit_version") != SOURCE_AUDIT_VERSION:
        raise ValueError(
            "M6.5.2 requires source audit "
            f"{SOURCE_AUDIT_VERSION}; got {data.get('audit_version')!r}"
        )
    if data.get("study_id") != STUDY_ID:
        raise ValueError(
            f"expected study_id={STUDY_ID!r}; got {data.get('study_id')!r}"
        )
    if data.get("record_count") != 71:
        raise ValueError(
            f"expected M6.5.1 record_count=71; got {data.get('record_count')!r}"
        )
    return data, raw


def _priority(treatment: str, categories: set[str]) -> str:
    if treatment == "MIGRATE_TO_CANONICAL_INTERFACE":
        return "P0" if "latest_alias" in categories else "P1"
    if treatment == "WRAP_WITH_COMPATIBILITY_LAYER":
        return "P1"
    if treatment in {
        "USE_CANONICAL_DERIVED_EXPORT",
        "REVIEW_BEFORE_MIGRATION",
    }:
        return "P2"
    return "NONE"


def _target_interface(treatment: str) -> str:
    if treatment == "MIGRATE_TO_CANONICAL_INTERFACE":
        return CANONICAL_LOADER
    if treatment in {
        "WRAP_WITH_COMPATIBILITY_LAYER",
        "USE_CANONICAL_DERIVED_EXPORT",
    }:
        return PROPOSED_EXPORT
    return NO_DIRECT_TARGET


def _rationale(
    *,
    path: str,
    source_classification: str,
    treatment: str,
    categories: set[str],
) -> str:
    if treatment == "ALREADY_CANONICAL":
        return (
            "Already consumes or implements the governed Paper 1 canonical "
            "execution/evidence linkage interface; no migration is required."
        )
    if treatment == "CONFIGURATION_ONLY":
        return (
            "Configuration records a historical Paper 1 data/artifact dependency. "
            "Do not rewrite historical path identity during M6.5; handle only through "
            "explicit compatibility/portability work if a future rerun requires it."
        )
    if treatment == "GOVERNANCE_TOOL":
        return (
            "TrustForge governance/audit tooling intentionally reads historical "
            "evidence surfaces to establish provenance; preserve its audit role."
        )
    if treatment == "PRESERVE_HISTORICAL":
        return (
            "Historical producer/mutator or execution component. Rewriting it could "
            "change historical reproducibility or evidence semantics; preserve it."
        )
    if treatment == "REVIEW_BEFORE_MIGRATION":
        return (
            "Mixed orchestration surface includes destructive or evidence-writing "
            "operations. It requires a dedicated safety review before any migration."
        )
    if treatment == "MIGRATE_TO_CANONICAL_INTERFACE":
        hazard = (
            " It references latest.json, so eliminating alias-based authority is P0."
            if "latest_alias" in categories
            else ""
        )
        return (
            "Directly reads legacy Paper 1 summaries/evidence for reporting, "
            "integrity, metrics, or dashboard behavior. Replace authority selection "
            "with the canonical linkage loader." + hazard
        )
    if treatment == "WRAP_WITH_COMPATIBILITY_LAYER":
        return (
            "Builds or transforms legacy consolidated output. Preserve downstream "
            "format compatibility while sourcing authoritative rows from canonical "
            "Paper 1 linkage through a dedicated export adapter."
        )
    if treatment == "USE_CANONICAL_DERIVED_EXPORT":
        return (
            "Consumes the consolidated JSONL contract rather than choosing historical "
            "authority itself. Keep the consumer contract stable and feed it from a "
            "canonical-derived export instead of migrating this file directly."
        )
    raise AssertionError(f"unhandled treatment for {path}: {treatment}")


def _constraints(treatment: str) -> list[str]:
    base = [
        "historical_artifacts_modified=NO",
        "eval_runner_modified=NO",
        "latest_json_authoritative=NO",
    ]
    if treatment == "PRESERVE_HISTORICAL":
        base.append("historical_behavior_preserved=YES")
    if treatment == "CONFIGURATION_ONLY":
        base.append("automatic_config_rewrite=NO")
    if treatment in {
        "WRAP_WITH_COMPATIBILITY_LAYER",
        "USE_CANONICAL_DERIVED_EXPORT",
    }:
        base.append("legacy_output_contract_preserved=YES")
    return base


def _decision_for(record: dict[str, Any]) -> dict[str, Any]:
    path = record["path"]
    classification = record["classification"]
    categories = set(record["categories"])

    if classification == "canonical_consumer":
        treatment = "ALREADY_CANONICAL"
    elif classification == "configuration_dependency":
        treatment = "CONFIGURATION_ONLY"
    elif classification == "governance_auditor":
        treatment = "GOVERNANCE_TOOL"
    elif classification == "historical_pipeline":
        if path != "eval/runner.py":
            raise ValueError(
                f"unexpected historical_pipeline path requiring adjudication: {path}"
            )
        treatment = "PRESERVE_HISTORICAL"
    elif classification in {
        "historical_evidence_mutator_or_producer",
        "legacy_or_direct_consumer",
    }:
        try:
            treatment = PATH_TREATMENT[path]
        except KeyError as exc:
            raise ValueError(
                "unadjudicated executable consumer in M6.5.1 inventory: "
                f"{path} ({classification})"
            ) from exc
    else:
        raise ValueError(
            f"unexpected primary classification for {path}: {classification}"
        )

    if treatment not in TREATMENTS:
        raise AssertionError(f"invalid treatment {treatment!r}")

    return {
        "path": path,
        "source_classification": classification,
        "source_migration_policy": record["migration_policy"],
        "source_categories": sorted(categories),
        "source_accesses": sorted(record["accesses"]),
        "treatment": treatment,
        "priority": _priority(treatment, categories),
        "target_interface": _target_interface(treatment),
        "rationale": _rationale(
            path=path,
            source_classification=classification,
            treatment=treatment,
            categories=categories,
        ),
        "constraints": _constraints(treatment),
    }


def build_policy(repo_root: Path) -> dict[str, Any]:
    source, source_raw = _load_source(repo_root)
    records = [
        record
        for record in source["records"]
        if record["classification"] in PRIMARY_CLASSIFICATIONS
    ]

    if len(records) != 59:
        raise ValueError(
            f"expected 59 primary M6.5.2 records; got {len(records)}"
        )

    if len(PATH_TREATMENT) != EXPECTED_PATH_TREATMENT_COUNT:
        raise AssertionError(
            "PATH_TREATMENT count changed without updating the locked expectation"
        )

    decisions = [_decision_for(record) for record in records]
    decisions.sort(key=lambda item: item["path"])

    paths = [item["path"] for item in decisions]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate paths in M6.5.2 policy")

    treatment_counts = dict(
        sorted(Counter(item["treatment"] for item in decisions).items())
    )
    priority_counts = dict(
        sorted(Counter(item["priority"] for item in decisions).items())
    )

    return {
        "audit_version": AUDIT_VERSION,
        "study_id": STUDY_ID,
        "repository_head": _repo_head(repo_root),
        "source_audit": {
            "audit_version": source["audit_version"],
            "path": SOURCE_REL.as_posix(),
            "sha256": _sha256_bytes(source_raw),
            "record_count": source["record_count"],
            "primary_record_count": len(decisions),
        },
        "policy_principles": [
            "POLICY_DECISION != CODE_MODIFICATION",
            "SCIENTIFIC_EXPERIMENT_IDENTITY != HISTORICAL_EXECUTION_IDENTITY",
            "latest.json != AUTHORITATIVE_EVIDENCE",
            "historical_artifacts_modified=NO",
            "eval/runner.py remains protected",
            "configuration dependency != executable migration target",
            "canonical-derived compatibility export may preserve legacy consumer contracts",
        ],
        "record_count": len(decisions),
        "summary": {
            "treatments": treatment_counts,
            "priorities": priority_counts,
        },
        "records": decisions,
    }


def _write_json(path: Path, policy: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(policy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, policy: dict[str, Any]) -> None:
    fields = [
        "path",
        "source_classification",
        "source_migration_policy",
        "source_categories",
        "source_accesses",
        "treatment",
        "priority",
        "target_interface",
        "rationale",
        "constraints",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in policy["records"]:
            writer.writerow(
                {
                    "path": record["path"],
                    "source_classification": record["source_classification"],
                    "source_migration_policy": record["source_migration_policy"],
                    "source_categories": ";".join(record["source_categories"]),
                    "source_accesses": ";".join(record["source_accesses"]),
                    "treatment": record["treatment"],
                    "priority": record["priority"],
                    "target_interface": record["target_interface"],
                    "rationale": record["rationale"],
                    "constraints": ";".join(record["constraints"]),
                }
            )


def _write_md(path: Path, policy: dict[str, Any]) -> None:
    lines = [
        "# Paper 1 Consumer Migration Policy — M6.5.2",
        "",
        f"- Audit version: `{policy['audit_version']}`",
        f"- Study: `{policy['study_id']}`",
        f"- Repository HEAD: `{policy['repository_head']}`",
        f"- Source audit: `{policy['source_audit']['audit_version']}`",
        f"- Primary policy records: **{policy['record_count']}**",
        "",
        "## Policy principles",
        "",
    ]
    lines.extend(f"- `{item}`" for item in policy["policy_principles"])

    lines += [
        "",
        "## Treatment census",
        "",
        "| Treatment | Count |",
        "|---|---:|",
    ]
    for key, value in policy["summary"]["treatments"].items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Priority census",
        "",
        "| Priority | Count |",
        "|---|---:|",
    ]
    for key, value in policy["summary"]["priorities"].items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Decisions",
        "",
        "| Path | Source class | Treatment | Priority | Target |",
        "|---|---|---|---|---|",
    ]
    for record in policy["records"]:
        lines.append(
            "| `{path}` | `{source}` | `{treatment}` | `{priority}` | `{target}` |".format(
                path=record["path"],
                source=record["source_classification"],
                treatment=record["treatment"],
                priority=record["priority"],
                target=record["target_interface"],
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- `ALREADY_CANONICAL` requires no migration.",
        "- `CONFIGURATION_ONLY` records dependency identity; it is not authorization to rewrite historical configs.",
        "- `GOVERNANCE_TOOL` preserves evidence/audit responsibilities.",
        "- `PRESERVE_HISTORICAL` protects historical execution or evidence-mutating behavior.",
        "- `MIGRATE_TO_CANONICAL_INTERFACE` is a future direct-reader migration candidate.",
        "- `WRAP_WITH_COMPATIBILITY_LAYER` should produce the legacy output contract from canonical linkage.",
        "- `USE_CANONICAL_DERIVED_EXPORT` should remain decoupled from authority selection and consume a canonical-derived compatibility export.",
        "- `REVIEW_BEFORE_MIGRATION` requires a separate safety review before any code change.",
        "",
        "This audit is policy only. It does not modify any consumer or historical artifact.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_policy(repo_root: Path, policy: dict[str, Any]) -> Path:
    out_dir = (repo_root / OUTPUT_DIR_REL).resolve()
    expected = (repo_root.resolve() / OUTPUT_DIR_REL).resolve()
    if out_dir != expected:
        raise ValueError("refusing unexpected M6.5.2 output path")
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / JSON_NAME, policy)
    _write_csv(out_dir / CSV_NAME, policy)
    _write_md(out_dir / MD_NAME, policy)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    policy = build_policy(repo_root)

    print(f"[M6.5.2] Repository HEAD: {policy['repository_head']}")
    print(f"[M6.5.2] Primary policy records: {policy['record_count']}")
    for key, value in policy["summary"]["treatments"].items():
        print(f"[M6.5.2] Treatment {key}: {value}")
    for key, value in policy["summary"]["priorities"].items():
        print(f"[M6.5.2] Priority {key}: {value}")
    print("[M6.5.2] Historical artifacts modified: NO")
    print("[M6.5.2] Consumer behavior modified: NO")

    if args.dry_run:
        print("[M6.5.2] Dry run: no policy files written.")
        return 0

    out_dir = write_policy(repo_root, policy)
    print(f"[M6.5.2] Wrote policy directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
