#!/usr/bin/env python3
"""M6.5.1 — Paper 1 downstream consumer inventory.

READ-ONLY REPOSITORY AUDITOR.

This tool inventories tracked repository files that consume, produce, mutate,
reference, or already use canonical interfaces related to Paper 1 evidence.
It scans repository source text only and never opens historical evidence roots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

AUDIT_VERSION = "M6.5.1-v1.0"
STUDY_ID = "paper01_benchmark"
DEFAULT_OUTPUT_DIR = Path("studies/paper01_benchmark/audits/m6_5_1")

TEXT_SUFFIXES = {
    ".py", ".sh", ".yaml", ".yml", ".json", ".jsonl", ".md",
    ".txt", ".toml", ".ini", ".cfg", ".conf", ".csv", ".mk",
}
TEXT_FILENAMES = {"Makefile", "Makefile.bak", "Runbook.md", "README.md"}
EXCLUDED_PREFIXES = (
    ".git/",
    # Evidence/data products are governance inputs, not downstream consumers.
    "studies/paper01_benchmark/execution_evidence/",
    "studies/paper01_benchmark/experiments/",
    "studies/paper01_benchmark/audits/",
)

EXCLUDED_PATHS = {
    # Prevent M6.5.1 from inventorying itself after it becomes tracked.
    "scripts/inventory_paper01_downstream_consumers.py",
    "tests/trustforge/test_paper01_downstream_consumers.py",
    # Paper 1 governance data is evidence, not a downstream consumer.
    "studies/paper01_benchmark/evidence.yaml",
    "studies/paper01_benchmark/lineage.yaml",
}

PROTECTED_HISTORICAL_PATHS = {"eval/runner.py"}

GOVERNANCE_AUDIT_PATHS = {
    "scripts/inventory_paper01_execution_evidence.py",
}

CONFIG_SUFFIXES = {".yaml", ".yml"}

EXPLICIT_PAPER1_CATEGORIES = {
    "canonical_linkage_interface",
    "authoritative_score_table",
    "consolidated_jsonl",
    "historical_paper1_root",
}

CATEGORY_PATTERNS = {
    "canonical_linkage_interface": (
        re.compile(r"\btrustforge\.paper01_execution_evidence\b"),
        re.compile(r"\bload_paper01_linkage_study\b"),
        re.compile(r"studies/paper01_benchmark/execution_evidence(?:/|\b)"),
    ),
    "authoritative_score_table": (
        re.compile(r"\bphase1_scores_dedup\.csv\b"),
    ),
    "timestamped_summary": (
        re.compile(r"\bsummary_\*\.json\b"),
        re.compile(r"\bsummary_[A-Za-z0-9_{}%-]*\.json\b"),
        re.compile(r"/summaries/summary_"),
        re.compile(r"\bsummaries\b.*\bsummary_"),
    ),
    "latest_alias": (
        re.compile(r"\blatest\.json\b"),
    ),
    "consolidated_jsonl": (
        re.compile(r"\bphase1_summaries\.jsonl\b"),
    ),
    "manifest_evidence": (
        re.compile(r"\bmanifest\.json\b"),
        re.compile(r"\bmanifest_path\b"),
        re.compile(r"\bseed[_ -]?manifest\b", re.IGNORECASE),
    ),
    "historical_paper1_root": (
        re.compile(r"\bartifacts_paper1_[A-Za-z0-9_./-]*"),
        re.compile(r"\bartifacts_paper1\b"),
        re.compile(r"~/gencys/[A-Za-z0-9_./-]*paper1[A-Za-z0-9_./-]*"),
        re.compile(r"/home/[A-Za-z0-9_.-]+/gencys/[A-Za-z0-9_./-]*paper1"),
    ),
}

WRITE_HINTS = (
    "write_text(", "write_bytes(", "json.dump(", "yaml.safe_dump(",
    ".write(", "to_csv(", "to_json(", "rm -f ", "-delete", "unlink(",
)
READ_HINTS = (
    "read_text(", "read_bytes(", "json.load(", "yaml.safe_load(",
    "glob(", ".glob(", "pd.read_csv(", "read_csv(",
)
REFERENCE_ONLY_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class Match:
    category: str
    line_number: int
    line: str
    access: str


@dataclass(frozen=True)
class ConsumerRecord:
    path: str
    sha256: str
    categories: tuple[str, ...]
    accesses: tuple[str, ...]
    classification: str
    migration_policy: str
    evidence: tuple[Match, ...]


def _run_git(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or "git command failed"
        )
    return completed.stdout


def _is_nonconsumer_data_path(relative_path: str) -> bool:
    """Return whether a tracked path is data/evidence rather than a consumer."""

    if relative_path.startswith("schemas/examples/"):
        return True

    if relative_path.startswith("papers/") and "/results/" in relative_path:
        return True

    return False


def tracked_files(repo_root: Path) -> list[Path]:
    output = _run_git(repo_root, ["ls-files", "-z"])
    paths = []
    for raw in output.split("\0"):
        if not raw:
            continue
        normalized = raw.replace("\\", "/")
        if normalized in EXCLUDED_PATHS:
            continue
        if any(normalized.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        if _is_nonconsumer_data_path(normalized):
            continue
        path = repo_root / raw
        if path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES:
            paths.append(path)
    return sorted(paths, key=lambda p: p.relative_to(repo_root).as_posix())


def _access_for_line(path: Path, line: str) -> str:
    lowered = line.lower()
    if path.suffix.lower() in REFERENCE_ONLY_SUFFIXES:
        return "reference"
    has_write = any(hint.lower() in lowered for hint in WRITE_HINTS)
    has_read = any(hint.lower() in lowered for hint in READ_HINTS)
    if has_write and has_read:
        return "read_write"
    if has_write:
        return "write"
    if has_read:
        return "read"
    return "reference"


def scan_text(path: Path, text: str) -> tuple[Match, ...]:
    found = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        for category, patterns in CATEGORY_PATTERNS.items():
            if any(pattern.search(line) for pattern in patterns):
                match = Match(
                    category=category,
                    line_number=line_number,
                    line=line.strip()[:500],
                    access=_access_for_line(path, line),
                )
                found[(match.category, match.line_number, match.line, match.access)] = match
    return tuple(sorted(
        found.values(),
        key=lambda item: (item.line_number, item.category, item.access, item.line),
    ))


def _file_has_write_signal(path: Path, text: str) -> bool:
    """Return whether a source file contains any mutation/write signal."""

    if path.suffix.lower() in REFERENCE_ONLY_SUFFIXES:
        return False

    lowered = text.lower()

    return any(
        hint.lower() in lowered
        for hint in WRITE_HINTS
    )


def classify(
    relative_path: str,
    evidence: Sequence[Match],
    *,
    file_has_write: bool = False,
) -> tuple[str, str]:
    categories = {item.category for item in evidence}
    accesses = {item.access for item in evidence}

    if "canonical_linkage_interface" in categories:
        return "canonical_consumer", "already_canonical"
    if relative_path in PROTECTED_HISTORICAL_PATHS:
        return "historical_pipeline", "protected_do_not_migrate_in_m6_5_1"
    if relative_path in GOVERNANCE_AUDIT_PATHS:
        return "governance_auditor", "preserve_governance_tool"
    if relative_path.startswith("tests/") or relative_path.startswith(".github/"):
        return "test_or_ci", "review_only"
    if Path(relative_path).suffix.lower() in REFERENCE_ONLY_SUFFIXES:
        return "documentation", "review_only"
    if Path(relative_path).suffix.lower() in CONFIG_SUFFIXES:
        return "configuration_dependency", "review_dependency_no_automatic_migration"
    if file_has_write or accesses & {"write", "read_write"}:
        return "historical_evidence_mutator_or_producer", "review_before_any_migration"
    if categories & {
        "authoritative_score_table", "timestamped_summary", "latest_alias",
        "consolidated_jsonl", "manifest_evidence", "historical_paper1_root",
    }:
        return "legacy_or_direct_consumer", "candidate_for_future_canonical_migration"
    return "reference_only", "review_only"


def scan_file(repo_root: Path, path: Path) -> ConsumerRecord | None:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    evidence = scan_text(path, text)
    if not evidence:
        return None

    relative_path = path.relative_to(repo_root).as_posix()

    categories = {
        match.category
        for match in evidence
    }

    lowered_path = relative_path.lower()

    # Generic manifest references alone do not establish a Paper 1
    # downstream-consumer relationship. Require another Paper 1 anchor unless
    # the repository path itself is explicitly Paper 1-specific.
    if categories == {"manifest_evidence"}:
        if "paper01" not in lowered_path and "paper1" not in lowered_path:
            return None

    # Papers 2-4 have their own summary/latest/manifest surfaces. A generic
    # match inside those paper trees is not a Paper 1 dependency unless the
    # file also carries an explicit Paper 1 anchor.
    if relative_path.startswith(
        (
            "papers/paper2_",
            "papers/paper3_",
            "papers/paper4_",
        )
    ):
        if not categories & EXPLICIT_PAPER1_CATEGORIES:
            return None

    file_has_write = _file_has_write_signal(path, text)
    classification, migration_policy = classify(
        relative_path,
        evidence,
        file_has_write=file_has_write,
    )

    accesses = {
        match.access
        for match in evidence
    }

    if file_has_write:
        accesses.add("write")

    return ConsumerRecord(
        path=relative_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        categories=tuple(sorted({match.category for match in evidence})),
        accesses=tuple(sorted(accesses)),
        classification=classification,
        migration_policy=migration_policy,
        evidence=evidence,
    )


def build_inventory(repo_root: Path) -> list[ConsumerRecord]:
    records = []
    for path in tracked_files(repo_root):
        record = scan_file(repo_root, path)
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda record: record.path)


def _summary_counts(records: Sequence[ConsumerRecord]) -> dict[str, dict[str, int]]:
    categories = {}
    classifications = {}
    migration_policies = {}
    for record in records:
        classifications[record.classification] = classifications.get(record.classification, 0) + 1
        migration_policies[record.migration_policy] = migration_policies.get(record.migration_policy, 0) + 1
        for category in record.categories:
            categories[category] = categories.get(category, 0) + 1
    return {
        "categories": dict(sorted(categories.items())),
        "classifications": dict(sorted(classifications.items())),
        "migration_policies": dict(sorted(migration_policies.items())),
    }


def inventory_payload(repo_root: Path, records: Sequence[ConsumerRecord]) -> dict[str, object]:
    head = _run_git(repo_root, ["rev-parse", "HEAD"]).strip()
    return {
        "audit_version": AUDIT_VERSION,
        "study_id": STUDY_ID,
        "repository_head": head,
        "record_count": len(records),
        "summary": _summary_counts(records),
        "records": [
            {
                **{key: value for key, value in asdict(record).items() if key != "evidence"},
                "evidence": [asdict(item) for item in record.evidence],
            }
            for record in records
        ],
    }


def _ensure_output_dir(repo_root: Path, output_dir: Path) -> Path:
    root = repo_root.resolve()
    target = (output_dir if output_dir.is_absolute() else root / output_dir).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError("output directory must be inside repository") from exc
    expected = DEFAULT_OUTPUT_DIR.as_posix()
    if relative.as_posix() != expected:
        raise ValueError(f"M6.5.1 output directory must be exactly {expected}")
    return target


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: Sequence[ConsumerRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "path", "sha256", "classification", "migration_policy",
            "categories", "accesses", "evidence_count", "evidence_lines",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "path": record.path,
                "sha256": record.sha256,
                "classification": record.classification,
                "migration_policy": record.migration_policy,
                "categories": ";".join(record.categories),
                "accesses": ";".join(record.accesses),
                "evidence_count": len(record.evidence),
                "evidence_lines": ";".join(str(item.line_number) for item in record.evidence),
            })


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    records = payload["records"]
    summary = payload["summary"]
    lines = [
        "# M6.5.1 — Paper 1 Downstream Consumer Inventory",
        "",
        f"- Audit version: `{payload['audit_version']}`",
        f"- Study: `{payload['study_id']}`",
        f"- Repository HEAD: `{payload['repository_head']}`",
        f"- Inventory records: **{payload['record_count']}**",
        "",
        "## Governance boundary",
        "",
        "This audit inventories repository references and consumers only.",
        "It does not migrate code, modify historical artifacts, infer provenance,",
        "or promote any legacy summary/alias to authoritative evidence.",
        "",
        "## Classification census",
        "",
    ]
    for key, value in summary["classifications"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Evidence-category census", ""]
    for key, value in summary["categories"].items():
        lines.append(f"- `{key}`: {value}")
    lines += [
        "", "## Inventory", "",
        "| Path | Classification | Migration policy | Categories | Access |",
        "|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                record["path"], record["classification"], record["migration_policy"],
                ", ".join(record["categories"]), ", ".join(record["accesses"]),
            )
        )
    lines += [
        "", "## Interpretation rules", "",
        "- `canonical_consumer` already references the governed canonical linkage interface.",
        "- `legacy_or_direct_consumer` directly reads or references legacy Paper 1 evidence surfaces.",
        "- `configuration_dependency` declares a Paper 1 storage/evidence dependency but is not executable consumer code.",
        "- `governance_auditor` is TrustForge migration/audit tooling and must not be treated as legacy downstream code.",
        "- `historical_evidence_mutator_or_producer` contains write/mutation signals near Paper 1 evidence references; this is not authorization to change it.",
        "- `historical_pipeline` is explicitly protected from migration in M6.5.1.",
        "- Paper result tables, frozen/raw result data, schema examples, canonical evidence records, and prior audits are excluded from consumer counts.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory tracked repository consumers of Paper 1 evidence.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    if not (repo_root / ".git").exists():
        parser.error(f"not a Git repository root: {repo_root}")

    records = build_inventory(repo_root)
    payload = inventory_payload(repo_root, records)
    summary = payload["summary"]

    print(f"[M6.5.1] Repository HEAD: {payload['repository_head']}")
    print(f"[M6.5.1] Inventory records: {payload['record_count']}")
    for key, value in summary["classifications"].items():
        print(f"[M6.5.1] Classification {key}: {value}")
    for key, value in summary["categories"].items():
        print(f"[M6.5.1] Category {key}: {value}")
    print("[M6.5.1] Historical artifacts modified: NO")
    print("[M6.5.1] Consumer behavior modified: NO")

    if args.dry_run:
        print("[M6.5.1] Dry run: no audit files written.")
        return 0

    output_dir = _ensure_output_dir(repo_root, Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "paper01_downstream_consumer_inventory.json", payload)
    write_csv(output_dir / "paper01_downstream_consumer_inventory.csv", records)
    write_markdown(output_dir / "paper01_downstream_consumer_inventory.md", payload)
    print(f"[M6.5.1] Wrote audit directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
