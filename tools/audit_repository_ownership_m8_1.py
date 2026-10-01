#!/usr/bin/env python3
"""
M8.1 — Repository-wide ownership and dependency inventory.

READ-ONLY with respect to historical/scientific assets. The only writes are
generated audit artifacts under:

    studies/repository_migration/audits/m8_1/

This audit records observed repository structure and dependency signals.
It does NOT assign future owners, authorize moves, or perform migration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required for M8.1 audit generation") from exc


CLASSIFICATIONS = {
    "SHARED_TRUSTFORGE_CORE",
    "PAPER01_SPECIFIC",
    "PAPER02_SPECIFIC",
    "PAPER03_SPECIFIC",
    "PAPER04_SPECIFIC",
    "MULTI_PAPER_SHARED_RUNTIME",
    "HISTORICAL_COMPATIBILITY",
    "OPERATIONAL_INFRASTRUCTURE",
    "SCAFFOLD_ONLY",
    "AMBIGUOUS_REQUIRES_REVIEW",
}

BASELINE_REF = "d60f9c97e6b01b78cd615aa0fdb4aa61f3acce51"

OUTPUT_REL = Path("studies/repository_migration/audits/m8_1")

PAPER_REF_RE = re.compile(
    r"paper[ _-]?0?[1234]|paper1|paper2|paper3|paper4",
    re.IGNORECASE,
)
TRUSTFORGE_IMPORT_RE = re.compile(r"\b(?:from|import)\s+trustforge(?:\.|\b)")
ROOT_IMPORT_RE = re.compile(
    r"\b(?:from|import)\s+"
    r"(adapters|common|eval|gan|vae|diffusion|autoregressive|"
    r"gaussianmixture|restrictedboltzmann|maskedautoflow)"
    r"(?:\.|\s|$)"
)
APP_MAIN_RE = re.compile(r"\bpython(?:3)?\s+-m\s+app\.main\b")


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def verify_baseline(repo: Path, ref: str) -> str:
    return run_git(repo, "rev-parse", f"{ref}^{{commit}}").strip()


def current_repository_identity(repo: Path) -> dict[str, str]:
    return {
        "current_head": run_git(repo, "rev-parse", "HEAD").strip(),
        "branch": run_git(repo, "branch", "--show-current").strip(),
    }


def tracked_files_at_ref(repo: Path, ref: str) -> list[str]:
    return sorted(
        line.strip()
        for line in run_git(repo, "ls-tree", "-r", "--name-only", ref).splitlines()
        if line.strip()
    )

def classify_path(path: str) -> str:
    p = path.replace("\\", "/")
    name = Path(p).name

    # ------------------------------------------------------------------
    # Canonical TrustForge shared contracts / foundation infrastructure.
    # ------------------------------------------------------------------
    if p.startswith("src/trustforge/provenance/"):
        return "SHARED_TRUSTFORGE_CORE"
    if p.startswith("src/trustforge/storage/"):
        return "SHARED_TRUSTFORGE_CORE"

    if p in {
        "schemas/experiment.schema.yaml",
        "schemas/manifest.schema.yaml",
        "schemas/study.schema.yaml",
        "docs/REPRODUCIBILITY_CONTRACT.md",
        "docs/RESEARCH_LINEAGE.md",
        "docs/STORAGE_AND_PATHS.md",
        "docs/TRUSTFORGE_ARCHITECTURE.md",
        "docs/PORTABILITY_INVENTORY.md",
        "scripts/trustforge_foundation_check.py",
        "scripts/trustforge_doctor.py",
    }:
        return "SHARED_TRUSTFORGE_CORE"

    if p.startswith("src/trustforge/") and name == "__init__.py":
        return "SHARED_TRUSTFORGE_CORE"

    # ------------------------------------------------------------------
    # Canonical Paper 1.
    # ------------------------------------------------------------------
    if p.startswith("studies/paper01_benchmark/"):
        return "PAPER01_SPECIFIC"
    if p.startswith("src/trustforge/paper01_"):
        return "PAPER01_SPECIFIC"
    if p.startswith("schemas/examples/paper01/"):
        return "PAPER01_SPECIFIC"
    if p == "manifests/schemas/paper01_execution_evidence_linkage.schema.json":
        return "PAPER01_SPECIFIC"
    if p == "docs/paper01_execution_evidence_linkage_schema.md":
        return "PAPER01_SPECIFIC"

    if "paper01_" in name or "paper1_" in name:
        if p.startswith(("scripts/", "tools/", "tests/trustforge/")):
            return "PAPER01_SPECIFIC"

    # Phase-1 utilities are Paper-1-specific consumers/compatibility tooling.
    if p in {
        "scripts/phase1_html.py",
        "scripts/phase1_report.py",
        "tools/aggregate_phase1.py",
        "tools/build_phase1_scores.py",
        "tools/check_phase1_integrity.py",
        "tools/freeze_phase1_snapshots.py",
        "tools/normalize_phase1_json.py",
        "tools/phase1_gate.sh",
        "tools/phase1_reval_all.py",
        "tools/phase1_sync_and_reval.py",
    }:
        return "PAPER01_SPECIFIC"

    # ------------------------------------------------------------------
    # Historical active studies.
    # ------------------------------------------------------------------
    if p.startswith("papers/paper2_conditional_generation_done_right/"):
        return "PAPER02_SPECIFIC"
    if p.startswith("papers/paper3_when_does_synth_help/"):
        return "PAPER03_SPECIFIC"
    if p.startswith("papers/paper4_selective_synth_policies/"):
        return "PAPER04_SPECIFIC"

    # Placeholder/scaffold study directories.
    if p.startswith("papers/paper03_when_does_synth_help/"):
        return "SCAFFOLD_ONLY"
    if p.startswith("papers/paper05_shift_calibration/"):
        return "SCAFFOLD_ONLY"

    # ------------------------------------------------------------------
    # Historical/compatibility configs and execution artifacts.
    # ------------------------------------------------------------------
    if re.fullmatch(r"configs/paper1_.*\.ya?ml", p):
        return "HISTORICAL_COMPATIBILITY"
    if p in {
        "configs/paper2_500.yaml",
        "configs/paper2_1000.yaml",
        "configs/paper2_2000.yaml",
        "slurm/run_paper1.slurm",
        "slurm/backfill_paper1_talon-gpu32.slurm",
    }:
        return "HISTORICAL_COMPATIBILITY"
    if p.startswith("slurm/legacy/") or p.startswith("slurm/oneoffs/"):
        return "HISTORICAL_COMPATIBILITY"

    # ------------------------------------------------------------------
    # Observed runtime shared by Papers 2–4.
    # ------------------------------------------------------------------
    shared_roots = (
        "app/",
        "adapters/",
        "common/",
        "eval/",
        "gan/",
        "vae/",
        "diffusion/",
        "autoregressive/",
        "gaussianmixture/",
        "restrictedboltzmann/",
        "maskedautoflow/",
    )
    if p.startswith(shared_roots):
        return "MULTI_PAPER_SHARED_RUNTIME"
    if p == "configs/config.yaml":
        return "MULTI_PAPER_SHARED_RUNTIME"
    if p == "Makefile":
        return "MULTI_PAPER_SHARED_RUNTIME"

    # ------------------------------------------------------------------
    # Repository / CI / development operational infrastructure.
    # This is operational ownership, not scientific ownership.
    # ------------------------------------------------------------------
    if p in {
        ".gitattributes",
        ".gitignore",
        ".gitmodules",
        "README.md",
        "Runbook.md",
        "requirements.ci.txt",
        "requirements.txt",
        "repo_tree.py",
        "tests/test_smoke.py",
    }:
        return "OPERATIONAL_INFRASTRUCTURE"
    if p.startswith(".github/"):
        return "OPERATIONAL_INFRASTRUCTURE"
    if p.startswith("model-template/"):
        return "OPERATIONAL_INFRASTRUCTURE"

    return "AMBIGUOUS_REQUIRES_REVIEW"


def is_text_candidate(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in {
        ".py", ".sh", ".slurm", ".sbatch", ".yaml", ".yml", ".md",
        ".txt", ".json", ".toml", ".ini", ".cfg", ".make", ""
    }


def read_text_at_ref(repo: Path, ref: str, rel: str) -> str | None:
    if not is_text_candidate(rel):
        return None
    try:
        proc = subprocess.run(
            ["git", "show", f"{ref}:{rel}"],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        return None

    if len(proc.stdout) > 2_000_000:
        return None

    return proc.stdout.decode("utf-8", errors="replace")

def dependency_signals(repo: Path, ref: str, files: Iterable[str]) -> dict[str, Any]:
    paper_refs: Counter[str] = Counter()
    trustforge_imports: list[dict[str, Any]] = []
    root_imports: list[dict[str, Any]] = []
    app_main_invocations: list[dict[str, Any]] = []

    for rel in files:
        text = read_text_at_ref(repo, ref, rel)
        if text is None:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if PAPER_REF_RE.search(line):
                top = rel.split("/", 1)[0]
                paper_refs[top] += 1

            if TRUSTFORGE_IMPORT_RE.search(line):
                trustforge_imports.append(
                    {"path": rel, "line": lineno, "text": line.strip()[:500]}
                )

            root_match = ROOT_IMPORT_RE.search(line)
            if root_match:
                root_imports.append(
                    {
                        "path": rel,
                        "line": lineno,
                        "module": root_match.group(1),
                        "text": line.strip()[:500],
                    }
                )

            if APP_MAIN_RE.search(line):
                app_main_invocations.append(
                    {"path": rel, "line": lineno, "text": line.strip()[:500]}
                )

    def stable(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda x: (x["path"], x["line"], x.get("module", "")))

    return {
        "paper_reference_counts_by_top_level": dict(sorted(paper_refs.items())),
        "trustforge_imports": stable(trustforge_imports),
        "root_runtime_imports": stable(root_imports),
        "app_main_invocations": stable(app_main_invocations),
    }


def component_rollup(files: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rel in files:
        top = rel.split("/", 1)[0]
        cls = classify_path(rel)
        grouped[(top, cls)].append(rel)

    rows = []
    for (top, cls), members in sorted(grouped.items()):
        rows.append(
            {
                "top_level_component": top,
                "classification": cls,
                "tracked_file_count": len(members),
                "examples": sorted(members)[:8],
            }
        )
    return rows


def build_inventory(repo: Path, baseline_ref: str = BASELINE_REF) -> dict[str, Any]:
    baseline_commit = verify_baseline(repo, baseline_ref)
    files = tracked_files_at_ref(repo, baseline_commit)

    by_class: dict[str, list[str]] = defaultdict(list)
    for rel in files:
        by_class[classify_path(rel)].append(rel)

    classified_files = [
        {
            "path": rel,
            "classification": classify_path(rel),
            "future_owner": None,
            "migration_action": None,
            "authority_decision": None,
        }
        for rel in files
    ]

    return {
        "schema_version": 3,
        "milestone": "M8.1",
        "title": "Repository-wide ownership and dependency inventory",
        "mode": "READ_ONLY_AUDIT",
        "snapshot": {
            "baseline_ref": baseline_ref,
            "baseline_commit": baseline_commit,
            "tracked_file_count": len(files),
        },
        "generation_context": current_repository_identity(repo),
        "authority_boundary": {
            "future_owner_assignments_performed": False,
            "migration_actions_authorized": False,
            "historical_artifacts_modified": False,
            "authority_decision": None,
        },
        "classification_definitions": {
            "SHARED_TRUSTFORGE_CORE": "Observed paper-neutral TrustForge contracts/foundation core.",
            "PAPER01_SPECIFIC": "Canonical Paper 1 study/evidence/consumer implementation.",
            "PAPER02_SPECIFIC": "Paper 2 study-local implementation/evidence.",
            "PAPER03_SPECIFIC": "Paper 3 study-local implementation/evidence.",
            "PAPER04_SPECIFIC": "Paper 4 study-local implementation/evidence.",
            "MULTI_PAPER_SHARED_RUNTIME": "Historically shared runtime used across multiple papers; not automatically permanent core.",
            "HISTORICAL_COMPATIBILITY": "Historical or compatibility infrastructure preserved for provenance/reproduction.",
            "OPERATIONAL_INFRASTRUCTURE": "Repository, CI, developer, template, or smoke infrastructure; not assigned scientific ownership.",
            "SCAFFOLD_ONLY": "Placeholder/scaffold structure, not the authoritative active study implementation.",
            "AMBIGUOUS_REQUIRES_REVIEW": "Ownership is not yet resolved by M8.1 evidence.",
        },
        "summary": {
            "tracked_file_count": len(files),
            "classification_counts": {
                cls: len(by_class.get(cls, []))
                for cls in sorted(CLASSIFICATIONS)
            },
        },
        "component_rollup": component_rollup(files),
        "dependency_signals": dependency_signals(repo, baseline_commit, files),
        "files": classified_files,
    }

def render_markdown(inv: dict[str, Any]) -> str:
    lines: list[str] = []
    lines += [
        "# M8.1 — Repository Ownership & Dependency Inventory",
        "",
        f"**Mode:** {inv['mode']}",
        f"**Baseline commit:** `{inv['snapshot']['baseline_commit']}`",
        f"**Generation branch:** `{inv['generation_context']['branch']}`",
        "",
        "## Snapshot boundary",
        "",
        "This audit is pinned to the accepted pre-M8 repository snapshot.",
        "Its classification counts remain stable after M8.1 itself is committed.",
        "",
        "## Authority boundary",
        "",
        "This audit records observed ownership/dependency evidence only.",
        "It does not assign future owners, authorize moves, or perform migration.",
        "",
        "## Classification summary",
        "",
        "| Classification | Files |",
        "|---|---:|",
    ]
    for cls, count in inv["summary"]["classification_counts"].items():
        lines.append(f"| `{cls}` | {count} |")

    lines += [
        "",
        "## Top-level component rollup",
        "",
        "| Component | Classification | Tracked files |",
        "|---|---|---:|",
    ]
    for row in inv["component_rollup"]:
        lines.append(
            f"| `{row['top_level_component']}` | `{row['classification']}` | "
            f"{row['tracked_file_count']} |"
        )

    signals = inv["dependency_signals"]
    lines += [
        "",
        "## Dependency signals",
        "",
        f"- TrustForge imports observed: {len(signals['trustforge_imports'])}",
        f"- Root-runtime imports observed: {len(signals['root_runtime_imports'])}",
        f"- `python -m app.main` invocations observed: {len(signals['app_main_invocations'])}",
        "",
        "### Paper-reference counts by top-level component",
        "",
        "| Component | Reference lines |",
        "|---|---:|",
    ]
    for comp, count in signals["paper_reference_counts_by_top_level"].items():
        lines.append(f"| `{comp}` | {count} |")

    lines += [
        "",
        "## M8.1 interpretation guardrails",
        "",
        "- `MULTI_PAPER_SHARED_RUNTIME` does not imply `SHARED_TRUSTFORGE_CORE`.",
        "- `OPERATIONAL_INFRASTRUCTURE` is not a scientific ownership assignment.",
        "- Current location does not determine future ownership.",
        "- Imports/invocations are dependency evidence, not migration authorization.",
        "- Historical study files and path-bearing evidence remain untouched.",
        "- `future_owner`, `migration_action`, and `authority_decision` remain null.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(repo: Path, inv: dict[str, Any]) -> list[Path]:
    out = repo / OUTPUT_REL
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "repository_dependency_inventory.json"
    yaml_path = out / "repository_ownership_inventory.yaml"
    md_path = out / "repository_ownership_inventory.md"

    json_path.write_text(
        json.dumps(inv, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    yaml_path.write_text(
        yaml.safe_dump(
            inv,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(inv), encoding="utf-8")

    return [json_path, yaml_path, md_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--baseline-ref",
        default=BASELINE_REF,
        help="Git commit/tree to audit. Defaults to the accepted pre-M8 snapshot.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build and validate inventory in memory; write no files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()

    if not (repo / ".git").exists():
        raise SystemExit(f"Not a Git repository root: {repo}")

    inv = build_inventory(repo, args.baseline_ref)

    unknown = {
        row["classification"]
        for row in inv["files"]
        if row["classification"] not in CLASSIFICATIONS
    }
    if unknown:
        raise SystemExit(f"Unknown classifications produced: {sorted(unknown)}")

    print(f"[m8.1] baseline={inv['snapshot']['baseline_commit']}")
    print(f"[m8.1] current_head={inv['generation_context']['current_head']}")
    print(f"[m8.1] branch={inv['generation_context']['branch']}")
    print(f"[m8.1] tracked_files={inv['summary']['tracked_file_count']}")
    for cls, count in inv["summary"]["classification_counts"].items():
        print(f"[m8.1] {cls}={count}")

    if args.check_only:
        print("[m8.1] check-only: no files written")
        return 0

    written = write_outputs(repo, inv)
    for path in written:
        print(f"[m8.1] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
