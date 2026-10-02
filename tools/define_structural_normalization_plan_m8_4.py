#!/usr/bin/env python3
"""
M8.4.1 — TrustForge repository structural normalization design.

This milestone defines the intended normalized repository roles and destinations.
It does NOT move files, rename directories, rewrite imports, or migrate papers.

Generated artifacts:
    studies/repository_migration/audits/m8_4/structural_normalization_plan.yaml
    studies/repository_migration/audits/m8_4/structural_normalization_plan.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


OUTPUT_REL = Path("studies/repository_migration/audits/m8_4")
SCHEMA_REL = OUTPUT_REL / "structural_normalization_plan.schema.yaml"

M8_3_ACCEPTANCE_COMMIT = "c2e06fa"

NORMALIZATION_CLASSES = {
    "CREATE_NATIVE",
    "PRESERVE_CURRENT",
    "MIGRATE_LATER",
    "COMPATIBILITY_ONLY",
    "STUDY_LOCAL",
    "PLUGIN",
    "DEFERRED",
}


def item(
    component: str,
    current_location: str,
    normalization_class: str,
    target_role: str,
    target_location: str | None,
    rationale: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "component": component,
        "current_location": current_location,
        "normalization_class": normalization_class,
        "target_role": target_role,
        "target_location": target_location,
        "move_authorized": False,
        "rename_authorized": False,
        "rewrite_authorized": False,
        "rationale": rationale,
        "evidence": evidence,
    }


ITEMS = [
    item(
        "trustforge_provenance",
        "src/trustforge/provenance/",
        "PRESERVE_CURRENT",
        "Permanent paper-neutral provenance core.",
        "src/trustforge/provenance/",
        "M8.2 already accepted this implementation as CORE_CONTRACT.",
        ["Current TrustForge provenance package contains execution, Git, hashing, manifest, and validation primitives."],
    ),
    item(
        "trustforge_storage",
        "src/trustforge/storage/",
        "PRESERVE_CURRENT",
        "Permanent paper-neutral storage/path core.",
        "src/trustforge/storage/",
        "M8.2 already accepted this implementation as CORE_CONTRACT.",
        ["Current storage package resolves portable roots and canonical artifact layout."],
    ),
    item(
        "trustforge_data_namespace",
        "src/trustforge/data/",
        "CREATE_NATIVE",
        "Paper-neutral dataset identity/loading interfaces.",
        "src/trustforge/data/",
        "Namespace exists but contains only package scaffolding.",
        ["M8.2 defines dataset abstraction as CORE_INTERFACE while common/data.py remains transitional."],
    ),
    item(
        "trustforge_evaluation_namespace",
        "src/trustforge/evaluation/",
        "CREATE_NATIVE",
        "Paper-neutral evaluator contracts and native implementations.",
        "src/trustforge/evaluation/",
        "Namespace exists but historical eval/runner.py is not accepted as core implementation.",
        ["M8.2 classifies evaluation abstraction as CORE_INTERFACE and eval/runner.py as TRANSITIONAL_RUNTIME."],
    ),
    item(
        "trustforge_orchestration_namespace",
        "src/trustforge/orchestration/",
        "CREATE_NATIVE",
        "Paper-neutral train/synth/eval orchestration contracts.",
        "src/trustforge/orchestration/",
        "Namespace exists but current app/main.py remains transitional compatibility runtime.",
        ["M8.2 distinguishes orchestration interface from current historical CLI implementation."],
    ),
    item(
        "trustforge_models_namespace",
        "src/trustforge/models/",
        "CREATE_NATIVE",
        "Plugin/adapter interfaces and model registration contracts, not model implementations.",
        "src/trustforge/models/",
        "Framework core should own plugin contracts while implementations remain plugins.",
        ["M8.2 classifies model implementations as MODEL_PLUGIN rather than core."],
    ),
    item(
        "trustforge_policies_namespace",
        "src/trustforge/policies/",
        "CREATE_NATIVE",
        "Future paper-neutral policy interfaces only.",
        "src/trustforge/policies/",
        "Namespace exists but Paper 4 policy semantics remain study-owned.",
        ["M8.2 classifies Paper 4 selective policies as STUDY_OWNED."],
    ),
    item(
        "trustforge_augmentation_namespace",
        "src/trustforge/augmentation/",
        "CREATE_NATIVE",
        "Future paper-neutral augmentation interfaces only.",
        "src/trustforge/augmentation/",
        "Namespace exists but Paper 3 augmentation semantics remain study-owned.",
        ["M8.2 classifies Paper 3 augmentation behavior as STUDY_OWNED."],
    ),
    item(
        "trustforge_uncertainty_namespace",
        "src/trustforge/uncertainty/",
        "DEFERRED",
        "Future uncertainty/calibration capability.",
        "src/trustforge/uncertainty/",
        "Architecture reserves this area, but current Papers 1–4 migration does not yet require implementation.",
        ["Namespace currently contains only __init__.py."],
    ),
    item(
        "paper01_native_study",
        "studies/paper01_benchmark/",
        "PRESERVE_CURRENT",
        "Canonical TrustForge-native Paper 1 study/evidence package.",
        "studies/paper01_benchmark/",
        "Paper 1 migration is complete and accepted.",
        ["Current study contains canonical experiments, execution evidence, and audit history."],
    ),
    item(
        "paper02_study",
        "papers/paper2_conditional_generation_done_right/",
        "MIGRATE_LATER",
        "Paper 2 canonical TrustForge study package.",
        "studies/paper02_conditioning_audit/",
        "Paper 2 has rich configs/results/scripts/slurm structure but has not yet been migrated.",
        ["Tracked structure includes configs, manifests, notes, paper, results, scripts, and slurm."],
    ),
    item(
        "paper03_study",
        "papers/paper3_when_does_synth_help/",
        "MIGRATE_LATER",
        "Paper 3 canonical TrustForge study package.",
        "studies/paper03_augmentation_regimes/",
        "Paper 3 has rich configs/results/scripts/slurm structure but remains historical active study.",
        ["Tracked structure includes 82 configs, 122 results, 12 scripts, and 38 slurm files."],
    ),
    item(
        "paper04_study",
        "papers/paper4_selective_synth_policies/",
        "MIGRATE_LATER",
        "Paper 4 canonical TrustForge study package.",
        "studies/paper04_selective_policies/",
        "Paper 4 has rich configs/results/scripts/slurm structure but remains historical active study.",
        ["Tracked structure includes configs, results, scripts, and slurm."],
    ),
    item(
        "paper03_placeholder",
        "papers/paper03_when_does_synth_help/",
        "COMPATIBILITY_ONLY",
        "Historical/scaffold placeholder retained until cleanup is explicitly authorized.",
        None,
        "M8.1 classified this location as scaffold-only, distinct from active Paper 3.",
        ["Active Paper 3 is papers/paper3_when_does_synth_help/."],
    ),
    item(
        "paper05_placeholder",
        "papers/paper05_shift_calibration/",
        "DEFERRED",
        "Future study placeholder outside current Papers 1–4 migration.",
        None,
        "Paper 5 is outside current migration scope.",
        ["M8.1 classified this location as scaffold-only."],
    ),
    item(
        "historical_cli_runtime",
        "app/",
        "COMPATIBILITY_ONLY",
        "Historical orchestration compatibility surface.",
        None,
        "M8.2/M8.3 explicitly classify current CLI runtime as transitional, not native core.",
        ["Papers 2–4 historically invoke python -m app.main."],
    ),
    item(
        "historical_adapters",
        "adapters/",
        "COMPATIBILITY_ONLY",
        "Historical concrete adapter compatibility surface.",
        None,
        "Concrete adapters remain tied to historical model packages.",
        ["M8.2 classifies adapter interface as core but concrete adapters as TRANSITIONAL_RUNTIME."],
    ),
    item(
        "historical_dataset_runtime",
        "common/",
        "COMPATIBILITY_ONLY",
        "Historical dataset/runtime compatibility surface.",
        None,
        "common/data.py is multi-paper shared runtime, not accepted native core.",
        ["Papers/models import common.data directly."],
    ),
    item(
        "historical_evaluator",
        "eval/",
        "COMPATIBILITY_ONLY",
        "Historical evaluation compatibility surface.",
        None,
        "eval/runner.py mixes cross-paper and paper-era semantics.",
        ["M8.2 classifies historical evaluator as TRANSITIONAL_RUNTIME."],
    ),
    item(
        "gan_plugin",
        "gan/",
        "PLUGIN",
        "GAN family implementation plugin.",
        None,
        "Model implementation is reusable but not framework core.",
        ["M8.2 classifies historical model implementations as MODEL_PLUGIN."],
    ),
    item(
        "vae_plugin",
        "vae/",
        "PLUGIN",
        "VAE family implementation plugin.",
        None,
        "Model implementation is reusable but not framework core.",
        ["M8.2 classifies historical model implementations as MODEL_PLUGIN."],
    ),
    item(
        "diffusion_plugin",
        "diffusion/",
        "PLUGIN",
        "Diffusion family implementation plugin.",
        None,
        "Model implementation is reusable but not framework core.",
        ["M8.2 classifies historical model implementations as MODEL_PLUGIN."],
    ),
    item(
        "autoregressive_plugin",
        "autoregressive/",
        "PLUGIN",
        "Autoregressive family implementation plugin.",
        None,
        "Model implementation is reusable but not framework core.",
        ["M8.2 classifies historical model implementations as MODEL_PLUGIN."],
    ),
    item(
        "gaussianmixture_plugin",
        "gaussianmixture/",
        "PLUGIN",
        "Gaussian-mixture implementation plugin.",
        None,
        "Model implementation is reusable but not framework core.",
        ["M8.2 classifies historical model implementations as MODEL_PLUGIN."],
    ),
    item(
        "restrictedboltzmann_plugin",
        "restrictedboltzmann/",
        "PLUGIN",
        "Restricted Boltzmann implementation plugin.",
        None,
        "Model implementation is reusable but not framework core.",
        ["M8.2 classifies historical model implementations as MODEL_PLUGIN."],
    ),
    item(
        "maskedautoflow_plugin",
        "maskedautoflow/",
        "PLUGIN",
        "Masked AutoFlow implementation plugin.",
        None,
        "Model implementation is reusable but not framework core.",
        ["M8.2 classifies historical model implementations as MODEL_PLUGIN."],
    ),
    item(
        "generic_schemas",
        "schemas/",
        "PRESERVE_CURRENT",
        "Repository-level TrustForge contract schemas.",
        "schemas/",
        "Generic study/experiment/manifest schemas are already accepted shared core contracts.",
        ["Paper 1 examples remain explicitly separated under schemas/examples/paper01/."],
    ),
    item(
        "manifests",
        "manifests/",
        "PRESERVE_CURRENT",
        "Repository-level manifest schemas/registry area.",
        "manifests/",
        "Existing Paper 1 schema material is explicit and tracked.",
        ["M8.1 distinguished generic schemas from Paper 1-specific manifest schema."],
    ),
    item(
        "configs_root",
        "configs/",
        "MIGRATE_LATER",
        "Future repository-level paper-neutral configuration area after mixed historical content is partitioned.",
        "configs/",
        "The current root is mixed: some files are historical compatibility while the normalized repository still requires a native configs/ surface.",
        [
            "M8.1 left many root configs ambiguous or historical compatibility.",
            "M8.3 requires historical configs to remain compatibility-only rather than become native defaults.",
            "The target architecture retains configs/ as a normalized repository-level surface.",
        ],
    ),
    item(
        "slurm_root",
        "slurm/;run_gcs.slurm;run_suite.slurm;run_tuning.slurm",
        "COMPATIBILITY_ONLY",
        "Historical/global HPC compatibility area.",
        None,
        "Root Slurm contains legacy, one-off, and historical shared runners.",
        ["M8.3 allows historical Slurm only as compatibility, not native default."],
    ),
    item(
        "hpc_native_root",
        "hpc/",
        "CREATE_NATIVE",
        "Future TrustForge-native backend-neutral/local/Slurm execution support.",
        "hpc/",
        "Target architecture calls for hpc/ and M8.2 defines execution backend as a core interface.",
        ["No native hpc/ directory currently exists."],
    ),
    item(
        "scripts_root",
        "scripts/",
        "MIGRATE_LATER",
        "Repository-level general automation only; study-specific scripts should live with studies.",
        "scripts/",
        "Current scripts mix generic utilities, plots, metrics, and Paper 1-era functionality.",
        ["M8.1 classified many scripts as ambiguous and some as Paper 1-specific."],
    ),
    item(
        "tools_root",
        "tools/",
        "PRESERVE_CURRENT",
        "Repository engineering/audit tooling.",
        "tools/",
        "TrustForge migration and audit tooling already lives here.",
        ["M8.1/M8.2/M8.3 generators are repository tooling, not scientific study code."],
    ),
    item(
        "tests_root",
        "tests/",
        "PRESERVE_CURRENT",
        "Repository-level tests, including TrustForge contract tests.",
        "tests/",
        "Current tests/trustforge suite validates foundation and migration contracts.",
        ["Current TrustForge suite contains migration milestone tests."],
    ),
    item(
        "docs_root",
        "docs/",
        "PRESERVE_CURRENT",
        "Architecture, portability, reproducibility, and lineage documentation.",
        "docs/",
        "Current docs define accepted TrustForge architecture and research lineage.",
        ["M8.1 classified core architecture docs as SHARED_TRUSTFORGE_CORE."],
    ),
    item(
        "model_template",
        "model-template/",
        "DEFERRED",
        "Developer/template infrastructure subject to later review.",
        None,
        "M8.1 classified it as operational infrastructure, not scientific core.",
        ["No migration action has yet been authorized."],
    ),
    item(
        "gcs_core_submodule",
        "gcs-core/",
        "DEFERRED",
        "Legacy/external component requiring dedicated review.",
        None,
        "M8.1 left this component ambiguous.",
        ["Current repository tracks gcs-core as a separate top-level component."],
    ),
]


def build_plan() -> dict[str, Any]:
    counts = {name: 0 for name in sorted(NORMALIZATION_CLASSES)}
    for record in ITEMS:
        counts[record["normalization_class"]] += 1

    return {
        "schema_version": 1,
        "milestone": "M8.4.1",
        "title": "TrustForge Repository Structural Normalization Plan",
        "status": "STRUCTURAL_DESIGN_DEFINED",
        "source_isolation_contract": {
            "m8_3_acceptance_commit": M8_3_ACCEPTANCE_COMMIT,
        },
        "authority_boundary": {
            "moves_authorized": False,
            "renames_authorized": False,
            "import_rewrites_authorized": False,
            "paper_migrations_authorized": False,
            "historical_cleanup_authorized": False,
        },
        "principles": [
            "NORMALIZED_TARGET != IMMEDIATE_MOVE",
            "COMPATIBILITY_SURFACE != NATIVE_CORE",
            "STUDY_LOCAL_SCIENCE_STAYS_WITH_STUDY",
            "MODEL_IMPLEMENTATION == PLUGIN",
            "INTERFACE_FIRST_BEFORE_IMPLEMENTATION_MIGRATION",
            "HISTORICAL_EVIDENCE_REMAINS_PRESERVED",
        ],
        "target_repository_shape": [
            "src/trustforge/",
            "studies/paper01_benchmark/",
            "studies/paper02_conditioning_audit/",
            "studies/paper03_augmentation_regimes/",
            "studies/paper04_selective_policies/",
            "configs/",
            "manifests/",
            "schemas/",
            "hpc/",
            "scripts/",
            "tools/",
            "tests/",
            "docs/",
        ],
        "summary": {
            "component_count": len(ITEMS),
            "class_counts": counts,
        },
        "components": ITEMS,
    }


def load_schema(repo: Path) -> dict[str, Any]:
    data = yaml.safe_load((repo / SCHEMA_REL).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("M8.4 schema must be a mapping")
    return data


def validate(data: dict[str, Any], schema: dict[str, Any]) -> None:
    for key in schema["required_top_level"]:
        if key not in data:
            raise ValueError(f"missing top-level field: {key}")

    allowed = set(schema["allowed_normalization_classes"])
    seen: set[str] = set()

    for record in data["components"]:
        for key in schema["component_required"]:
            if key not in record:
                raise ValueError(f"{record.get('component')}: missing {key}")

        if record["component"] in seen:
            raise ValueError(f"duplicate component: {record['component']}")
        seen.add(record["component"])

        if record["normalization_class"] not in allowed:
            raise ValueError(
                f"unknown normalization class: {record['normalization_class']}"
            )

        if record["move_authorized"] is not False:
            raise ValueError("M8.4.1 must not authorize moves")
        if record["rename_authorized"] is not False:
            raise ValueError("M8.4.1 must not authorize renames")
        if record["rewrite_authorized"] is not False:
            raise ValueError("M8.4.1 must not authorize rewrites")

        if not record["current_location"].strip():
            raise ValueError(f"{record['component']}: current_location required")
        if not record["evidence"]:
            raise ValueError(f"{record['component']}: evidence required")
        if not record["rationale"].strip():
            raise ValueError(f"{record['component']}: rationale required")


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# M8.4.1 — TrustForge Repository Structural Normalization Plan",
        "",
        f"**Status:** `{data['status']}`",
        f"**M8.3 acceptance commit:** `{data['source_isolation_contract']['m8_3_acceptance_commit']}`",
        "",
        "## Authority boundary",
        "",
        "This milestone defines normalized structural roles only.",
        "No file move, rename, import rewrite, paper migration, or historical cleanup is authorized.",
        "",
        "## Principles",
        "",
    ]
    lines += [f"- `{p}`" for p in data["principles"]]

    lines += [
        "",
        "## Target repository shape",
        "",
    ]
    lines += [f"- `{p}`" for p in data["target_repository_shape"]]

    lines += [
        "",
        "## Summary",
        "",
        f"Components classified: **{data['summary']['component_count']}**",
        "",
        "| Normalization class | Components |",
        "|---|---:|",
    ]
    lines += [
        f"| `{name}` | {count} |"
        for name, count in data["summary"]["class_counts"].items()
    ]

    lines += ["", "## Component decisions", ""]

    for record in data["components"]:
        lines += [
            f"### `{record['component']}`",
            "",
            f"- Current location: `{record['current_location']}`",
            f"- Normalization class: `{record['normalization_class']}`",
            f"- Target role: {record['target_role']}",
            f"- Target location: `{record['target_location']}`"
            if record["target_location"] is not None
            else "- Target location: `deferred`",
            "- Move authorized: `false`",
            "- Rename authorized: `false`",
            "- Rewrite authorized: `false`",
            f"- Rationale: {record['rationale']}",
            "- Evidence:",
        ]
        lines += [f"  - {entry}" for entry in record["evidence"]]
        lines += [""]

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(repo: Path, data: dict[str, Any]) -> list[Path]:
    out = repo / OUTPUT_REL
    out.mkdir(parents=True, exist_ok=True)

    yaml_path = out / "structural_normalization_plan.yaml"
    md_path = out / "structural_normalization_plan.md"

    yaml_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(data), encoding="utf-8")
    return [yaml_path, md_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()

    data = build_plan()
    schema = load_schema(repo)
    validate(data, schema)

    print(f"[m8.4.1] status={data['status']}")
    print(f"[m8.4.1] component_count={data['summary']['component_count']}")
    for cls, count in data["summary"]["class_counts"].items():
        print(f"[m8.4.1] {cls}={count}")

    if args.check_only:
        print("[m8.4.1] check-only: no generated artifacts written")
        return 0

    for path in write_outputs(repo, data):
        print(f"[m8.4.1] wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
