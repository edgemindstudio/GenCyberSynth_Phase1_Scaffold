#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import yaml

OUTPUT_REL = Path("studies/repository_migration/audits/m8_2")
SCHEMA_REL = OUTPUT_REL / "shared_core_boundary.schema.yaml"
M8_1_BASELINE = "d60f9c97e6b01b78cd615aa0fdb4aa61f3acce51"
M8_1_ACCEPTANCE_COMMIT = "9487df0"

BOUNDARY_CLASSES = {
    "CORE_CONTRACT",
    "CORE_INTERFACE",
    "TRANSITIONAL_RUNTIME",
    "MODEL_PLUGIN",
    "STUDY_OWNED",
    "HISTORICAL_COMPATIBILITY",
    "DEFERRED",
}

def rec(capability, locations, current_role, future_role, boundary_class, rationale, evidence):
    return {
        "capability": capability,
        "current_locations": locations,
        "current_role": current_role,
        "future_role": future_role,
        "boundary_class": boundary_class,
        "migration_authorized": False,
        "implementation_move_authorized": False,
        "evidence": evidence,
        "rationale": rationale,
    }

CAPABILITIES = [
    rec(
        "study_identity_contract",
        ["schemas/study.schema.yaml", "docs/REPRODUCIBILITY_CONTRACT.md"],
        "Defines stable study identity and scientific scope.",
        "Permanent TrustForge study identity contract.",
        "CORE_CONTRACT",
        "Study identity is framework-wide and not owned by one paper.",
        ["STUDY is distinct from CODE, CONFIG, MANIFEST, and ARTIFACT.",
         "Study identity is stable independently of manuscript filenames."],
    ),
    rec(
        "experiment_identity_contract",
        ["schemas/experiment.schema.yaml", "docs/REPRODUCIBILITY_CONTRACT.md"],
        "Defines exact scientific intent including dataset, model, seed, evaluation, and scientific parameters.",
        "Permanent TrustForge experiment-definition contract.",
        "CORE_CONTRACT",
        "Experiment identity must remain paper-neutral and machine-independent.",
        ["Experiment describes scientific intent.",
         "Experiment identity is distinct from execution identity."],
    ),
    rec(
        "execution_identity_contract",
        ["src/trustforge/provenance/execution.py", "schemas/manifest.schema.yaml"],
        "Defines one concrete execution attempt.",
        "Permanent TrustForge execution-identity contract.",
        "CORE_CONTRACT",
        "Execution identity is operational provenance and paper-neutral.",
        ["Execution realizes an Experiment.",
         "Execution identifiers are not scientific experiment identifiers."],
    ),
    rec(
        "storage_resolution",
        ["src/trustforge/storage/paths.py", "src/trustforge/storage/layout.py"],
        "Resolves portable roots and canonical dataset/study/experiment/execution paths.",
        "Permanent TrustForge storage contract and implementation.",
        "CORE_CONTRACT",
        "Storage semantics are already paper-neutral TrustForge code.",
        ["Scientific identity is separate from machine-specific paths.",
         "Canonical paths derive from stable identifiers."],
    ),
    rec(
        "provenance_hashing",
        ["src/trustforge/provenance/hashing.py"],
        "Provides SHA256 exact-file and canonical structured-data hashing.",
        "Permanent TrustForge provenance primitive.",
        "CORE_CONTRACT",
        "Hashing semantics apply across all studies.",
        ["Exact file hashes and canonical structured hashes answer different provenance questions."],
    ),
    rec(
        "git_provenance",
        ["src/trustforge/provenance/git.py"],
        "Captures commit, branch, repository root, and dirty state read-only.",
        "Permanent TrustForge execution provenance primitive.",
        "CORE_CONTRACT",
        "Git provenance is independent of study-specific science.",
        ["Executions must record exact source identity.",
         "Dirty state must not be represented as clean."],
    ),
    rec(
        "native_manifest_contract",
        ["schemas/manifest.schema.yaml", "src/trustforge/provenance/manifest.py",
         "src/trustforge/provenance/validation.py"],
        "Defines and validates execution reality.",
        "Permanent TrustForge manifest contract and native implementation.",
        "CORE_CONTRACT",
        "Execution manifests are universal framework records.",
        ["CONFIG describes intended work; MANIFEST describes execution reality.",
         "Manifest completion does not itself establish scientific acceptance."],
    ),
    rec(
        "accepted_evidence_contract",
        ["src/trustforge/paper01_execution_evidence.py",
         "studies/paper01_benchmark/execution_evidence/"],
        "Paper 1 implementation separates production, evaluation, and authoritative results.",
        "Future paper-neutral accepted-evidence contract.",
        "DEFERRED",
        "The concept belongs in core, but current implementation is Paper 1-specific.",
        ["Operational completion is not accepted scientific evidence.",
         "Paper 1 demonstrates producer/evaluator/result separation."],
    ),
    rec(
        "adapter_interface",
        ["adapters/base.py"],
        "Defines generic Adapter and synth(config)->manifest semantics.",
        "TrustForge-native model adapter interface.",
        "CORE_INTERFACE",
        "The abstraction is reusable even though the implementation is historical runtime.",
        ["Adapter defines a generic synthesis contract."],
    ),
    rec(
        "adapter_registry_interface",
        ["adapters/registry.py"],
        "Registers and resolves adapters by model key.",
        "TrustForge-native plugin registry interface.",
        "CORE_INTERFACE",
        "Registry semantics belong in core; historical registrations do not.",
        ["Registry is implementation-agnostic at its public interface."],
    ),
    rec(
        "orchestration_interface",
        ["app/main.py"],
        "Routes train, synth, eval, configuration, paper scoping, and runtime metadata.",
        "TrustForge-native orchestration interface.",
        "CORE_INTERFACE",
        "Train/synth/eval orchestration is core as an interface, not as the current CLI implementation.",
        ["Current app.main mixes generic orchestration with paper-era behavior."],
    ),
    rec(
        "evaluation_interface",
        ["eval/runner.py"],
        "Historical evaluator mixes metrics, manifests, paths, config interpretation, and paper-era behavior.",
        "TrustForge-native evaluator request/result interface.",
        "CORE_INTERFACE",
        "Evaluation is cross-study, but the current monolith is not accepted wholesale as core.",
        ["eval/runner.py accumulated Paper 1, Paper 3, and Paper 4 behavior."],
    ),
    rec(
        "dataset_interface",
        ["common/data.py"],
        "Historical shared loading, normalization, label encoding, and dataset construction.",
        "TrustForge-native dataset representation/loading interface.",
        "CORE_INTERFACE",
        "Dataset abstraction is core; current loader remains transitional.",
        ["Multiple historical model trainers import common.data."],
    ),
    rec(
        "execution_backend_interface",
        ["docs/TRUSTFORGE_ARCHITECTURE.md", "slurm/"],
        "Historical Slurm scripts plus an architecture-level HPC contract.",
        "Backend-neutral execution plan with local and Slurm backends.",
        "CORE_INTERFACE",
        "Execution backend abstraction is core; current Slurm scripts require later review.",
        ["HPC is an execution backend, not scientific experiment definition."],
    ),
    rec(
        "historical_shared_cli_runtime",
        ["app/main.py"],
        "Shared CLI incrementally modified across papers.",
        "Compatibility runtime during migration.",
        "TRANSITIONAL_RUNTIME",
        "Shared historical use does not establish permanent core ownership.",
        ["Git history shows Paper 1, Paper 2, and Paper 3 changes."],
    ),
    rec(
        "historical_shared_evaluator",
        ["eval/runner.py"],
        "Cross-paper evaluator with accumulated historical semantics.",
        "Compatibility runtime until native evaluation contracts exist.",
        "TRANSITIONAL_RUNTIME",
        "The implementation is coupled to historical study semantics.",
        ["Git history includes Paper 1 fixes, Paper 3 changes, and Paper 4 behavior."],
    ),
    rec(
        "historical_shared_dataset_runtime",
        ["common/data.py"],
        "Shared historical loader consumed by multiple trainers.",
        "Compatibility runtime until native dataset interface exists.",
        "TRANSITIONAL_RUNTIME",
        "Shared use does not prove permanent API suitability.",
        ["GAN, VAE, Diffusion, and Autoregressive trainers import common.data."],
    ),
    rec(
        "historical_concrete_adapters",
        ["adapters/gan_adapter.py", "adapters/vae_adapter.py", "adapters/diffusion_adapter.py",
         "adapters/autoregressive_adapter.py", "adapters/gaussianmixture_adapter.py",
         "adapters/restrictedboltzmann_adapter.py", "adapters/maskedautoflow_adapter.py"],
        "Concrete bridges from shared CLI into historical model packages.",
        "Compatibility adapters or validated future plugins.",
        "TRANSITIONAL_RUNTIME",
        "Concrete adapters are implementations, not the adapter abstraction.",
        ["Concrete adapters import historical model-specific packages."],
    ),
    rec(
        "historical_model_implementations",
        ["gan/", "vae/", "diffusion/", "autoregressive/", "gaussianmixture/",
         "restrictedboltzmann/", "maskedautoflow/"],
        "Seven historical generative-model implementations.",
        "Replaceable model plugins outside permanent core.",
        "MODEL_PLUGIN",
        "Reusable model code is a plugin, not framework core.",
        ["Model implementations are replaceable while TrustForge contracts remain stable."],
    ),
    rec(
        "paper02_conditioning_study",
        ["papers/paper2_conditional_generation_done_right/", "gan/models_acgan.py", "gan/train_acgan.py"],
        "Conditional-generation interventions and audits.",
        "Paper 2 study-owned implementation and evidence.",
        "STUDY_OWNED",
        "These semantics answer Paper 2's scientific question.",
        ["Paper 2 owns conditioning/class-faithfulness audit behavior."],
    ),
    rec(
        "paper03_augmentation_study",
        ["papers/paper3_when_does_synth_help/", "gan/sample.py", "vae/sample.py",
         "adapters/diffusion_adapter.py"],
        "Augmentation budgets, minority targeting, and class-restricted synthesis.",
        "Paper 3 study-owned implementation and evidence.",
        "STUDY_OWNED",
        "Paper-specific interventions must remain explicit study semantics.",
        ["Paper 3 owns augmentation regimes and minority-heavy c4/c7 behavior."],
    ),
    rec(
        "paper04_selective_policy_study",
        ["papers/paper4_selective_synth_policies/", "eval/runner.py"],
        "Keep-all, confidence, top-k, and class-repair interventions.",
        "Paper 4 study-owned implementation and evidence.",
        "STUDY_OWNED",
        "Policy semantics belong to Paper 4 unless later generalized explicitly.",
        ["Paper 4 treats synthetic inclusion as a policy-controlled intervention."],
    ),
    rec(
        "legacy_path_and_config_compatibility",
        ["configs/paper1_*.yaml", "configs/paper2_500.yaml", "configs/paper2_1000.yaml",
         "configs/paper2_2000.yaml", "slurm/legacy/", "slurm/oneoffs/"],
        "Historical reproduction and compatibility surfaces.",
        "Preserved compatibility until evidence permits retirement.",
        "HISTORICAL_COMPATIBILITY",
        "Historical path/config state may be provenance and must not be rewritten blindly.",
        ["Migration doctrine preserves historical path-bearing evidence."],
    ),
]

def build_boundary():
    counts = {c: 0 for c in sorted(BOUNDARY_CLASSES)}
    for item in CAPABILITIES:
        counts[item["boundary_class"]] += 1
    return {
        "schema_version": 1,
        "milestone": "M8.2",
        "title": "TrustForge Shared Core Boundary Definition",
        "status": "BOUNDARY_DEFINED",
        "source_baseline": {
            "m8_1_repository_baseline": M8_1_BASELINE,
            "m8_1_acceptance_commit": M8_1_ACCEPTANCE_COMMIT,
        },
        "authority_boundary": {
            "migration_authorized": False,
            "implementation_moves_authorized": False,
            "historical_runtime_rewrites_authorized": False,
            "paper_migrations_authorized": False,
        },
        "principles": [
            "MULTI_PAPER_SHARED_RUNTIME != SHARED_TRUSTFORGE_CORE",
            "REUSABLE != CORE",
            "INTERFACE_OWNERSHIP != CURRENT_IMPLEMENTATION_OWNERSHIP",
            "SCIENTIFIC_EXPERIMENT_IDENTITY != HISTORICAL_EXECUTION_IDENTITY",
            "HPC_BACKEND != SCIENTIFIC_EXPERIMENT_DEFINITION",
            "HISTORICAL_COMPATIBILITY != NATIVE_TRUSTFORGE_CORE",
        ],
        "boundary_class_definitions": {
            "CORE_CONTRACT": "Permanent paper-neutral TrustForge contract.",
            "CORE_INTERFACE": "Permanent paper-neutral capability/interface; current historical implementation is not automatically native core.",
            "TRANSITIONAL_RUNTIME": "Historically shared implementation retained during migration; not permanent core by current evidence.",
            "MODEL_PLUGIN": "Replaceable model-family implementation outside framework core.",
            "STUDY_OWNED": "Scientific behavior or interpretation owned by a specific study.",
            "HISTORICAL_COMPATIBILITY": "Preserved compatibility/reproduction surface.",
            "DEFERRED": "Concept needed later, but evidence is insufficient to freeze a paper-neutral contract now.",
        },
        "summary": {"capability_count": len(CAPABILITIES), "class_counts": counts},
        "capabilities": CAPABILITIES,
    }

def load_schema(repo):
    data = yaml.safe_load((repo / SCHEMA_REL).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("M8.2 schema must be a mapping")
    return data

def validate(data, schema):
    for key in schema["required_top_level"]:
        if key not in data:
            raise ValueError(f"missing top-level field: {key}")
    allowed = set(schema["allowed_boundary_classes"])
    for item in data["capabilities"]:
        for key in schema["capability_required"]:
            if key not in item:
                raise ValueError(f"{item.get('capability')}: missing {key}")
        if item["boundary_class"] not in allowed:
            raise ValueError(f"unknown boundary class: {item['boundary_class']}")
        if item["migration_authorized"] is not False:
            raise ValueError("M8.2 cannot authorize migration")
        if item["implementation_move_authorized"] is not False:
            raise ValueError("M8.2 cannot authorize moves")
        if not item["current_locations"] or not item["evidence"] or not item["rationale"].strip():
            raise ValueError(f"incomplete capability: {item['capability']}")

def render_markdown(data):
    lines = [
        "# M8.2 — TrustForge Shared Core Boundary Definition", "",
        f"**Status:** `{data['status']}`",
        f"**M8.1 baseline:** `{data['source_baseline']['m8_1_repository_baseline']}`",
        f"**M8.1 acceptance commit:** `{data['source_baseline']['m8_1_acceptance_commit']}`",
        "", "## Authority boundary", "",
        "M8.2 defines architectural ownership only.",
        "It does not authorize file moves, runtime rewrites, or paper migrations.",
        "", "## Governing principles", "",
    ]
    lines += [f"- `{p}`" for p in data["principles"]]
    lines += ["", "## Summary", "", "| Boundary class | Capabilities |", "|---|---:|"]
    lines += [f"| `{k}` | {v} |" for k, v in data["summary"]["class_counts"].items()]
    lines += ["", "## Capability decisions", ""]
    for item in data["capabilities"]:
        lines += [
            f"### `{item['capability']}`", "",
            f"- Boundary class: `{item['boundary_class']}`",
            f"- Current role: {item['current_role']}",
            f"- Future role: {item['future_role']}",
            "- Migration authorized: `false`",
            "- Implementation move authorized: `false`",
            f"- Rationale: {item['rationale']}",
            "- Current locations:",
        ]
        lines += [f"  - `{x}`" for x in item["current_locations"]]
        lines += ["- Evidence:"]
        lines += [f"  - {x}" for x in item["evidence"]]
        lines += [""]
    return "\n".join(lines).rstrip() + "\n"

def write_outputs(repo, data):
    out = repo / OUTPUT_REL
    out.mkdir(parents=True, exist_ok=True)
    y = out / "shared_core_boundary.yaml"
    m = out / "shared_core_boundary.md"
    y.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    m.write_text(render_markdown(data), encoding="utf-8")
    return [y, m]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    data = build_boundary()
    schema = load_schema(repo)
    validate(data, schema)
    print(f"[m8.2] status={data['status']}")
    print(f"[m8.2] capability_count={data['summary']['capability_count']}")
    for k, v in data["summary"]["class_counts"].items():
        print(f"[m8.2] {k}={v}")
    if args.check_only:
        print("[m8.2] check-only: no generated artifacts written")
        return 0
    for p in write_outputs(repo, data):
        print(f"[m8.2] wrote {p}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
