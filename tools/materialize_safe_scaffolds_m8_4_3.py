#!/usr/bin/env python3
"""
M8.4.3 — Materialize only the safe scaffolds authorized by M8.4.2.

Authorized targets:
    hpc/
    studies/paper02_conditioning_audit/
    studies/paper03_augmentation_regimes/
    studies/paper04_selective_policies/

This tool does NOT:
- move or copy historical runtime files,
- rewrite imports,
- migrate Papers 2–4,
- relocate model plugins,
- modify protected historical evidence.

Each scaffold contains only README.md with an explicit authority boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path


M8_4_2_ACCEPTANCE_COMMIT = "954ad5f"

SCAFFOLDS = {
    "hpc/README.md": """# TrustForge HPC

**Status:** `SCAFFOLD_ONLY`

This directory is the native TrustForge destination for future execution-backend support.

Creating this directory does **not** migrate, replace, supersede, or reinterpret any historical Slurm script.

Current historical/global HPC compatibility surfaces remain authoritative in their existing locations, including:

- `slurm/`
- `run_gcs.slurm`
- `run_suite.slurm`
- `run_tuning.slurm`

Authority boundary:

- `SCAFFOLD_CREATION != MIGRATION`
- `DESTINATION_EXISTENCE != AUTHORITY_TRANSFER`
- Historical execution semantics remain unchanged.
- No historical Slurm file has been moved or copied here.
- No import, execution, or scheduling behavior is changed by this scaffold.

Authorized by M8.4.2 acceptance commit `954ad5f`.
""",
    "studies/paper02_conditioning_audit/README.md": """# Paper 2 — Conditioning Audit

**Status:** `SCAFFOLD_ONLY`

This directory is the future canonical TrustForge study destination for Paper 2.

It is **not** yet the authoritative Paper 2 study package.

The historical source remains authoritative at:

`papers/paper2_conditional_generation_done_right/`

Authority boundary:

- `SCAFFOLD_CREATION != MIGRATION`
- `DESTINATION_EXISTENCE != AUTHORITY_TRANSFER`
- No Paper 2 historical file has been moved or copied here.
- No scientific result, configuration, manifest, script, log, or artifact is reclassified by this scaffold.
- Paper 2 migration requires a dedicated evidence-mapping and compatibility-validation milestone.

Authorized by M8.4.2 acceptance commit `954ad5f`.
""",
    "studies/paper03_augmentation_regimes/README.md": """# Paper 3 — Augmentation Regimes

**Status:** `SCAFFOLD_ONLY`

This directory is the future canonical TrustForge study destination for Paper 3.

It is **not** yet the authoritative Paper 3 study package.

The historical source remains authoritative at:

`papers/paper3_when_does_synth_help/`

Authority boundary:

- `SCAFFOLD_CREATION != MIGRATION`
- `DESTINATION_EXISTENCE != AUTHORITY_TRANSFER`
- No Paper 3 historical file has been moved or copied here.
- No scientific result, configuration, script, log, artifact, augmentation regime, or minority-class intervention is reclassified by this scaffold.
- Paper 3 migration requires a dedicated evidence-mapping and compatibility-validation milestone.

Authorized by M8.4.2 acceptance commit `954ad5f`.
""",
    "studies/paper04_selective_policies/README.md": """# Paper 4 — Selective Policies

**Status:** `SCAFFOLD_ONLY`

This directory is the future canonical TrustForge study destination for Paper 4.

It is **not** yet the authoritative Paper 4 study package.

The historical source remains authoritative at:

`papers/paper4_selective_synth_policies/`

Authority boundary:

- `SCAFFOLD_CREATION != MIGRATION`
- `DESTINATION_EXISTENCE != AUTHORITY_TRANSFER`
- No Paper 4 historical file has been moved or copied here.
- No scientific result, configuration, script, log, artifact, keep-all policy, confidence rule, top-k rule, or class-repair rule is reclassified by this scaffold.
- Paper 4 migration requires a dedicated evidence-mapping and compatibility-validation milestone.

Authorized by M8.4.2 acceptance commit `954ad5f`.
""",
}

HISTORICAL_AUTHORITIES = [
    "slurm",
    "papers/paper2_conditional_generation_done_right",
    "papers/paper3_when_does_synth_help",
    "papers/paper4_selective_synth_policies",
]


def normalized(text: str) -> str:
    return text.rstrip() + "\n"


def check_historical_authorities(repo: Path) -> None:
    missing = [path for path in HISTORICAL_AUTHORITIES if not (repo / path).exists()]
    if missing:
        raise RuntimeError(
            "Historical authority precondition failed; missing: " + ", ".join(missing)
        )


def inspect_target(repo: Path, relative_file: str) -> str:
    path = repo / relative_file
    expected = normalized(SCAFFOLDS[relative_file])

    if not path.parent.exists():
        return "ABSENT"

    entries = sorted(p.name for p in path.parent.iterdir())
    if not entries:
        return "EMPTY_DIRECTORY"

    if entries != ["README.md"]:
        raise RuntimeError(
            f"Refusing scaffold operation: {path.parent} contains unexpected entries: "
            + ", ".join(entries)
        )

    if not path.exists():
        raise RuntimeError(f"Expected README missing in existing scaffold: {path}")

    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError(f"Existing scaffold README differs from governed content: {path}")

    return "MATERIALIZED"


def check(repo: Path) -> dict[str, str]:
    check_historical_authorities(repo)
    return {relative_file: inspect_target(repo, relative_file) for relative_file in SCAFFOLDS}


def materialize(repo: Path) -> list[Path]:
    states = check(repo)
    written: list[Path] = []

    for relative_file, state in states.items():
        path = repo / relative_file

        if state == "MATERIALIZED":
            continue

        if state not in {"ABSENT", "EMPTY_DIRECTORY"}:
            raise RuntimeError(f"Unexpected scaffold state for {relative_file}: {state}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalized(SCAFFOLDS[relative_file]), encoding="utf-8")
        written.append(path)

    final_states = check(repo)
    if set(final_states.values()) != {"MATERIALIZED"}:
        raise RuntimeError(f"Post-materialization verification failed: {final_states}")

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()

    states = check(repo)

    print(f"[m8.4.3] authority_commit={M8_4_2_ACCEPTANCE_COMMIT}")
    for relative_file, state in states.items():
        print(f"[m8.4.3] {relative_file}={state}")

    if args.check_only:
        print("[m8.4.3] check-only: no scaffold files written")
        return 0

    written = materialize(repo)
    for path in written:
        print(f"[m8.4.3] wrote {path}")

    if not written:
        print("[m8.4.3] all authorized scaffolds already materialized")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
