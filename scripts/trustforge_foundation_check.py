#!/usr/bin/env python3
"""
TrustForge foundation integration validation.

This script verifies that the architectural pieces introduced during the
TrustForge foundation migration operate together coherently.

It performs no scientific computation and writes no artifacts.

The validation chain is:

study contract
    ->
experiment contract
    ->
study/experiment linkage
    ->
portable dataset resolution
    ->
execution identity
    ->
canonical artifact-path construction
    ->
Git provenance
    ->
native manifest construction
    ->
manifest contract validation

Historical Paper 1 examples are used only as contract inputs. Historical
artifacts, manifests, configurations, and evidence are never modified.

Important distinction:

    dataset_id  = logical scientific dataset identity
    dataset_key = machine-storage directory identifier

Those values need not be identical.
"""

from __future__ import annotations

import argparse
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from trustforge.provenance import (
    build_execution_id,
    build_native_manifest,
    capture_git_state,
    validate_contract_file,
    validate_manifest_contract,
)
from trustforge.storage import (
    dataset_path,
    execution_artifact_root,
    resolve_repo_root,
)


DEFAULT_STUDY_EXAMPLE = (
    "schemas/examples/paper01/study.yaml"
)

DEFAULT_EXPERIMENT_EXAMPLE = (
    "schemas/examples/paper01/"
    "experiment_ustc_gan_seed42.yaml"
)

DEFAULT_STUDY_SCHEMA = (
    "schemas/study.schema.yaml"
)

DEFAULT_EXPERIMENT_SCHEMA = (
    "schemas/experiment.schema.yaml"
)

DEFAULT_MANIFEST_SCHEMA = (
    "schemas/manifest.schema.yaml"
)

DEFAULT_DATASET_KEY = (
    "USTC-TFC2016_malware_nhwc"
)


class FoundationValidationError(RuntimeError):
    """Raised when TrustForge foundation integration validation fails."""


def load_yaml_mapping(
    path: Path,
) -> dict[str, Any]:
    """Load one YAML document and require a mapping at the root."""

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            value = yaml.safe_load(
                handle
            )
    except OSError as exc:
        raise FoundationValidationError(
            f"Unable to read YAML file {path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise FoundationValidationError(
            f"Invalid YAML in {path}: {exc}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise FoundationValidationError(
            f"Expected YAML mapping in {path}, "
            f"found {type(value).__name__}."
        )

    return value


def validate_study_experiment_linkage(
    study: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> None:
    """Validate scientific linkage between study and experiment contracts."""

    study_id = study.get(
        "study_id"
    )

    experiment_study_id = experiment.get(
        "study_id"
    )

    if experiment_study_id != study_id:
        raise FoundationValidationError(
            "Experiment study_id does not match study: "
            f"{experiment_study_id!r} != {study_id!r}"
        )

    scope = study.get(
        "scope"
    )

    if not isinstance(
        scope,
        Mapping,
    ):
        raise FoundationValidationError(
            "Study scope is not a mapping."
        )

    dataset = experiment.get(
        "dataset"
    )

    if not isinstance(
        dataset,
        Mapping,
    ):
        raise FoundationValidationError(
            "Experiment dataset is not a mapping."
        )

    dataset_id = dataset.get(
        "dataset_id"
    )

    study_datasets = scope.get(
        "datasets"
    )

    if not isinstance(
        study_datasets,
        list,
    ):
        raise FoundationValidationError(
            "Study scope.datasets is not a list."
        )

    if dataset_id not in study_datasets:
        raise FoundationValidationError(
            "Experiment dataset_id is not declared "
            f"in study scope: {dataset_id!r}"
        )

    model = experiment.get(
        "model"
    )

    if not isinstance(
        model,
        Mapping,
    ):
        raise FoundationValidationError(
            "Experiment model is not a mapping."
        )

    model_id = model.get(
        "model_id"
    )

    study_models = scope.get(
        "models"
    )

    if not isinstance(
        study_models,
        list,
    ):
        raise FoundationValidationError(
            "Study scope.models is not a list."
        )

    if model_id not in study_models:
        raise FoundationValidationError(
            "Experiment model_id is not declared "
            f"in study scope: {model_id!r}"
        )


def validate_resolved_dataset(
    resolved_dataset_path: Path,
) -> None:
    """Require the resolved physical dataset directory to be usable."""

    if not resolved_dataset_path.exists():
        raise FoundationValidationError(
            "Resolved dataset does not exist: "
            f"{resolved_dataset_path}"
        )

    if not resolved_dataset_path.is_dir():
        raise FoundationValidationError(
            "Resolved dataset is not a directory: "
            f"{resolved_dataset_path}"
        )


def build_foundation_manifest(
    *,
    repo_root: Path,
    study: Mapping[str, Any],
    experiment: Mapping[str, Any],
    dataset_key: str,
    started_at: datetime,
    machine: str,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
]:
    """
    Construct an in-memory native manifest for foundation validation.

    No directory or manifest file is created.
    """

    if started_at.tzinfo is None:
        raise FoundationValidationError(
            "started_at must be timezone-aware."
        )

    study_id = study.get(
        "study_id"
    )

    experiment_id = experiment.get(
        "experiment_id"
    )

    dataset = experiment.get(
        "dataset"
    )

    if not isinstance(
        study_id,
        str,
    ):
        raise FoundationValidationError(
            "Study study_id is missing or invalid."
        )

    if not isinstance(
        experiment_id,
        str,
    ):
        raise FoundationValidationError(
            "Experiment experiment_id is missing or invalid."
        )

    if not isinstance(
        dataset,
        Mapping,
    ):
        raise FoundationValidationError(
            "Experiment dataset is missing or invalid."
        )

    dataset_id = dataset.get(
        "dataset_id"
    )

    if not isinstance(
        dataset_id,
        str,
    ):
        raise FoundationValidationError(
            "Experiment dataset_id is missing or invalid."
        )

    seed = experiment.get(
        "seed"
    )

    if isinstance(
        seed,
        bool,
    ) or not isinstance(
        seed,
        int,
    ):
        raise FoundationValidationError(
            "Experiment seed is missing or invalid."
        )

    lineage = study.get(
        "lineage"
    )

    if not isinstance(
        lineage,
        Mapping,
    ):
        raise FoundationValidationError(
            "Study lineage is missing or invalid."
        )

    repository = lineage.get(
        "repository"
    )

    if not isinstance(
        repository,
        str,
    ):
        raise FoundationValidationError(
            "Study lineage.repository is missing or invalid."
        )

    resolved_dataset_path = dataset_path(
        dataset_key
    )

    validate_resolved_dataset(
        resolved_dataset_path
    )

    execution_id = build_execution_id(
        started_at,
        machine,
    )

    artifact_root = execution_artifact_root(
        study_id,
        experiment_id,
        execution_id,
    )

    if artifact_root.exists():
        raise FoundationValidationError(
            "Foundation validation artifact path "
            "already exists; refusing to use it: "
            f"{artifact_root}"
        )

    git_state = capture_git_state(
        repo_root
    )

    resolved_config = {
        "study": dict(
            study
        ),
        "experiment": dict(
            experiment
        ),
        "dataset_binding": {
            "dataset_id": dataset_id,
            "dataset_key": dataset_key,
            "resolved_path": str(
                resolved_dataset_path
            ),
        },
    }

    manifest = build_native_manifest(
        study_id=study_id,
        experiment_id=experiment_id,
        execution_id=execution_id,
        repository=repository,
        repo_root=repo_root,
        resolved_config=resolved_config,
        dataset_id=dataset_id,
        dataset_path=resolved_dataset_path,
        artifact_root=artifact_root,
        status="planned",
        seed=seed,
        started_at=started_at,
        machine_role="foundation_validation",
        git_state=git_state,
    )

    return (
        manifest,
        resolved_dataset_path,
        artifact_root,
    )


def run_foundation_validation(
    *,
    repo_root: Path,
    study_example: Path,
    experiment_example: Path,
    study_schema: Path,
    experiment_schema: Path,
    manifest_schema: Path,
    dataset_key: str,
    started_at: datetime,
    machine: str,
) -> dict[str, Any]:
    """Run the complete read-only foundation integration validation."""

    study = load_yaml_mapping(
        study_example
    )

    experiment = load_yaml_mapping(
        experiment_example
    )

    validate_contract_file(
        study,
        study_schema,
    )

    validate_contract_file(
        experiment,
        experiment_schema,
    )

    validate_study_experiment_linkage(
        study,
        experiment,
    )

    (
        manifest,
        resolved_dataset_path,
        artifact_root,
    ) = build_foundation_manifest(
        repo_root=repo_root,
        study=study,
        experiment=experiment,
        dataset_key=dataset_key,
        started_at=started_at,
        machine=machine,
    )

    validate_manifest_contract(
        manifest,
        manifest_schema,
    )

    if artifact_root.exists():
        raise FoundationValidationError(
            "Foundation validation unexpectedly "
            "created an artifact directory: "
            f"{artifact_root}"
        )

    if manifest.get(
        "study_id"
    ) != study.get(
        "study_id"
    ):
        raise FoundationValidationError(
            "Native manifest study_id linkage failed."
        )

    if manifest.get(
        "experiment_id"
    ) != experiment.get(
        "experiment_id"
    ):
        raise FoundationValidationError(
            "Native manifest experiment_id linkage failed."
        )

    manifest_dataset = manifest.get(
        "dataset"
    )

    if not isinstance(
        manifest_dataset,
        Mapping,
    ):
        raise FoundationValidationError(
            "Native manifest dataset is invalid."
        )

    experiment_dataset = experiment.get(
        "dataset"
    )

    if not isinstance(
        experiment_dataset,
        Mapping,
    ):
        raise FoundationValidationError(
            "Experiment dataset is invalid."
        )

    if manifest_dataset.get(
        "dataset_id"
    ) != experiment_dataset.get(
        "dataset_id"
    ):
        raise FoundationValidationError(
            "Native manifest dataset_id linkage failed."
        )

    if manifest_dataset.get(
        "resolved_path"
    ) != str(
        resolved_dataset_path
    ):
        raise FoundationValidationError(
            "Native manifest physical dataset path linkage failed."
        )

    manifest_artifacts = manifest.get(
        "artifacts"
    )

    if not isinstance(
        manifest_artifacts,
        Mapping,
    ):
        raise FoundationValidationError(
            "Native manifest artifacts section is invalid."
        )

    if manifest_artifacts.get(
        "artifact_root"
    ) != str(
        artifact_root
    ):
        raise FoundationValidationError(
            "Native manifest artifact-root linkage failed."
        )

    return {
        "study_id": study[
            "study_id"
        ],
        "experiment_id": experiment[
            "experiment_id"
        ],
        "dataset_id": experiment_dataset[
            "dataset_id"
        ],
        "dataset_key": dataset_key,
        "dataset_path": str(
            resolved_dataset_path
        ),
        "execution_id": manifest[
            "execution_id"
        ],
        "artifact_root": str(
            artifact_root
        ),
        "git_commit": manifest[
            "provenance"
        ][
            "git_commit"
        ],
        "git_dirty": manifest[
            "provenance"
        ].get(
            "git_dirty"
        ),
        "manifest_status": manifest[
            "status"
        ],
    }


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only TrustForge foundation integration check."
        )
    )

    parser.add_argument(
        "--dataset-key",
        default=DEFAULT_DATASET_KEY,
        help=(
            "physical dataset directory identifier under "
            "TRUSTFORGE_DATA_ROOT"
        ),
    )

    parser.add_argument(
        "--machine",
        default=socket.gethostname(),
        help=(
            "machine component used only for the hypothetical "
            "execution identity"
        ),
    )

    arguments = parser.parse_args(
        argv
    )

    repo_root = resolve_repo_root()

    if repo_root is None:
        raise FoundationValidationError(
            "Unable to resolve TrustForge repository root."
        )

    started_at = datetime.now(
        timezone.utc
    )

    result = run_foundation_validation(
        repo_root=repo_root,
        study_example=(
            repo_root
            / DEFAULT_STUDY_EXAMPLE
        ),
        experiment_example=(
            repo_root
            / DEFAULT_EXPERIMENT_EXAMPLE
        ),
        study_schema=(
            repo_root
            / DEFAULT_STUDY_SCHEMA
        ),
        experiment_schema=(
            repo_root
            / DEFAULT_EXPERIMENT_SCHEMA
        ),
        manifest_schema=(
            repo_root
            / DEFAULT_MANIFEST_SCHEMA
        ),
        dataset_key=arguments.dataset_key,
        started_at=started_at,
        machine=arguments.machine,
    )

    print(
        "TrustForge Foundation Integration"
    )
    print(
        "================================="
    )
    print(
        "Study       :",
        result["study_id"],
    )
    print(
        "Experiment  :",
        result["experiment_id"],
    )
    print(
        "Dataset ID  :",
        result["dataset_id"],
    )
    print(
        "Dataset key :",
        result["dataset_key"],
    )
    print(
        "Dataset path:",
        result["dataset_path"],
    )
    print(
        "Execution   :",
        result["execution_id"],
    )
    print(
        "Artifact    :",
        result["artifact_root"],
    )
    print(
        "Git commit  :",
        result["git_commit"],
    )
    print(
        "Git dirty   :",
        result["git_dirty"],
    )
    print(
        "Status      :",
        result["manifest_status"],
    )
    print(
        "Artifact created: no"
    )
    print(
        "M5.3 FOUNDATION INTEGRATION: PASS"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except FoundationValidationError as exc:
        print(
            "M5.3 FOUNDATION INTEGRATION: FAIL",
            file=sys.stderr,
        )
        print(
            str(exc),
            file=sys.stderr,
        )
        raise SystemExit(
            1
        )
