"""
Portable TrustForge storage resolution and canonical layout construction.

Machine-specific filesystem locations describe execution reality only.
They must never define scientific identity.
"""

from trustforge.storage.layout import (
    InvalidStorageIdentifier,
    dataset_path,
    execution_artifact_root,
    experiment_artifact_root,
    study_artifact_root,
    validate_storage_identifier,
)
from trustforge.storage.paths import (
    LEGACY_ARTIFACTS_ROOT,
    LEGACY_DATA_ROOT,
    LEGACY_REPO_ROOT,
    TRUSTFORGE_ARTIFACTS_ROOT,
    TRUSTFORGE_DATA_ROOT,
    TRUSTFORGE_REPO_ROOT,
    StorageResolutionError,
    StorageRoots,
    discover_repo_root,
    resolve_artifacts_root,
    resolve_data_root,
    resolve_repo_root,
    resolve_storage_roots,
)

__all__ = [
    "LEGACY_ARTIFACTS_ROOT",
    "LEGACY_DATA_ROOT",
    "LEGACY_REPO_ROOT",
    "TRUSTFORGE_ARTIFACTS_ROOT",
    "TRUSTFORGE_DATA_ROOT",
    "TRUSTFORGE_REPO_ROOT",
    "InvalidStorageIdentifier",
    "StorageResolutionError",
    "StorageRoots",
    "dataset_path",
    "discover_repo_root",
    "execution_artifact_root",
    "experiment_artifact_root",
    "resolve_artifacts_root",
    "resolve_data_root",
    "resolve_repo_root",
    "resolve_storage_roots",
    "study_artifact_root",
    "validate_storage_identifier",
]
