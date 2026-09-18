"""
Portable dataset and artifact storage resolution.

Machine-specific filesystem locations describe execution reality only.
They must never define scientific identity.
"""

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
    "StorageResolutionError",
    "StorageRoots",
    "discover_repo_root",
    "resolve_artifacts_root",
    "resolve_data_root",
    "resolve_repo_root",
    "resolve_storage_roots",
]
