"""
Portable filesystem-root resolution for TrustForge.

Scientific identity must never depend on a machine-specific absolute path.

TrustForge therefore separates logical research identity from the concrete
filesystem locations used on an execution machine. Dataset and artifact roots
are supplied through environment variables. The repository root may also be
supplied explicitly, or it may be discovered by walking upward from a known
location.

Canonical environment variables
--------------------------------
TRUSTFORGE_DATA_ROOT
TRUSTFORGE_ARTIFACTS_ROOT
TRUSTFORGE_REPO_ROOT

Temporary compatibility aliases
--------------------------------
GCS_DATA_ROOT
GCS_ARTIFACTS_ROOT
GCS_REPO_ROOT

The TRUSTFORGE_* names always take precedence over legacy aliases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TRUSTFORGE_DATA_ROOT = "TRUSTFORGE_DATA_ROOT"
TRUSTFORGE_ARTIFACTS_ROOT = "TRUSTFORGE_ARTIFACTS_ROOT"
TRUSTFORGE_REPO_ROOT = "TRUSTFORGE_REPO_ROOT"

LEGACY_DATA_ROOT = "GCS_DATA_ROOT"
LEGACY_ARTIFACTS_ROOT = "GCS_ARTIFACTS_ROOT"
LEGACY_REPO_ROOT = "GCS_REPO_ROOT"


class StorageResolutionError(RuntimeError):
    """Raised when a required TrustForge filesystem root cannot be resolved."""


@dataclass(frozen=True)
class StorageRoots:
    """
    Concrete filesystem roots for one execution environment.

    These paths describe execution reality only. They are not scientific
    identifiers and must not be used as substitutes for dataset IDs,
    experiment IDs, study IDs, or artifact identities.
    """

    repo_root: Path
    data_root: Path | None
    artifacts_root: Path | None


def _normalize_path(value: str) -> Path:
    """
    Convert an environment-variable value into a normalized absolute Path.

    User-home markers and environment variables embedded inside the value are
    expanded. The path does not need to exist.
    """

    expanded = os.path.expandvars(os.path.expanduser(value))
    return Path(expanded).resolve(strict=False)


def _first_configured_environment_value(
    primary: str,
    aliases: Iterable[str] = (),
) -> tuple[str, str] | None:
    """
    Return the first configured non-empty environment variable.

    The canonical variable is always checked before compatibility aliases.
    """

    for name in (primary, *aliases):
        value = os.environ.get(name)

        if value is not None and value.strip():
            return name, value.strip()

    return None


def _resolve_environment_root(
    primary: str,
    aliases: Iterable[str] = (),
    *,
    required: bool,
) -> Path | None:
    """Resolve a root from canonical and compatibility environment variables."""

    configured = _first_configured_environment_value(primary, aliases)

    if configured is None:
        if required:
            alias_text = ", ".join(aliases)

            if alias_text:
                raise StorageResolutionError(
                    f"Unable to resolve required root. Set {primary}. "
                    f"Legacy compatibility variables also recognized: "
                    f"{alias_text}."
                )

            raise StorageResolutionError(
                f"Unable to resolve required root. Set {primary}."
            )

        return None

    _, value = configured
    return _normalize_path(value)


def resolve_data_root(*, required: bool = True) -> Path | None:
    """Resolve the TrustForge data root for the current machine."""

    return _resolve_environment_root(
        TRUSTFORGE_DATA_ROOT,
        (LEGACY_DATA_ROOT,),
        required=required,
    )


def resolve_artifacts_root(*, required: bool = True) -> Path | None:
    """Resolve the TrustForge artifact root for the current machine."""

    return _resolve_environment_root(
        TRUSTFORGE_ARTIFACTS_ROOT,
        (LEGACY_ARTIFACTS_ROOT,),
        required=required,
    )


def discover_repo_root(start: Path | str | None = None) -> Path | None:
    """
    Discover a Git repository root by walking upward.

    No Git subprocess is invoked. Discovery looks only for a `.git` entry.

    Parameters
    ----------
    start:
        File or directory from which discovery should begin. If omitted, the
        current working directory is used.
    """

    candidate = Path.cwd() if start is None else Path(start)
    candidate = candidate.expanduser().resolve(strict=False)

    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory

    return None


def resolve_repo_root(*, required: bool = True) -> Path | None:
    """
    Resolve the TrustForge repository root.

    Resolution order:

    1. TRUSTFORGE_REPO_ROOT
    2. GCS_REPO_ROOT compatibility alias
    3. repository discovery from the current working directory
    4. repository discovery from this installed source file

    Unlike data and artifact roots, repository discovery is portable and does
    not encode a machine-specific filesystem default.
    """

    configured = _first_configured_environment_value(
        TRUSTFORGE_REPO_ROOT,
        (LEGACY_REPO_ROOT,),
    )

    if configured is not None:
        _, value = configured
        return _normalize_path(value)

    discovered = discover_repo_root()

    if discovered is None:
        discovered = discover_repo_root(Path(__file__))

    if discovered is not None:
        return discovered

    if required:
        raise StorageResolutionError(
            "Unable to resolve the TrustForge repository root. "
            f"Set {TRUSTFORGE_REPO_ROOT} explicitly."
        )

    return None


def resolve_storage_roots(
    *,
    require_data: bool = True,
    require_artifacts: bool = True,
) -> StorageRoots:
    """
    Resolve the filesystem roots needed by a TrustForge execution.

    Repository resolution is always required because TrustForge execution and
    provenance must be anchored to a concrete repository checkout.
    """

    repo_root = resolve_repo_root(required=True)

    if repo_root is None:
        raise StorageResolutionError(
            "Repository root resolution unexpectedly returned no path."
        )

    return StorageRoots(
        repo_root=repo_root,
        data_root=resolve_data_root(required=require_data),
        artifacts_root=resolve_artifacts_root(required=require_artifacts),
    )
