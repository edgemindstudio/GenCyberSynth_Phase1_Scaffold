"""
Read-only Git provenance capture for TrustForge.

The functions in this module inspect repository state only. They never commit,
checkout, reset, clean, stage, modify, or otherwise mutate a repository.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitProvenanceError(RuntimeError):
    """Raised when required Git provenance cannot be captured."""


@dataclass(frozen=True)
class GitState:
    """
    Git state associated with one TrustForge execution.

    Attributes
    ----------
    repo_root:
        Concrete repository checkout used for the execution.

    commit:
        Full Git commit SHA.

    branch:
        Current branch name, or None when HEAD is detached.

    dirty:
        True when tracked or untracked working-tree changes are present.
    """

    repo_root: Path
    commit: str
    branch: str | None
    dirty: bool


def _run_git(
    repo_root: Path,
    *arguments: str,
    allow_returncodes: tuple[int, ...] = (0,),
) -> tuple[int, str]:
    """Execute one read-only Git command and return code plus stdout."""

    command = [
        "git",
        "-C",
        str(repo_root),
        *arguments,
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GitProvenanceError(
            "Git executable was not found."
        ) from exc

    if result.returncode not in allow_returncodes:
        detail = result.stderr.strip() or result.stdout.strip()

        raise GitProvenanceError(
            f"Git command failed: {' '.join(command)}"
            + (f": {detail}" if detail else "")
        )

    return result.returncode, result.stdout.strip()


def capture_git_state(repo_root: Path | str) -> GitState:
    """
    Capture full commit, branch, and dirty-state provenance.

    The supplied path may point anywhere inside the repository. The returned
    repo_root is Git's actual top-level working-tree path.
    """

    candidate = Path(repo_root).expanduser().resolve(strict=False)

    if not candidate.exists():
        raise GitProvenanceError(
            f"Repository path does not exist: {candidate}"
        )

    _, top_level_text = _run_git(
        candidate,
        "rev-parse",
        "--show-toplevel",
    )

    top_level = Path(top_level_text).resolve(strict=False)

    _, commit = _run_git(
        top_level,
        "rev-parse",
        "HEAD",
    )

    branch_returncode, branch_text = _run_git(
        top_level,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
        allow_returncodes=(0, 1),
    )

    branch = branch_text if branch_returncode == 0 else None

    _, status = _run_git(
        top_level,
        "status",
        "--porcelain",
        "--untracked-files=normal",
    )

    return GitState(
        repo_root=top_level,
        commit=commit,
        branch=branch,
        dirty=bool(status),
    )
