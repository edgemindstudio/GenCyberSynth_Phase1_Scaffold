#!/usr/bin/env python3
"""
TrustForge engineering-readiness doctor.

This script is intentionally standalone and standard-library-only.

It must be able to start even when the active Python interpreter is too old
to import the TrustForge package. That allows it to diagnose an unsupported
environment rather than failing before it can explain the problem.

The doctor is read-only. It does not:

- install packages;
- create storage roots;
- create dataset directories;
- create probe files;
- modify shell configuration;
- activate Conda environments;
- alter Git state;
- submit scheduler jobs;
- initialize GPUs;
- write artifacts.

Core readiness and optional compute capabilities are reported separately.

Dataset readiness is opt-in through ``--dataset`` because TrustForge is a
multi-study research system and no single dataset is universally required.
"""

from __future__ import print_function

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


MINIMUM_PYTHON = (3, 10)

CANONICAL_DATA_ROOT = "TRUSTFORGE_DATA_ROOT"
CANONICAL_ARTIFACTS_ROOT = "TRUSTFORGE_ARTIFACTS_ROOT"
CANONICAL_REPO_ROOT = "TRUSTFORGE_REPO_ROOT"

LEGACY_DATA_ROOT = "GCS_DATA_ROOT"
LEGACY_ARTIFACTS_ROOT = "GCS_ARTIFACTS_ROOT"
LEGACY_REPO_ROOT = "GCS_REPO_ROOT"

CORE_SCHEMA_FILES = (
    "study.schema.yaml",
    "experiment.schema.yaml",
    "manifest.schema.yaml",
)


@dataclass(frozen=True)
class CheckResult:
    """One doctor check result."""

    level: str
    name: str
    detail: str


def _run_command(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
) -> Tuple[int, str, str]:
    """Run a command without mutating repository or environment state."""

    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        return 127, "", str(exc)

    return (
        completed.returncode,
        completed.stdout.strip(),
        completed.stderr.strip(),
    )


def discover_repo_root(start: Path) -> Optional[Path]:
    """
    Find the nearest parent containing a .git entry.

    No Git subprocess is required for discovery.
    """

    candidate = start.expanduser().resolve()

    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate,) + tuple(candidate.parents):
        if (current / ".git").exists():
            return current

    return None


def _environment_value(
    canonical: str,
    legacy: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve one TrustForge environment variable.

    Canonical names take precedence over legacy aliases.
    """

    canonical_value = os.environ.get(canonical)

    if canonical_value and canonical_value.strip():
        return canonical_value.strip(), canonical

    legacy_value = os.environ.get(legacy)

    if legacy_value and legacy_value.strip():
        return legacy_value.strip(), legacy

    return None, None


def _resolve_environment_path(
    canonical: str,
    legacy: str,
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve one configured TrustForge path from canonical or legacy env."""

    value, source = _environment_value(
        canonical,
        legacy,
    )

    if value is None:
        return None, None

    path = Path(
        os.path.expandvars(
            os.path.expanduser(value)
        )
    ).resolve()

    return path, source


def _format_bytes(value: int) -> str:
    """Format a byte count using binary units."""

    amount = float(value)

    units = (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
        "PiB",
    )

    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            if unit == "B":
                return "%d %s" % (
                    int(amount),
                    unit,
                )

            return "%.1f %s" % (
                amount,
                unit,
            )

        amount /= 1024.0

    return "%d B" % value


def _validate_dataset_identifier(
    dataset_id: str,
) -> Optional[str]:
    """
    Validate an optional dataset identifier for safe root-relative lookup.

    This standalone doctor intentionally uses only the minimal portability
    rules required to prevent path traversal. Scientific naming policy belongs
    to TrustForge contracts rather than the doctor.
    """

    if not isinstance(dataset_id, str):
        return "dataset identifier must be a string"

    if not dataset_id:
        return "dataset identifier must not be empty"

    if dataset_id != dataset_id.strip():
        return (
            "dataset identifier must not contain leading "
            "or trailing whitespace"
        )

    if dataset_id in (".", ".."):
        return "dataset identifier must not be '.' or '..'"

    if "/" in dataset_id or "\\" in dataset_id:
        return "dataset identifier must not contain path separators"

    if any(ord(character) < 32 for character in dataset_id):
        return "dataset identifier must not contain control characters"

    return None


def check_python_version() -> CheckResult:
    """Check whether the active interpreter satisfies TrustForge minimum."""

    current = sys.version_info[:3]
    minimum_text = ".".join(
        str(part)
        for part in MINIMUM_PYTHON
    )
    current_text = ".".join(
        str(part)
        for part in current
    )

    if current >= MINIMUM_PYTHON:
        return CheckResult(
            "PASS",
            "Python version",
            "%s >= %s (%s)"
            % (
                current_text,
                minimum_text,
                sys.executable,
            ),
        )

    return CheckResult(
        "FAIL",
        "Python version",
        "%s < %s (%s)"
        % (
            current_text,
            minimum_text,
            sys.executable,
        ),
    )


def check_git_executable() -> CheckResult:
    """Check Git availability."""

    executable = shutil.which("git")

    if executable is None:
        return CheckResult(
            "FAIL",
            "Git executable",
            "git was not found on PATH",
        )

    code, stdout, stderr = _run_command(
        [executable, "--version"]
    )

    if code != 0:
        return CheckResult(
            "FAIL",
            "Git executable",
            stderr or stdout or "git --version failed",
        )

    return CheckResult(
        "PASS",
        "Git executable",
        "%s (%s)"
        % (
            stdout,
            executable,
        ),
    )


def check_repository(
    repo_root: Optional[Path],
) -> CheckResult:
    """Check repository discovery and Git work-tree validity."""

    if repo_root is None:
        return CheckResult(
            "FAIL",
            "Repository",
            "no .git repository found from current location",
        )

    git = shutil.which("git")

    if git is None:
        return CheckResult(
            "FAIL",
            "Repository",
            "repository found at %s, but Git is unavailable"
            % repo_root,
        )

    code, stdout, stderr = _run_command(
        [
            git,
            "rev-parse",
            "--is-inside-work-tree",
        ],
        cwd=repo_root,
    )

    if code != 0 or stdout != "true":
        return CheckResult(
            "FAIL",
            "Repository",
            stderr or "Git work-tree validation failed",
        )

    if not os.access(
        str(repo_root),
        os.R_OK | os.X_OK,
    ):
        return CheckResult(
            "FAIL",
            "Repository",
            "%s is not readable/traversable"
            % repo_root,
        )

    return CheckResult(
        "PASS",
        "Repository",
        str(repo_root),
    )


def check_git_state(
    repo_root: Optional[Path],
) -> CheckResult:
    """
    Report current commit, branch, and dirty state.

    A dirty tree is a warning rather than a failure because development work
    may legitimately be uncommitted. Scientific execution policy can later
    impose stricter rules.
    """

    if repo_root is None:
        return CheckResult(
            "SKIP",
            "Git state",
            "repository unavailable",
        )

    git = shutil.which("git")

    if git is None:
        return CheckResult(
            "SKIP",
            "Git state",
            "Git unavailable",
        )

    commit_code, commit, commit_error = _run_command(
        [
            git,
            "rev-parse",
            "HEAD",
        ],
        cwd=repo_root,
    )

    if commit_code != 0:
        return CheckResult(
            "FAIL",
            "Git state",
            commit_error or "unable to resolve HEAD",
        )

    branch_code, branch, _ = _run_command(
        [
            git,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ],
        cwd=repo_root,
    )

    if branch_code != 0:
        branch = "DETACHED"

    status_code, status, status_error = _run_command(
        [
            git,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        cwd=repo_root,
    )

    if status_code != 0:
        return CheckResult(
            "FAIL",
            "Git state",
            status_error or "unable to inspect work tree",
        )

    dirty = bool(status)

    level = "WARN" if dirty else "PASS"

    return CheckResult(
        level,
        "Git state",
        "commit=%s branch=%s dirty=%s"
        % (
            commit,
            branch,
            str(dirty).lower(),
        ),
    )


def _check_storage_root(
    *,
    name: str,
    canonical: str,
    legacy: str,
    require_read: bool,
    require_write: bool,
    require_traverse: bool,
) -> CheckResult:
    """
    Check one configured TrustForge storage root.

    Permission checks use os.access and never create a probe file.
    """

    path, source = _resolve_environment_path(
        canonical,
        legacy,
    )

    if path is None:
        return CheckResult(
            "FAIL",
            name,
            "%s is not set"
            % canonical,
        )

    if not path.exists():
        return CheckResult(
            "FAIL",
            name,
            "%s -> %s does not exist"
            % (
                source,
                path,
            ),
        )

    if not path.is_dir():
        return CheckResult(
            "FAIL",
            name,
            "%s -> %s is not a directory"
            % (
                source,
                path,
            ),
        )

    missing = []

    if require_read and not os.access(
        str(path),
        os.R_OK,
    ):
        missing.append("read")

    if require_write and not os.access(
        str(path),
        os.W_OK,
    ):
        missing.append("write")

    if require_traverse and not os.access(
        str(path),
        os.X_OK,
    ):
        missing.append("traverse")

    if missing:
        return CheckResult(
            "FAIL",
            name,
            "%s -> %s missing access: %s"
            % (
                source,
                path,
                ", ".join(missing),
            ),
        )

    access = []

    if require_read:
        access.append("read")

    if require_write:
        access.append("write")

    if require_traverse:
        access.append("traverse")

    return CheckResult(
        "PASS",
        name,
        "%s -> %s access=%s"
        % (
            source,
            path,
            ",".join(access),
        ),
    )


def check_repo_environment(
    repo_root: Optional[Path],
) -> CheckResult:
    """
    Check TRUSTFORGE_REPO_ROOT when configured.

    Repository discovery remains valid even when the variable is unset.
    """

    value, source = _environment_value(
        CANONICAL_REPO_ROOT,
        LEGACY_REPO_ROOT,
    )

    if value is None:
        if repo_root is None:
            return CheckResult(
                "FAIL",
                "Repository environment",
                "%s is unset and repository discovery failed"
                % CANONICAL_REPO_ROOT,
            )

        return CheckResult(
            "INFO",
            "Repository environment",
            "%s unset; discovered %s"
            % (
                CANONICAL_REPO_ROOT,
                repo_root,
            ),
        )

    configured = Path(
        os.path.expandvars(
            os.path.expanduser(value)
        )
    ).resolve()

    if not configured.exists():
        return CheckResult(
            "FAIL",
            "Repository environment",
            "%s -> %s does not exist"
            % (
                source,
                configured,
            ),
        )

    if not configured.is_dir():
        return CheckResult(
            "FAIL",
            "Repository environment",
            "%s -> %s is not a directory"
            % (
                source,
                configured,
            ),
        )

    if repo_root is not None and configured != repo_root:
        return CheckResult(
            "WARN",
            "Repository environment",
            "%s -> %s differs from current repository %s"
            % (
                source,
                configured,
                repo_root,
            ),
        )

    return CheckResult(
        "PASS",
        "Repository environment",
        "%s -> %s"
        % (
            source,
            configured,
        ),
    )


def check_schemas(
    repo_root: Optional[Path],
) -> CheckResult:
    """Check presence of the M3 TrustForge contract files."""

    if repo_root is None:
        return CheckResult(
            "SKIP",
            "Contract schemas",
            "repository unavailable",
        )

    schema_root = repo_root / "schemas"

    missing = [
        filename
        for filename in CORE_SCHEMA_FILES
        if not (schema_root / filename).is_file()
    ]

    if missing:
        return CheckResult(
            "FAIL",
            "Contract schemas",
            "missing: %s"
            % ", ".join(missing),
        )

    return CheckResult(
        "PASS",
        "Contract schemas",
        ", ".join(CORE_SCHEMA_FILES),
    )


def check_pyyaml() -> CheckResult:
    """Check whether PyYAML is importable without importing TrustForge."""

    if importlib.util.find_spec("yaml") is None:
        return CheckResult(
            "FAIL",
            "PyYAML",
            "yaml module is not importable",
        )

    return CheckResult(
        "PASS",
        "PyYAML",
        "yaml module available",
    )


def check_trustforge_import(
    repo_root: Optional[Path],
) -> CheckResult:
    """
    Check TrustForge package import in a subprocess.

    This is skipped on Python < 3.10 so the doctor itself can still explain
    the unsupported interpreter cleanly.
    """

    if sys.version_info[:2] < MINIMUM_PYTHON:
        return CheckResult(
            "SKIP",
            "TrustForge import",
            "requires Python >= 3.10",
        )

    if repo_root is None:
        return CheckResult(
            "SKIP",
            "TrustForge import",
            "repository unavailable",
        )

    src = repo_root / "src"

    if not src.is_dir():
        return CheckResult(
            "FAIL",
            "TrustForge import",
            "src directory not found: %s"
            % src,
        )

    environment = os.environ.copy()

    old_pythonpath = environment.get(
        "PYTHONPATH",
        "",
    )

    if old_pythonpath:
        environment["PYTHONPATH"] = (
            str(src)
            + os.pathsep
            + old_pythonpath
        )
    else:
        environment["PYTHONPATH"] = str(src)

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import trustforge; "
                    "print(trustforge.__version__)"
                ),
            ],
            cwd=str(repo_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        return CheckResult(
            "FAIL",
            "TrustForge import",
            str(exc),
        )

    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "TrustForge import failed"
        )

        return CheckResult(
            "FAIL",
            "TrustForge import",
            detail,
        )

    return CheckResult(
        "PASS",
        "TrustForge import",
        "version=%s"
        % completed.stdout.strip(),
    )


def check_filesystem_capacity(
    *,
    name: str,
    canonical: str,
    legacy: str,
) -> CheckResult:
    """
    Report capacity for a configured filesystem.

    Capacity is informational because appropriate free-space requirements vary
    by experiment.
    """

    path, source = _resolve_environment_path(
        canonical,
        legacy,
    )

    if path is None:
        return CheckResult(
            "SKIP",
            name,
            "%s is not set"
            % canonical,
        )

    if not path.exists():
        return CheckResult(
            "SKIP",
            name,
            "%s -> %s does not exist"
            % (
                source,
                path,
            ),
        )

    try:
        usage = shutil.disk_usage(
            str(path)
        )
    except OSError as exc:
        return CheckResult(
            "INFO",
            name,
            "unable to inspect %s: %s"
            % (
                path,
                exc,
            ),
        )

    if usage.total:
        used_percent = (
            float(usage.used)
            / float(usage.total)
            * 100.0
        )
    else:
        used_percent = 0.0

    return CheckResult(
        "INFO",
        name,
        "%s free=%s total=%s used=%.1f%%"
        % (
            path,
            _format_bytes(usage.free),
            _format_bytes(usage.total),
            used_percent,
        ),
    )


def check_dataset(
    dataset_id: str,
) -> CheckResult:
    """
    Check one explicitly requested dataset under TRUSTFORGE_DATA_ROOT.

    The check is intentionally generic and does not impose dataset-specific
    file naming or scientific semantics.
    """

    problem = _validate_dataset_identifier(
        dataset_id
    )

    if problem is not None:
        return CheckResult(
            "FAIL",
            "Dataset %s" % dataset_id,
            problem,
        )

    data_root, source = _resolve_environment_path(
        CANONICAL_DATA_ROOT,
        LEGACY_DATA_ROOT,
    )

    if data_root is None:
        return CheckResult(
            "FAIL",
            "Dataset %s" % dataset_id,
            "%s is not set"
            % CANONICAL_DATA_ROOT,
        )

    if not data_root.is_dir():
        return CheckResult(
            "FAIL",
            "Dataset %s" % dataset_id,
            "%s -> %s is unavailable"
            % (
                source,
                data_root,
            ),
        )

    dataset_path = (
        data_root
        / dataset_id
    ).resolve()

    try:
        dataset_path.relative_to(
            data_root.resolve()
        )
    except ValueError:
        return CheckResult(
            "FAIL",
            "Dataset %s" % dataset_id,
            "resolved path escapes data root",
        )

    if not dataset_path.exists():
        return CheckResult(
            "FAIL",
            "Dataset %s" % dataset_id,
            "not found at %s"
            % dataset_path,
        )

    if not dataset_path.is_dir():
        return CheckResult(
            "FAIL",
            "Dataset %s" % dataset_id,
            "%s is not a directory"
            % dataset_path,
        )

    missing = []

    if not os.access(
        str(dataset_path),
        os.R_OK,
    ):
        missing.append("read")

    if not os.access(
        str(dataset_path),
        os.X_OK,
    ):
        missing.append("traverse")

    if missing:
        return CheckResult(
            "FAIL",
            "Dataset %s" % dataset_id,
            "%s missing access: %s"
            % (
                dataset_path,
                ", ".join(missing),
            ),
        )

    return CheckResult(
        "PASS",
        "Dataset %s" % dataset_id,
        "%s access=read,traverse"
        % dataset_path,
    )


def check_tensorflow_capability() -> CheckResult:
    """
    Report TensorFlow availability without importing TensorFlow.

    Avoiding the import prevents CUDA initialization and expensive runtime
    startup during a lightweight doctor check.
    """

    try:
        from importlib.metadata import (
            PackageNotFoundError,
            version,
        )
    except ImportError:
        from importlib_metadata import (  # type: ignore
            PackageNotFoundError,
            version,
        )

    try:
        tensorflow_version = version(
            "tensorflow"
        )
    except PackageNotFoundError:
        return CheckResult(
            "INFO",
            "TensorFlow",
            "not installed in current interpreter",
        )

    return CheckResult(
        "INFO",
        "TensorFlow",
        "version=%s"
        % tensorflow_version,
    )


def check_slurm_capability() -> CheckResult:
    """Report whether Slurm submission tooling is visible."""

    sbatch = shutil.which("sbatch")

    if sbatch is None:
        return CheckResult(
            "INFO",
            "Slurm",
            "sbatch not found on PATH",
        )

    return CheckResult(
        "INFO",
        "Slurm",
        "sbatch=%s"
        % sbatch,
    )


def check_gpu_capability() -> CheckResult:
    """
    Report current-host NVIDIA visibility.

    Lack of a visible GPU is informational only. Login nodes and development
    machines may legitimately have no accelerator.
    """

    nvidia_smi = shutil.which(
        "nvidia-smi"
    )

    if nvidia_smi is None:
        return CheckResult(
            "INFO",
            "GPU",
            "nvidia-smi not found on current host",
        )

    code, stdout, stderr = _run_command(
        [
            nvidia_smi,
            "-L",
        ]
    )

    if code != 0:
        return CheckResult(
            "INFO",
            "GPU",
            stderr or "no NVIDIA GPU visible on current host",
        )

    if not stdout:
        return CheckResult(
            "INFO",
            "GPU",
            "no NVIDIA GPU reported on current host",
        )

    return CheckResult(
        "INFO",
        "GPU",
        stdout.replace(
            "\n",
            "; ",
        ),
    )


def collect_results(
    start: Path,
    *,
    datasets: Optional[Sequence[str]] = None,
) -> List[CheckResult]:
    """Run all doctor checks."""

    repo_root = discover_repo_root(
        start
    )

    results = [
        check_python_version(),
        check_git_executable(),
        check_repository(repo_root),
        check_git_state(repo_root),
        check_repo_environment(repo_root),
        _check_storage_root(
            name="Data root",
            canonical=CANONICAL_DATA_ROOT,
            legacy=LEGACY_DATA_ROOT,
            require_read=True,
            require_write=False,
            require_traverse=True,
        ),
        _check_storage_root(
            name="Artifacts root",
            canonical=CANONICAL_ARTIFACTS_ROOT,
            legacy=LEGACY_ARTIFACTS_ROOT,
            require_read=True,
            require_write=True,
            require_traverse=True,
        ),
        check_schemas(repo_root),
        check_pyyaml(),
        check_trustforge_import(repo_root),
        check_filesystem_capacity(
            name="Data filesystem",
            canonical=CANONICAL_DATA_ROOT,
            legacy=LEGACY_DATA_ROOT,
        ),
        check_filesystem_capacity(
            name="Artifacts filesystem",
            canonical=CANONICAL_ARTIFACTS_ROOT,
            legacy=LEGACY_ARTIFACTS_ROOT,
        ),
    ]

    for dataset_id in datasets or ():
        results.append(
            check_dataset(
                dataset_id
            )
        )

    results.extend(
        [
            check_tensorflow_capability(),
            check_slurm_capability(),
            check_gpu_capability(),
        ]
    )

    return results


def _print_results(
    results: Sequence[CheckResult],
) -> None:
    """Print human-readable doctor output."""

    print("TrustForge Doctor")
    print("=================")

    for result in results:
        print(
            "%-5s %-24s %s"
            % (
                result.level,
                result.name,
                result.detail,
            )
        )

    core_failures = sum(
        1
        for result in results
        if result.level == "FAIL"
    )

    warnings = sum(
        1
        for result in results
        if result.level == "WARN"
    )

    print("")
    print(
        "Summary: %d failure(s), %d warning(s)"
        % (
            core_failures,
            warnings,
        )
    )


def _json_results(
    results: Sequence[CheckResult],
) -> str:
    """Return machine-readable doctor output."""

    payload = {
        "checks": [
            {
                "level": result.level,
                "name": result.name,
                "detail": result.detail,
            }
            for result in results
        ],
        "failure_count": sum(
            1
            for result in results
            if result.level == "FAIL"
        ),
        "warning_count": sum(
            1
            for result in results
            if result.level == "WARN"
        ),
    }

    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run TrustForge engineering-readiness diagnostics."""

    parser = argparse.ArgumentParser(
        description=(
            "Read-only TrustForge engineering-readiness diagnostics."
        )
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
    )

    parser.add_argument(
        "--start",
        default=os.getcwd(),
        help=(
            "path from which repository discovery begins "
            "(default: current working directory)"
        ),
    )

    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "dataset identifier to verify under TRUSTFORGE_DATA_ROOT; "
            "may be supplied more than once"
        ),
    )

    arguments = parser.parse_args(
        argv
    )

    results = collect_results(
        Path(arguments.start),
        datasets=arguments.dataset,
    )

    if arguments.json:
        print(
            _json_results(
                results
            )
        )
    else:
        _print_results(
            results
        )

    has_failure = any(
        result.level == "FAIL"
        for result in results
    )

    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
