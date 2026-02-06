#!/usr/bin/env python3
"""
repo_tree.py — Print a complete repository tree from the *current directory* downward.

Why this script exists
----------------------
When you want to share your repository structure in a new chat (or with collaborators),
a plain `ls` isn't enough. This script prints a deterministic, readable tree of
all files and folders under the directory where it is executed — including hidden
entries (e.g., .gitignore, .env.example, .github/).

Key guarantees
--------------
1) Scope: Only lists the directory you run it in (and everything below it).
   It will not "walk up" to parent directories.

2) Hidden files: Included by default.

3) Safety & edge cases:
   - Handles symlinks (does not follow them by default to prevent cycles).
   - Detects and reports broken symlinks.
   - Handles permission errors:
        * If any path cannot be accessed, the script will stop.
        * If you are not running with admin privileges, it will recommend rerunning with sudo.
   - Uses stable ordering (directories first, then files, both alphabetical).

Usage
-----
  python3 repo_tree.py
  python3 repo_tree.py --output repo_tree.txt
  python3 repo_tree.py --max-depth 6
  python3 repo_tree.py --follow-symlinks         # Not recommended unless you know your repo has no cycles
  python3 repo_tree.py --include-sizes           # Adds human-readable sizes for files
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Options:
    max_depth: Optional[int]
    follow_symlinks: bool
    include_sizes: bool
    output: Optional[str]


def is_admin() -> bool:
    """
    Return True if running with elevated privileges.

    - On Unix-like systems: checks effective user id (root == 0)
    - On Windows: best-effort check; we treat it as non-admin if unknown.
    """
    if os.name == "posix":
        return os.geteuid() == 0
    # Windows: avoid importing ctypes unless needed; return False by default
    return False


def human_size(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.1f}{u}" if u != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{num_bytes}B"


def safe_stat(path: str, follow_symlinks: bool) -> Optional[os.stat_result]:
    """
    Safely stat a path.
    Returns None if it cannot be stat'ed (e.g., broken symlink).
    """
    try:
        return os.stat(path, follow_symlinks=follow_symlinks)
    except FileNotFoundError:
        return None
    except OSError:
        return None


def list_dir_entries(path: str) -> List[os.DirEntry]:
    """
    List directory entries with robust error handling.
    Uses os.scandir for performance and to reduce repeated stat calls.
    """
    try:
        with os.scandir(path) as it:
            return list(it)
    except PermissionError as e:
        raise PermissionError(f"Permission denied while scanning: {path}") from e
    except FileNotFoundError as e:
        # Directory disappeared mid-walk; treat as fatal to keep output trustworthy.
        raise FileNotFoundError(f"Directory not found during scan (race): {path}") from e


def sort_entries(entries: Iterable[os.DirEntry]) -> List[os.DirEntry]:
    """
    Stable, predictable ordering:
    - directories first
    - then files
    - alphabetical within each group (case-insensitive)
    """
    def key_fn(de: os.DirEntry) -> Tuple[int, str]:
        try:
            is_dir = de.is_dir(follow_symlinks=False)
        except OSError:
            # If we can't determine, treat as file to avoid descending.
            is_dir = False
        return (0 if is_dir else 1, de.name.lower())

    return sorted(entries, key=key_fn)


def format_name(de: os.DirEntry, opts: Options) -> str:
    """
    Format a directory entry name with annotations:
    - symlink targets
    - broken symlink markers
    - optional file sizes
    """
    name = de.name

    try:
        is_link = de.is_symlink()
    except OSError:
        is_link = False

    # File size (only for regular files, not directories)
    size_suffix = ""
    if opts.include_sizes:
        try:
            if de.is_file(follow_symlinks=False):
                st = safe_stat(de.path, follow_symlinks=False)
                if st is not None:
                    size_suffix = f" ({human_size(st.st_size)})"
        except OSError:
            # Non-fatal: just skip size if inaccessible
            pass

    if is_link:
        try:
            target = os.readlink(de.path)
            # Broken link detection (best effort)
            st = safe_stat(de.path, follow_symlinks=True)
            if st is None:
                return f"{name} -> {target} [BROKEN]{size_suffix}"
            return f"{name} -> {target}{size_suffix}"
        except OSError:
            return f"{name} [SYMLINK]{size_suffix}"

    return f"{name}{size_suffix}"


def build_tree_lines(root: str, opts: Options) -> List[str]:
    """
    Build the tree output as a list of lines.
    Prints from root downward; does not traverse above root.
    """
    lines: List[str] = []

    # Print the root directory path (absolute) and a relative label
    abs_root = os.path.abspath(root)
    root_label = os.path.basename(abs_root) or abs_root
    lines.append(f"{root_label}/  (root: {abs_root})")

    # Walk recursively
    def walk(dir_path: str, prefix: str, depth: int) -> None:
        if opts.max_depth is not None and depth > opts.max_depth:
            return

        entries = sort_entries(list_dir_entries(dir_path))

        total = len(entries)
        for idx, de in enumerate(entries):
            is_last = (idx == total - 1)
            branch = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            # Determine if directory (without following symlinks unless enabled)
            try:
                is_dir = de.is_dir(follow_symlinks=opts.follow_symlinks)
            except OSError:
                is_dir = False

            display = format_name(de, opts)
            if is_dir:
                lines.append(f"{prefix}{branch}{display}/")
                # Avoid descending into symlinked directories unless explicitly requested
                if de.is_symlink() and not opts.follow_symlinks:
                    continue
                walk(de.path, next_prefix, depth + 1)
            else:
                lines.append(f"{prefix}{branch}{display}")

    walk(root, prefix="", depth=1)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a full tree of the current directory (including hidden entries)."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth to print (1 means only direct children). Default: unlimited.",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinks when deciding directory traversal. Default: off (safer).",
    )
    parser.add_argument(
        "--include-sizes",
        action="store_true",
        help="Include human-readable sizes for files. Default: off.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write output to a file instead of stdout.",
    )

    args = parser.parse_args()
    opts = Options(
        max_depth=args.max_depth,
        follow_symlinks=args.follow_symlinks,
        include_sizes=args.include_sizes,
        output=args.output,
    )

    root = os.getcwd()  # Only list the folder where the script is run
    admin = is_admin()

    try:
        lines = build_tree_lines(root, opts)
    except PermissionError as e:
        # If we hit a permission issue, provide a clear next step.
        msg = [
            "ERROR: Permission denied while listing the repository tree.",
            f"Reason: {e}",
            "",
            "Recommendation:",
            "  Re-run with elevated privileges to include restricted paths:",
            "    sudo python3 repo_tree.py",
            "",
            "If you do NOT want to use admin privileges, remove/relocate the restricted files,",
            "or ensure the current user has read/execute permissions for those directories.",
        ]
        print("\n".join(msg), file=sys.stderr)
        return 2

    # Write output
    output_text = "\n".join(lines) + "\n"
    if opts.output:
        try:
            with open(opts.output, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Wrote repository tree to: {opts.output}")
        except PermissionError:
            print(
                f"ERROR: Cannot write to output file '{opts.output}' (permission denied).",
                file=sys.stderr,
            )
            return 3
    else:
        sys.stdout.write(output_text)

    # Optional note about admin privileges (informational only)
    # We don't *require* admin privileges unless a permission error occurs.
    if not admin:
        # This is intentionally non-fatal and non-noisy.
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
