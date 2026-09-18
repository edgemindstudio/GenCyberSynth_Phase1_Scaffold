"""Tests for read-only TrustForge Git provenance capture."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from trustforge.provenance import (
    GitProvenanceError,
    capture_git_state,
)


class GitProvenanceTests(unittest.TestCase):
    """Validate commit, branch, and dirty-state capture."""

    def _git(
        self,
        repo: Path,
        *arguments: str,
    ) -> str:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                *arguments,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return result.stdout.strip()

    def _create_repository(self, root: Path) -> Path:
        repo = root / "repo"
        repo.mkdir()

        self._git(repo, "init")

        self._git(
            repo,
            "config",
            "user.name",
            "TrustForge Test",
        )
        self._git(
            repo,
            "config",
            "user.email",
            "trustforge-test@example.invalid",
        )

        tracked = repo / "tracked.txt"
        tracked.write_text(
            "initial\n",
            encoding="utf-8",
        )

        self._git(repo, "add", "tracked.txt")
        self._git(repo, "commit", "-m", "initial")

        return repo

    def test_clean_repository_state_is_captured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repository(Path(tmp))

            expected_commit = self._git(
                repo,
                "rev-parse",
                "HEAD",
            )

            expected_branch = self._git(
                repo,
                "branch",
                "--show-current",
            )

            state = capture_git_state(repo)

            self.assertEqual(
                state.repo_root,
                repo.resolve(),
            )
            self.assertEqual(
                state.commit,
                expected_commit,
            )
            self.assertEqual(
                state.branch,
                expected_branch,
            )
            self.assertFalse(state.dirty)

    def test_modified_tracked_file_sets_dirty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repository(Path(tmp))

            (repo / "tracked.txt").write_text(
                "modified\n",
                encoding="utf-8",
            )

            state = capture_git_state(repo)

            self.assertTrue(state.dirty)

    def test_untracked_file_sets_dirty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repository(Path(tmp))

            (repo / "untracked.txt").write_text(
                "new\n",
                encoding="utf-8",
            )

            state = capture_git_state(repo)

            self.assertTrue(state.dirty)

    def test_detached_head_has_no_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repository(Path(tmp))

            commit = self._git(
                repo,
                "rev-parse",
                "HEAD",
            )

            self._git(
                repo,
                "checkout",
                "--detach",
                commit,
            )

            state = capture_git_state(repo)

            self.assertEqual(
                state.commit,
                commit,
            )
            self.assertIsNone(state.branch)

    def test_non_repository_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GitProvenanceError):
                capture_git_state(Path(tmp))


if __name__ == "__main__":
    unittest.main()
