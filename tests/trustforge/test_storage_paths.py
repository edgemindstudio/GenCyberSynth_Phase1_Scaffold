"""Tests for TrustForge portable storage-root resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trustforge.storage import (
    StorageResolutionError,
    discover_repo_root,
    resolve_artifacts_root,
    resolve_data_root,
    resolve_repo_root,
    resolve_storage_roots,
)


class StoragePathTests(unittest.TestCase):
    """Validate portable and deterministic root resolution."""

    def test_canonical_data_root_is_resolved(self) -> None:
        with patch.dict(
            os.environ,
            {"TRUSTFORGE_DATA_ROOT": "/tmp/trustforge-data"},
            clear=True,
        ):
            self.assertEqual(
                resolve_data_root(),
                Path("/tmp/trustforge-data").resolve(),
            )

    def test_canonical_artifacts_root_is_resolved(self) -> None:
        with patch.dict(
            os.environ,
            {"TRUSTFORGE_ARTIFACTS_ROOT": "/tmp/trustforge-artifacts"},
            clear=True,
        ):
            self.assertEqual(
                resolve_artifacts_root(),
                Path("/tmp/trustforge-artifacts").resolve(),
            )

    def test_canonical_variable_wins_over_legacy_alias(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRUSTFORGE_DATA_ROOT": "/tmp/trustforge-data",
                "GCS_DATA_ROOT": "/tmp/legacy-data",
            },
            clear=True,
        ):
            self.assertEqual(
                resolve_data_root(),
                Path("/tmp/trustforge-data").resolve(),
            )

    def test_legacy_data_root_is_supported(self) -> None:
        with patch.dict(
            os.environ,
            {"GCS_DATA_ROOT": "/tmp/legacy-data"},
            clear=True,
        ):
            self.assertEqual(
                resolve_data_root(),
                Path("/tmp/legacy-data").resolve(),
            )

    def test_legacy_artifacts_root_is_supported(self) -> None:
        with patch.dict(
            os.environ,
            {"GCS_ARTIFACTS_ROOT": "/tmp/legacy-artifacts"},
            clear=True,
        ):
            self.assertEqual(
                resolve_artifacts_root(),
                Path("/tmp/legacy-artifacts").resolve(),
            )

    def test_missing_required_data_root_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(StorageResolutionError):
                resolve_data_root()

    def test_missing_optional_data_root_returns_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_data_root(required=False))

    def test_missing_required_artifacts_root_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(StorageResolutionError):
                resolve_artifacts_root()

    def test_missing_optional_artifacts_root_returns_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_artifacts_root(required=False))

    def test_empty_environment_value_is_treated_as_unset(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRUSTFORGE_DATA_ROOT": "   ",
                "GCS_DATA_ROOT": "/tmp/legacy-data",
            },
            clear=True,
        ):
            self.assertEqual(
                resolve_data_root(),
                Path("/tmp/legacy-data").resolve(),
            )

    def test_repo_root_environment_override_wins(self) -> None:
        with patch.dict(
            os.environ,
            {"TRUSTFORGE_REPO_ROOT": "/tmp/trustforge-repo"},
            clear=True,
        ):
            self.assertEqual(
                resolve_repo_root(),
                Path("/tmp/trustforge-repo").resolve(),
            )

    def test_repo_root_legacy_alias_is_supported(self) -> None:
        with patch.dict(
            os.environ,
            {"GCS_REPO_ROOT": "/tmp/legacy-repo"},
            clear=True,
        ):
            self.assertEqual(
                resolve_repo_root(),
                Path("/tmp/legacy-repo").resolve(),
            )

    def test_repository_discovery_walks_upward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()

            nested = root / "a" / "b" / "c"
            nested.mkdir(parents=True)

            self.assertEqual(
                discover_repo_root(nested),
                root.resolve(),
            )

    def test_repository_discovery_returns_none_without_git_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            start = Path(tmp) / "a" / "b"
            start.mkdir(parents=True)

            self.assertIsNone(discover_repo_root(start))

    def test_storage_roots_are_resolved_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            data = Path(tmp) / "data"
            artifacts = Path(tmp) / "artifacts"

            env = {
                "TRUSTFORGE_REPO_ROOT": str(repo),
                "TRUSTFORGE_DATA_ROOT": str(data),
                "TRUSTFORGE_ARTIFACTS_ROOT": str(artifacts),
            }

            with patch.dict(os.environ, env, clear=True):
                roots = resolve_storage_roots()

            self.assertEqual(roots.repo_root, repo.resolve())
            self.assertEqual(roots.data_root, data.resolve())
            self.assertEqual(roots.artifacts_root, artifacts.resolve())

    def test_storage_roots_can_leave_optional_roots_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"

            with patch.dict(
                os.environ,
                {"TRUSTFORGE_REPO_ROOT": str(repo)},
                clear=True,
            ):
                roots = resolve_storage_roots(
                    require_data=False,
                    require_artifacts=False,
                )

            self.assertEqual(roots.repo_root, repo.resolve())
            self.assertIsNone(roots.data_root)
            self.assertIsNone(roots.artifacts_root)


if __name__ == "__main__":
    unittest.main()
