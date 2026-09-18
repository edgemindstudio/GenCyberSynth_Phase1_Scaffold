"""Tests for TrustForge-native execution manifest construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trustforge.provenance import (
    GitState,
    ManifestConstructionError,
    ManifestWriteError,
    build_native_manifest,
    sha256_mapping,
    write_manifest_json,
)


class ManifestConstructionTests(unittest.TestCase):
    """Validate native TrustForge manifest semantics."""

    def _git_state(
        self,
        repo_root: Path,
        *,
        dirty: bool = False,
    ) -> GitState:
        return GitState(
            repo_root=repo_root.resolve(),
            commit="a" * 40,
            branch="migration/trustforge-foundation",
            dirty=dirty,
        )

    def _build_manifest(
        self,
        root: Path,
        **overrides: object,
    ) -> dict:
        config = {
            "model": "gan",
            "seed": 42,
            "samples_per_class": 2000,
        }

        arguments = {
            "study_id": "paper01_benchmark",
            "experiment_id": "ustc_gan_b2000_seed42",
            "execution_id": "20260917T203000Z_talon_job123456",
            "repository": "edgemindstudio/trustforge",
            "repo_root": root,
            "resolved_config": config,
            "dataset_id": "USTC-TFC2016_malware_nhwc",
            "dataset_path": root / "data",
            "artifact_root": root / "artifacts",
            "status": "running",
            "config_source": root / "config.yaml",
            "seed": 42,
            "started_at": datetime(
                2026,
                9,
                17,
                20,
                30,
                0,
                tzinfo=timezone.utc,
            ),
            "machine_role": "talon_hpc",
            "scheduler": {
                "backend": "slurm",
                "job_id": "123456",
                "array_task_id": "0",
            },
            "git_state": self._git_state(root),
        }

        arguments.update(overrides)

        return build_native_manifest(**arguments)

    def test_required_contract_fields_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._build_manifest(root)

            for required in (
                "study_id",
                "experiment_id",
                "execution_id",
                "status",
                "provenance",
                "runtime",
                "dataset",
                "artifacts",
            ):
                self.assertIn(required, manifest)

    def test_native_config_digest_uses_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            config = {
                "seed": 42,
                "model": "gan",
            }

            manifest = self._build_manifest(
                root,
                resolved_config=config,
            )

            digest = manifest["provenance"]["config_digest"]

            self.assertEqual(
                digest["algorithm"],
                "sha256",
            )
            self.assertEqual(
                digest["scope"],
                "canonical_json_mapping",
            )
            self.assertEqual(
                digest["value"],
                sha256_mapping(config),
            )

    def test_git_dirty_state_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._build_manifest(
                root,
                git_state=self._git_state(
                    root,
                    dirty=True,
                ),
            )

            self.assertTrue(
                manifest["provenance"]["git_dirty"]
            )

    def test_git_commit_and_branch_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._build_manifest(root)

            provenance = manifest["provenance"]

            self.assertEqual(
                provenance["git_commit"],
                "a" * 40,
            )
            self.assertEqual(
                provenance["git_branch"],
                "migration/trustforge-foundation",
            )

    def test_detached_head_omits_git_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            state = GitState(
                repo_root=root.resolve(),
                commit="b" * 40,
                branch=None,
                dirty=False,
            )

            manifest = self._build_manifest(
                root,
                git_state=state,
            )

            self.assertNotIn(
                "git_branch",
                manifest["provenance"],
            )

    def test_dataset_and_artifact_paths_are_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._build_manifest(root)

            self.assertTrue(
                Path(
                    manifest["dataset"]["resolved_path"]
                ).is_absolute()
            )

            self.assertTrue(
                Path(
                    manifest["artifacts"]["artifact_root"]
                ).is_absolute()
            )

    def test_started_at_requires_timezone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(
                ManifestConstructionError
            ):
                self._build_manifest(
                    root,
                    started_at=datetime(
                        2026,
                        9,
                        17,
                        20,
                        30,
                        0,
                    ),
                )

    def test_invalid_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(
                ManifestConstructionError
            ):
                self._build_manifest(
                    root,
                    status="accepted",
                )

    def test_boolean_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with self.assertRaises(
                ManifestConstructionError
            ):
                self._build_manifest(
                    root,
                    seed=True,
                )

    def test_manifest_write_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._build_manifest(root)
            destination = root / "manifest.json"

            result = write_manifest_json(
                manifest,
                destination,
            )

            self.assertEqual(
                result,
                destination.resolve(),
            )

            loaded = json.loads(
                destination.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                loaded,
                manifest,
            )

    def test_manifest_writer_refuses_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._build_manifest(root)
            destination = root / "manifest.json"

            write_manifest_json(
                manifest,
                destination,
            )

            with self.assertRaises(
                ManifestWriteError
            ):
                write_manifest_json(
                    manifest,
                    destination,
                )

    def test_manifest_writer_can_overwrite_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            first = self._build_manifest(root)

            second = self._build_manifest(
                root,
                status="completed",
            )

            destination = root / "manifest.json"

            write_manifest_json(
                first,
                destination,
            )

            write_manifest_json(
                second,
                destination,
                overwrite=True,
            )

            loaded = json.loads(
                destination.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                loaded["status"],
                "completed",
            )

    def test_manifest_writer_does_not_create_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._build_manifest(root)

            destination = (
                root
                / "missing"
                / "manifest.json"
            )

            with self.assertRaises(
                ManifestWriteError
            ):
                write_manifest_json(
                    manifest,
                    destination,
                )


if __name__ == "__main__":
    unittest.main()

