"""Tests for canonical TrustForge storage layout construction."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trustforge.storage import (
    InvalidStorageIdentifier,
    dataset_path,
    execution_artifact_root,
    experiment_artifact_root,
    study_artifact_root,
    validate_storage_identifier,
)


class StorageLayoutTests(unittest.TestCase):
    """Validate safe and deterministic TrustForge path construction."""

    def test_valid_identifier_is_preserved(self) -> None:
        value = "paper01_benchmark"

        self.assertEqual(
            validate_storage_identifier(value),
            value,
        )

    def test_historical_paper1_identifiers_are_valid(self) -> None:
        identifiers = [
            "paper01_benchmark",
            "ustc_gan_b2000_seed42",
            "historical_talon_slurm_240398_task0",
        ]

        for identifier in identifiers:
            with self.subTest(identifier=identifier):
                self.assertEqual(
                    validate_storage_identifier(identifier),
                    identifier,
                )

    def test_dataset_key_with_hyphens_and_underscores_is_valid(self) -> None:
        dataset_key = "USTC-TFC2016_malware_nhwc"

        self.assertEqual(
            validate_storage_identifier(dataset_key),
            dataset_key,
        )

    def test_empty_identifier_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier("")

    def test_dot_identifier_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier(".")

    def test_parent_traversal_identifier_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier("..")

    def test_forward_slash_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier("paper01/run1")

    def test_backslash_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier(r"paper01\run1")

    def test_windows_invalid_character_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier("paper01:run1")

    def test_leading_whitespace_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier(" paper01")

    def test_trailing_whitespace_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier("paper01 ")

    def test_windows_reserved_name_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier("CON")

    def test_windows_reserved_name_with_extension_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier("NUL.txt")

    def test_control_character_is_rejected(self) -> None:
        with self.assertRaises(InvalidStorageIdentifier):
            validate_storage_identifier("paper01\nrun1")

    def test_dataset_path_uses_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            result = dataset_path(
                "USTC-TFC2016_malware_nhwc",
                data_root=root,
            )

            self.assertEqual(
                result,
                root.resolve() / "USTC-TFC2016_malware_nhwc",
            )

    def test_dataset_path_uses_environment_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "TRUSTFORGE_DATA_ROOT": tmp,
            }

            with patch.dict(os.environ, env, clear=True):
                result = dataset_path("dataset_a")

            self.assertEqual(
                result,
                Path(tmp).resolve() / "dataset_a",
            )

    def test_study_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = study_artifact_root(
                "paper01_benchmark",
                artifacts_root=tmp,
            )

            self.assertEqual(
                result,
                Path(tmp).resolve() / "paper01_benchmark",
            )

    def test_experiment_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = experiment_artifact_root(
                "paper01_benchmark",
                "ustc_gan_b2000_seed42",
                artifacts_root=tmp,
            )

            self.assertEqual(
                result,
                Path(tmp).resolve()
                / "paper01_benchmark"
                / "ustc_gan_b2000_seed42",
            )

    def test_execution_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = execution_artifact_root(
                "paper01_benchmark",
                "ustc_gan_b2000_seed42",
                "20260917T143000Z_talon_job123456",
                artifacts_root=tmp,
            )

            self.assertEqual(
                result,
                Path(tmp).resolve()
                / "paper01_benchmark"
                / "ustc_gan_b2000_seed42"
                / "20260917T143000Z_talon_job123456",
            )

    def test_layout_construction_does_not_create_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = execution_artifact_root(
                "paper01_benchmark",
                "experiment_a",
                "execution_a",
                artifacts_root=tmp,
            )

            self.assertFalse(result.exists())

    def test_invalid_study_id_cannot_escape_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InvalidStorageIdentifier):
                study_artifact_root(
                    "../outside",
                    artifacts_root=tmp,
                )

    def test_invalid_experiment_id_cannot_escape_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InvalidStorageIdentifier):
                experiment_artifact_root(
                    "paper01_benchmark",
                    "../../outside",
                    artifacts_root=tmp,
                )

    def test_invalid_execution_id_cannot_escape_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InvalidStorageIdentifier):
                execution_artifact_root(
                    "paper01_benchmark",
                    "experiment_a",
                    "../outside",
                    artifacts_root=tmp,
                )


if __name__ == "__main__":
    unittest.main()
