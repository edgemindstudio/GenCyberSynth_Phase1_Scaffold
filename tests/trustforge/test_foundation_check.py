"""Integration tests for the TrustForge foundation check."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "trustforge_foundation_check.py"
)


def _load_foundation_module():
    spec = importlib.util.spec_from_file_location(
        "trustforge_foundation_check",
        SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load TrustForge foundation check."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


foundation = _load_foundation_module()


class TrustForgeFoundationCheckTests(
    unittest.TestCase
):
    """Validate the M5.3 integration boundary."""

    def test_load_yaml_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "record.yaml"
            )

            path.write_text(
                "name: example\n",
                encoding="utf-8",
            )

            value = foundation.load_yaml_mapping(
                path
            )

        self.assertEqual(
            value,
            {
                "name": "example",
            },
        )

    def test_load_yaml_mapping_rejects_list_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "record.yaml"
            )

            path.write_text(
                "- one\n- two\n",
                encoding="utf-8",
            )

            with self.assertRaises(
                foundation.FoundationValidationError
            ):
                foundation.load_yaml_mapping(
                    path
                )

    def test_study_experiment_linkage_passes_for_examples(self) -> None:
        study = foundation.load_yaml_mapping(
            REPO_ROOT
            / foundation.DEFAULT_STUDY_EXAMPLE
        )

        experiment = foundation.load_yaml_mapping(
            REPO_ROOT
            / foundation.DEFAULT_EXPERIMENT_EXAMPLE
        )

        foundation.validate_study_experiment_linkage(
            study,
            experiment,
        )

    def test_study_experiment_linkage_rejects_wrong_study(self) -> None:
        study = {
            "study_id": "study_one",
            "scope": {
                "datasets": [
                    "dataset_one",
                ],
                "models": [
                    "model_one",
                ],
            },
        }

        experiment = {
            "study_id": "study_two",
            "dataset": {
                "dataset_id": "dataset_one",
            },
            "model": {
                "model_id": "model_one",
            },
        }

        with self.assertRaises(
            foundation.FoundationValidationError
        ):
            foundation.validate_study_experiment_linkage(
                study,
                experiment,
            )

    def test_study_experiment_linkage_rejects_dataset_outside_scope(
        self,
    ) -> None:
        study = {
            "study_id": "study_one",
            "scope": {
                "datasets": [
                    "dataset_one",
                ],
                "models": [
                    "model_one",
                ],
            },
        }

        experiment = {
            "study_id": "study_one",
            "dataset": {
                "dataset_id": "dataset_two",
            },
            "model": {
                "model_id": "model_one",
            },
        }

        with self.assertRaises(
            foundation.FoundationValidationError
        ):
            foundation.validate_study_experiment_linkage(
                study,
                experiment,
            )

    def test_study_experiment_linkage_rejects_model_outside_scope(
        self,
    ) -> None:
        study = {
            "study_id": "study_one",
            "scope": {
                "datasets": [
                    "dataset_one",
                ],
                "models": [
                    "model_one",
                ],
            },
        }

        experiment = {
            "study_id": "study_one",
            "dataset": {
                "dataset_id": "dataset_one",
            },
            "model": {
                "model_id": "model_two",
            },
        }

        with self.assertRaises(
            foundation.FoundationValidationError
        ):
            foundation.validate_study_experiment_linkage(
                study,
                experiment,
            )

    def test_resolved_dataset_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = (
                Path(tmp)
                / "missing"
            )

            with self.assertRaises(
                foundation.FoundationValidationError
            ):
                foundation.validate_resolved_dataset(
                    missing
                )

    def test_resolved_dataset_directory_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = (
                Path(tmp)
                / "dataset"
            )

            dataset.mkdir()

            foundation.validate_resolved_dataset(
                dataset
            )

    def test_example_study_validates_against_schema(self) -> None:
        study = foundation.load_yaml_mapping(
            REPO_ROOT
            / foundation.DEFAULT_STUDY_EXAMPLE
        )

        foundation.validate_contract_file(
            study,
            REPO_ROOT
            / foundation.DEFAULT_STUDY_SCHEMA,
        )

    def test_example_experiment_validates_against_schema(self) -> None:
        experiment = foundation.load_yaml_mapping(
            REPO_ROOT
            / foundation.DEFAULT_EXPERIMENT_EXAMPLE
        )

        foundation.validate_contract_file(
            experiment,
            REPO_ROOT
            / foundation.DEFAULT_EXPERIMENT_SCHEMA,
        )

    def test_build_foundation_manifest_does_not_create_artifact_root(
        self,
    ) -> None:
        study = foundation.load_yaml_mapping(
            REPO_ROOT
            / foundation.DEFAULT_STUDY_EXAMPLE
        )

        experiment = foundation.load_yaml_mapping(
            REPO_ROOT
            / foundation.DEFAULT_EXPERIMENT_EXAMPLE
        )

        started_at = datetime(
            2026,
            9,
            18,
            12,
            0,
            0,
            tzinfo=timezone.utc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            data_root = (
                root
                / "data"
            )

            artifacts_root = (
                root
                / "artifacts"
            )

            dataset = (
                data_root
                / foundation.DEFAULT_DATASET_KEY
            )

            dataset.mkdir(
                parents=True
            )

            artifacts_root.mkdir()

            environment = {
                "TRUSTFORGE_DATA_ROOT": str(
                    data_root
                ),
                "TRUSTFORGE_ARTIFACTS_ROOT": str(
                    artifacts_root
                ),
            }

            fake_git_state = foundation.capture_git_state(
                REPO_ROOT
            )

            with mock.patch.dict(
                os.environ,
                environment,
                clear=False,
            ):
                with mock.patch.object(
                    foundation,
                    "capture_git_state",
                    return_value=fake_git_state,
                ):
                    (
                        manifest,
                        resolved_dataset,
                        artifact_root,
                    ) = foundation.build_foundation_manifest(
                        repo_root=REPO_ROOT,
                        study=study,
                        experiment=experiment,
                        dataset_key=foundation.DEFAULT_DATASET_KEY,
                        started_at=started_at,
                        machine="testmachine",
                    )

            self.assertTrue(
                resolved_dataset.exists()
            )

            self.assertFalse(
                artifact_root.exists()
            )

            self.assertEqual(
                manifest["status"],
                "planned",
            )

    def test_native_foundation_manifest_validates_against_schema(
        self,
    ) -> None:
        study = foundation.load_yaml_mapping(
            REPO_ROOT
            / foundation.DEFAULT_STUDY_EXAMPLE
        )

        experiment = foundation.load_yaml_mapping(
            REPO_ROOT
            / foundation.DEFAULT_EXPERIMENT_EXAMPLE
        )

        started_at = datetime(
            2026,
            9,
            18,
            12,
            30,
            0,
            tzinfo=timezone.utc,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            data_root = (
                root
                / "data"
            )

            artifacts_root = (
                root
                / "artifacts"
            )

            dataset = (
                data_root
                / foundation.DEFAULT_DATASET_KEY
            )

            dataset.mkdir(
                parents=True
            )

            artifacts_root.mkdir()

            environment = {
                "TRUSTFORGE_DATA_ROOT": str(
                    data_root
                ),
                "TRUSTFORGE_ARTIFACTS_ROOT": str(
                    artifacts_root
                ),
            }

            fake_git_state = foundation.capture_git_state(
                REPO_ROOT
            )

            with mock.patch.dict(
                os.environ,
                environment,
                clear=False,
            ):
                with mock.patch.object(
                    foundation,
                    "capture_git_state",
                    return_value=fake_git_state,
                ):
                    (
                        manifest,
                        _,
                        artifact_root,
                    ) = foundation.build_foundation_manifest(
                        repo_root=REPO_ROOT,
                        study=study,
                        experiment=experiment,
                        dataset_key=foundation.DEFAULT_DATASET_KEY,
                        started_at=started_at,
                        machine="testmachine",
                    )

            foundation.validate_manifest_contract(
                manifest,
                REPO_ROOT
                / foundation.DEFAULT_MANIFEST_SCHEMA,
            )

            self.assertFalse(
                artifact_root.exists()
            )

    def test_logical_dataset_id_is_independent_of_physical_key(
        self,
    ) -> None:
        experiment = foundation.load_yaml_mapping(
            REPO_ROOT
            / foundation.DEFAULT_EXPERIMENT_EXAMPLE
        )

        logical_id = experiment[
            "dataset"
        ][
            "dataset_id"
        ]

        physical_key = (
            foundation.DEFAULT_DATASET_KEY
        )

        self.assertEqual(
            logical_id,
            "ustc_tfc2016_malware_nhwc",
        )

        self.assertEqual(
            physical_key,
            "USTC-TFC2016_malware_nhwc",
        )

        self.assertNotEqual(
            logical_id,
            physical_key,
        )


if __name__ == "__main__":
    unittest.main()
