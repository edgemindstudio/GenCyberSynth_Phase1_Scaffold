"""Tests for the standalone TrustForge doctor."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]

DOCTOR_PATH = (
    REPO_ROOT
    / "scripts"
    / "trustforge_doctor.py"
)


def _load_doctor_module():
    spec = importlib.util.spec_from_file_location(
        "trustforge_doctor",
        DOCTOR_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load TrustForge doctor."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


doctor = _load_doctor_module()


class TrustForgeDoctorTests(unittest.TestCase):
    """Validate read-only doctor behavior."""

    def test_repository_discovery_finds_parent_git_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            nested = repo / "a" / "b"

            nested.mkdir(parents=True)
            (repo / ".git").mkdir()

            discovered = doctor.discover_repo_root(
                nested
            )

            self.assertEqual(
                discovered,
                repo.resolve(),
            )

    def test_repository_discovery_returns_none_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            discovered = doctor.discover_repo_root(
                Path(tmp)
            )

            self.assertIsNone(
                discovered
            )

    def test_canonical_environment_wins_over_legacy(self) -> None:
        environment = {
            "TRUSTFORGE_DATA_ROOT": "/canonical",
            "GCS_DATA_ROOT": "/legacy",
        }

        with mock.patch.dict(
            os.environ,
            environment,
            clear=True,
        ):
            value, source = doctor._environment_value(
                "TRUSTFORGE_DATA_ROOT",
                "GCS_DATA_ROOT",
            )

        self.assertEqual(
            value,
            "/canonical",
        )

        self.assertEqual(
            source,
            "TRUSTFORGE_DATA_ROOT",
        )

    def test_legacy_environment_is_supported(self) -> None:
        environment = {
            "GCS_DATA_ROOT": "/legacy",
        }

        with mock.patch.dict(
            os.environ,
            environment,
            clear=True,
        ):
            value, source = doctor._environment_value(
                "TRUSTFORGE_DATA_ROOT",
                "GCS_DATA_ROOT",
            )

        self.assertEqual(
            value,
            "/legacy",
        )

        self.assertEqual(
            source,
            "GCS_DATA_ROOT",
        )

    def test_missing_data_root_is_failure(self) -> None:
        with mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            result = doctor._check_storage_root(
                name="Data root",
                canonical="TRUSTFORGE_DATA_ROOT",
                legacy="GCS_DATA_ROOT",
                require_read=True,
                require_write=False,
                require_traverse=True,
            )

        self.assertEqual(
            result.level,
            "FAIL",
        )

    def test_existing_data_root_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = {
                "TRUSTFORGE_DATA_ROOT": tmp,
            }

            with mock.patch.dict(
                os.environ,
                environment,
                clear=True,
            ):
                result = doctor._check_storage_root(
                    name="Data root",
                    canonical="TRUSTFORGE_DATA_ROOT",
                    legacy="GCS_DATA_ROOT",
                    require_read=True,
                    require_write=False,
                    require_traverse=True,
                )

        self.assertEqual(
            result.level,
            "PASS",
        )

    def test_artifacts_root_requires_write_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = {
                "TRUSTFORGE_ARTIFACTS_ROOT": tmp,
            }

            with mock.patch.dict(
                os.environ,
                environment,
                clear=True,
            ):
                with mock.patch.object(
                    doctor.os,
                    "access",
                    side_effect=lambda path, mode: (
                        False
                        if mode == os.W_OK
                        else True
                    ),
                ):
                    result = doctor._check_storage_root(
                        name="Artifacts root",
                        canonical="TRUSTFORGE_ARTIFACTS_ROOT",
                        legacy="GCS_ARTIFACTS_ROOT",
                        require_read=True,
                        require_write=True,
                        require_traverse=True,
                    )

        self.assertEqual(
            result.level,
            "FAIL",
        )

        self.assertIn(
            "write",
            result.detail,
        )

    def test_dirty_git_state_is_warning_not_failure(self) -> None:
        with mock.patch.object(
            doctor.shutil,
            "which",
            return_value="/usr/bin/git",
        ):
            with mock.patch.object(
                doctor,
                "_run_command",
                side_effect=[
                    (
                        0,
                        "a" * 40,
                        "",
                    ),
                    (
                        0,
                        "main",
                        "",
                    ),
                    (
                        0,
                        " M file.py",
                        "",
                    ),
                ],
            ):
                result = doctor.check_git_state(
                    Path("/repo")
                )

        self.assertEqual(
            result.level,
            "WARN",
        )

    def test_clean_git_state_passes(self) -> None:
        with mock.patch.object(
            doctor.shutil,
            "which",
            return_value="/usr/bin/git",
        ):
            with mock.patch.object(
                doctor,
                "_run_command",
                side_effect=[
                    (
                        0,
                        "b" * 40,
                        "",
                    ),
                    (
                        0,
                        "main",
                        "",
                    ),
                    (
                        0,
                        "",
                        "",
                    ),
                ],
            ):
                result = doctor.check_git_state(
                    Path("/repo")
                )

        self.assertEqual(
            result.level,
            "PASS",
        )

    def test_valid_dataset_identifier_passes_validation(self) -> None:
        self.assertIsNone(
            doctor._validate_dataset_identifier(
                "USTC-TFC2016_malware_nhwc"
            )
        )

    def test_dataset_identifier_rejects_parent_traversal(self) -> None:
        problem = doctor._validate_dataset_identifier(
            ".."
        )

        self.assertIsNotNone(
            problem
        )

    def test_dataset_identifier_rejects_forward_slash(self) -> None:
        problem = doctor._validate_dataset_identifier(
            "../dataset"
        )

        self.assertIsNotNone(
            problem
        )

    def test_dataset_identifier_rejects_backslash(self) -> None:
        problem = doctor._validate_dataset_identifier(
            "..\\dataset"
        )

        self.assertIsNotNone(
            problem
        )

    def test_existing_dataset_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            dataset = data_root / "example_dataset"

            dataset.mkdir(
                parents=True
            )

            environment = {
                "TRUSTFORGE_DATA_ROOT": str(
                    data_root
                ),
            }

            with mock.patch.dict(
                os.environ,
                environment,
                clear=True,
            ):
                result = doctor.check_dataset(
                    "example_dataset"
                )

        self.assertEqual(
            result.level,
            "PASS",
        )

    def test_missing_dataset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"

            data_root.mkdir()

            environment = {
                "TRUSTFORGE_DATA_ROOT": str(
                    data_root
                ),
            }

            with mock.patch.dict(
                os.environ,
                environment,
                clear=True,
            ):
                result = doctor.check_dataset(
                    "missing_dataset"
                )

        self.assertEqual(
            result.level,
            "FAIL",
        )

    def test_dataset_check_requires_data_root(self) -> None:
        with mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            result = doctor.check_dataset(
                "example_dataset"
            )

        self.assertEqual(
            result.level,
            "FAIL",
        )

    def test_filesystem_capacity_is_informational(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = {
                "TRUSTFORGE_DATA_ROOT": tmp,
            }

            with mock.patch.dict(
                os.environ,
                environment,
                clear=True,
            ):
                result = doctor.check_filesystem_capacity(
                    name="Data filesystem",
                    canonical="TRUSTFORGE_DATA_ROOT",
                    legacy="GCS_DATA_ROOT",
                )

        self.assertEqual(
            result.level,
            "INFO",
        )

    def test_missing_gpu_is_informational(self) -> None:
        with mock.patch.object(
            doctor.shutil,
            "which",
            return_value=None,
        ):
            result = doctor.check_gpu_capability()

        self.assertEqual(
            result.level,
            "INFO",
        )

    def test_missing_slurm_is_informational(self) -> None:
        with mock.patch.object(
            doctor.shutil,
            "which",
            return_value=None,
        ):
            result = doctor.check_slurm_capability()

        self.assertEqual(
            result.level,
            "INFO",
        )

    def test_main_returns_failure_when_core_check_fails(self) -> None:
        results = [
            doctor.CheckResult(
                "FAIL",
                "Example",
                "failed",
            )
        ]

        with mock.patch.object(
            doctor,
            "collect_results",
            return_value=results,
        ):
            with mock.patch.object(
                doctor,
                "_print_results",
            ):
                code = doctor.main(
                    []
                )

        self.assertEqual(
            code,
            1,
        )

    def test_main_returns_success_without_failures(self) -> None:
        results = [
            doctor.CheckResult(
                "PASS",
                "Example",
                "passed",
            ),
            doctor.CheckResult(
                "INFO",
                "Capability",
                "optional",
            ),
        ]

        with mock.patch.object(
            doctor,
            "collect_results",
            return_value=results,
        ):
            with mock.patch.object(
                doctor,
                "_print_results",
            ):
                code = doctor.main(
                    []
                )

        self.assertEqual(
            code,
            0,
        )

    def test_main_forwards_dataset_arguments(self) -> None:
        with mock.patch.object(
            doctor,
            "collect_results",
            return_value=[],
        ) as collect:
            with mock.patch.object(
                doctor,
                "_print_results",
            ):
                code = doctor.main(
                    [
                        "--dataset",
                        "dataset_one",
                        "--dataset",
                        "dataset_two",
                    ]
                )

        self.assertEqual(
            code,
            0,
        )

        collect.assert_called_once()

        self.assertEqual(
            collect.call_args.kwargs[
                "datasets"
            ],
            [
                "dataset_one",
                "dataset_two",
            ],
        )


if __name__ == "__main__":
    unittest.main()
