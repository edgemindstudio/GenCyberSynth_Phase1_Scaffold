"""Tests for TrustForge contract validation."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trustforge.provenance import (
    ContractSchemaError,
    ContractValidationError,
    GitState,
    build_native_manifest,
    collect_contract_issues,
    load_contract_schema,
    validate_manifest_contract,
)


class ProvenanceValidationTests(unittest.TestCase):
    """Validate TrustForge runtime records against M3 contracts."""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]

        self.schema_path = (
            self.repo_root
            / "schemas"
            / "manifest.schema.yaml"
        )

    def _git_state(
        self,
        root: Path,
    ) -> GitState:
        return GitState(
            repo_root=root.resolve(),
            commit="a" * 40,
            branch="migration/trustforge-foundation",
            dirty=False,
        )

    def _native_manifest(
        self,
        root: Path,
    ) -> dict:
        return build_native_manifest(
            study_id="paper01_benchmark",
            experiment_id="ustc_gan_b2000_seed42",
            execution_id=(
                "20260918T013000Z_talon_job123456"
            ),
            repository="edgemindstudio/trustforge",
            repo_root=root,
            resolved_config={
                "model": "gan",
                "seed": 42,
            },
            dataset_id="USTC-TFC2016_malware_nhwc",
            dataset_path=root / "data",
            artifact_root=root / "artifacts",
            status="running",
            seed=42,
            started_at=datetime(
                2026,
                9,
                18,
                1,
                30,
                0,
                tzinfo=timezone.utc,
            ),
            machine_role="talon_hpc",
            scheduler={
                "backend": "slurm",
                "job_id": "123456",
                "array_task_id": "0",
            },
            git_state=self._git_state(root),
        )

    def test_manifest_schema_loads(self) -> None:
        schema = load_contract_schema(
            self.schema_path
        )

        self.assertEqual(
            schema["schema_type"],
            "manifest",
        )

        self.assertEqual(
            schema["schema_version"],
            "1.0",
        )

    def test_native_manifest_satisfies_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._native_manifest(root)

            validate_manifest_contract(
                manifest,
                self.schema_path,
            )

    def test_missing_required_top_level_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._native_manifest(root)

            del manifest["execution_id"]

            with self.assertRaises(
                ContractValidationError
            ):
                validate_manifest_contract(
                    manifest,
                    self.schema_path,
                )

    def test_missing_required_nested_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._native_manifest(root)

            del manifest["provenance"]["git_commit"]

            with self.assertRaises(
                ContractValidationError
            ):
                validate_manifest_contract(
                    manifest,
                    self.schema_path,
                )

    def test_wrong_string_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._native_manifest(root)

            manifest["study_id"] = 42

            with self.assertRaises(
                ContractValidationError
            ):
                validate_manifest_contract(
                    manifest,
                    self.schema_path,
                )

    def test_boolean_is_not_accepted_as_integer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._native_manifest(root)

            manifest["seed"] = True

            with self.assertRaises(
                ContractValidationError
            ):
                validate_manifest_contract(
                    manifest,
                    self.schema_path,
                )

    def test_invalid_enum_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._native_manifest(root)

            manifest["status"] = "accepted"

            with self.assertRaises(
                ContractValidationError
            ):
                validate_manifest_contract(
                    manifest,
                    self.schema_path,
                )

    def test_wrong_schema_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._native_manifest(root)

            manifest["schema_version"] = "999.0"

            with self.assertRaises(
                ContractValidationError
            ):
                validate_manifest_contract(
                    manifest,
                    self.schema_path,
                )

    def test_wrong_schema_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._native_manifest(root)

            manifest["schema_type"] = "experiment"

            with self.assertRaises(
                ContractValidationError
            ):
                validate_manifest_contract(
                    manifest,
                    self.schema_path,
                )

    def test_multiple_issues_are_collected(self) -> None:
        schema = load_contract_schema(
            self.schema_path
        )

        broken = {
            "study_id": 123,
            "status": "not-a-status",
        }

        issues = collect_contract_issues(
            broken,
            schema,
        )

        self.assertGreaterEqual(
            len(issues),
            2,
        )

    def test_non_manifest_schema_is_rejected(self) -> None:
        experiment_schema = (
            self.repo_root
            / "schemas"
            / "experiment.schema.yaml"
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = self._native_manifest(root)

            with self.assertRaises(
                ContractSchemaError
            ):
                validate_manifest_contract(
                    manifest,
                    experiment_schema,
                )

    def test_historical_example_satisfies_manifest_contract(self) -> None:
        import yaml

        example_path = (
            self.repo_root
            / "schemas"
            / "examples"
            / "paper01"
            / "manifest_ustc_gan_seed42.yaml"
        )

        historical = yaml.safe_load(
            example_path.read_text(
                encoding="utf-8"
            )
        )

        validate_manifest_contract(
            historical,
            self.schema_path,
        )


if __name__ == "__main__":
    unittest.main()
