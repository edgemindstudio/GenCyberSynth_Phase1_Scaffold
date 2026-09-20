from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

from trustforge.provenance import validate_contract_file


REPO_ROOT = Path(__file__).resolve().parents[2]

GENERATOR_PATH = (
    REPO_ROOT
    / "scripts/generate_paper01_experiments.py"
)

EXPERIMENT_SCHEMA = (
    REPO_ROOT
    / "schemas/experiment.schema.yaml"
)

STUDY_ROOT = (
    REPO_ROOT
    / "studies/paper01_benchmark"
)


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_paper01_experiments",
        GENERATOR_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load Paper 1 generator "
            f"from {GENERATOR_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


generator = load_generator()


class Paper01ExperimentGenerationTests(
    unittest.TestCase
):
    def test_grid_contains_exactly_42_contracts(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        contracts = [
            path
            for path in outputs
            if path.parts
            and path.parts[0] == "experiments"
        ]

        self.assertEqual(
            len(contracts),
            42,
        )

    def test_index_is_generated(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        self.assertIn(
            Path("experiment_index.yaml"),
            outputs,
        )

    def test_expected_dataset_directories(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        contract_paths = {
            path
            for path in outputs
            if path.parts
            and path.parts[0] == "experiments"
        }

        ustc = [
            path
            for path in contract_paths
            if path.parts[1] == "ustc"
        ]

        cic = [
            path
            for path in contract_paths
            if path.parts[1] == "cicmaldroid"
        ]

        self.assertEqual(
            len(ustc),
            21,
        )

        self.assertEqual(
            len(cic),
            21,
        )

    def test_every_model_seed_pair_exists_for_both_datasets(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        for dataset_key in (
            "ustc",
            "cicmaldroid",
        ):
            for model_id in generator.MODELS:
                for seed in generator.SEEDS:
                    exp_id = (
                        generator.experiment_id(
                            dataset_key,
                            model_id,
                            seed,
                        )
                    )

                    path = (
                        Path("experiments")
                        / dataset_key
                        / f"{exp_id}.yaml"
                    )

                    self.assertIn(
                        path,
                        outputs,
                    )

    def test_all_generated_contracts_validate(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        for relative_path, text in outputs.items():
            if (
                not relative_path.parts
                or relative_path.parts[0]
                != "experiments"
            ):
                continue

            record = yaml.safe_load(
                text
            )

            validate_contract_file(
                record,
                EXPERIMENT_SCHEMA,
            )

    def test_all_contracts_are_frozen_paper01_experiments(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        for relative_path, text in outputs.items():
            if (
                not relative_path.parts
                or relative_path.parts[0]
                != "experiments"
            ):
                continue

            record = yaml.safe_load(
                text
            )

            self.assertEqual(
                record["study_id"],
                "paper01_benchmark",
            )

            self.assertEqual(
                record["status"],
                "frozen",
            )

    def test_expected_git_commit_is_not_inferred(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        for relative_path, text in outputs.items():
            if (
                not relative_path.parts
                or relative_path.parts[0]
                != "experiments"
            ):
                continue

            record = yaml.safe_load(
                text
            )

            self.assertNotIn(
                "expected_git_commit",
                record["provenance"],
            )

    def test_historical_config_commit_is_recorded_separately(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        for dataset_key, dataset in (
            generator.DATASETS.items()
        ):
            resolved = (
                generator.resolve_git_commit(
                    REPO_ROOT,
                    dataset[
                        "config_git_anchor"
                    ],
                )
            )

            exp_id = (
                generator.experiment_id(
                    dataset_key,
                    "gan",
                    42,
                )
            )

            path = (
                Path("experiments")
                / dataset_key
                / f"{exp_id}.yaml"
            )

            record = yaml.safe_load(
                outputs[path]
            )

            self.assertEqual(
                record[
                    "scientific_parameters"
                ][
                    "historical_config_git_commit"
                ],
                resolved,
            )

            self.assertNotIn(
                "expected_git_commit",
                record["provenance"],
            )

    def test_historical_run_ids_follow_accepted_table_identity(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        for dataset_key in (
            "ustc",
            "cicmaldroid",
        ):
            for model_id in generator.MODELS:
                for seed in generator.SEEDS:
                    exp_id = (
                        generator.experiment_id(
                            dataset_key,
                            model_id,
                            seed,
                        )
                    )

                    path = (
                        Path("experiments")
                        / dataset_key
                        / f"{exp_id}.yaml"
                    )

                    record = yaml.safe_load(
                        outputs[path]
                    )

                    self.assertEqual(
                        record[
                            "scientific_parameters"
                        ][
                            "historical_run_id"
                        ],
                        f"{model_id}_s{seed}",
                    )

    def test_ustc_scientific_shape_and_budget(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        path = Path(
            "experiments/ustc/"
            "ustc_gan_b2000_seed42.yaml"
        )

        record = yaml.safe_load(
            outputs[path]
        )

        self.assertEqual(
            record["dataset"]["dataset_id"],
            "ustc_tfc2016_malware_nhwc",
        )

        self.assertEqual(
            record[
                "scientific_parameters"
            ][
                "image_shape"
            ],
            [40, 40, 1],
        )

        self.assertEqual(
            record[
                "scientific_parameters"
            ][
                "num_classes"
            ],
            9,
        )

        self.assertEqual(
            record["generation"][
                "samples_per_class"
            ],
            2000,
        )

        self.assertEqual(
            record["generation"][
                "total_samples"
            ],
            18000,
        )

    def test_cic_scientific_shape_and_budget(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        path = Path(
            "experiments/cicmaldroid/"
            "cicmaldroid_gan_b2000_seed42.yaml"
        )

        record = yaml.safe_load(
            outputs[path]
        )

        self.assertEqual(
            record["dataset"]["dataset_id"],
            "cicmaldroid2020_paper1",
        )

        self.assertEqual(
            record[
                "scientific_parameters"
            ][
                "image_shape"
            ],
            [12, 12, 1],
        )

        self.assertEqual(
            record[
                "scientific_parameters"
            ][
                "num_classes"
            ],
            5,
        )

        self.assertEqual(
            record["generation"][
                "samples_per_class"
            ],
            2000,
        )

        self.assertEqual(
            record["generation"][
                "total_samples"
            ],
            10000,
        )

    def test_authoritative_hashes_are_dataset_specific(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        ustc = yaml.safe_load(
            outputs[
                Path(
                    "experiments/ustc/"
                    "ustc_gan_b2000_seed42.yaml"
                )
            ]
        )

        cic = yaml.safe_load(
            outputs[
                Path(
                    "experiments/cicmaldroid/"
                    "cicmaldroid_gan_b2000_seed42.yaml"
                )
            ]
        )

        self.assertEqual(
            ustc[
                "scientific_parameters"
            ][
                "authoritative_result_sha256"
            ],
            (
                "c3c731f355f4cecfb930d024dd8ee42d"
                "97ce952d5abfca73c717af6b2a7a6031"
            ),
        )

        self.assertEqual(
            cic[
                "scientific_parameters"
            ][
                "authoritative_result_sha256"
            ],
            (
                "63fc2cd7b2cde81f2b964cc4407c6ef"
                "db1c50ee8bd61bfbbd4384d7eff8fc4f4"
            ),
        )

    def test_config_digest_matches_canonical_historical_git_blob(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        for dataset_key, dataset in (
            generator.DATASETS.items()
        ):
            resolved_commit = (
                generator.resolve_git_commit(
                    REPO_ROOT,
                    dataset[
                        "config_git_anchor"
                    ],
                )
            )

            relative_config = Path(
                dataset["config_source"]
            )

            blob = generator.git_blob_bytes(
                REPO_ROOT,
                resolved_commit,
                relative_config,
            )

            expected_digest = (
                generator.sha256_bytes(
                    blob
                )
            )

            self.assertEqual(
                expected_digest,
                dataset["config_sha256"],
            )

            exp_id = (
                generator.experiment_id(
                    dataset_key,
                    "gan",
                    42,
                )
            )

            path = (
                Path("experiments")
                / dataset_key
                / f"{exp_id}.yaml"
            )

            record = yaml.safe_load(
                outputs[path]
            )

            self.assertEqual(
                record["provenance"][
                    "config_sha256"
                ],
                expected_digest,
            )

    def test_known_canonical_config_hashes(
        self,
    ):
        self.assertEqual(
            generator.DATASETS[
                "ustc"
            ][
                "config_sha256"
            ],
            (
                "559d2f47bd35c72bda6dc0b06a55d383"
                "12dfec6c88b4e2f20591118fa5492800"
            ),
        )

        self.assertEqual(
            generator.DATASETS[
                "cicmaldroid"
            ][
                "config_sha256"
            ],
            (
                "1a0c328293875f59af243e0d7b9d0840"
                "d30b2656bed36b4739fb57f4e7e32e75"
            ),
        )

    def test_generation_is_byte_deterministic(
        self,
    ):
        first = generator.build_outputs(
            REPO_ROOT
        )

        second = generator.build_outputs(
            REPO_ROOT
        )

        self.assertEqual(
            first,
            second,
        )

    def test_check_mode_logic_detects_missing_output(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(
                directory
            )

            issues = generator.check_outputs(
                output_root=output_root,
                outputs=outputs,
            )

        self.assertTrue(
            issues
        )

        self.assertIn(
            "missing: experiment_index.yaml",
            issues,
        )

    def test_written_outputs_match_in_memory_generation(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(
                directory
            )

            generator.write_outputs(
                output_root=output_root,
                outputs=outputs,
            )

            issues = generator.check_outputs(
                output_root=output_root,
                outputs=outputs,
            )

            self.assertEqual(
                issues,
                [],
            )

    def test_index_contains_exact_axes_and_count(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        index = yaml.safe_load(
            outputs[
                Path("experiment_index.yaml")
            ]
        )

        self.assertEqual(
            index["study_id"],
            "paper01_benchmark",
        )

        self.assertEqual(
            index["counts"][
                "experiments"
            ],
            42,
        )

        self.assertEqual(
            index["counts"][
                "datasets"
            ],
            2,
        )

        self.assertEqual(
            index["counts"][
                "models"
            ],
            7,
        )

        self.assertEqual(
            index["counts"][
                "seeds"
            ],
            3,
        )

        self.assertEqual(
            index["axes"][
                "seeds"
            ],
            [42, 43, 44],
        )

    def test_index_records_canonical_config_sources(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        index = yaml.safe_load(
            outputs[
                Path("experiment_index.yaml")
            ]
        )

        sources = index[
            "historical_config_sources"
        ]

        self.assertEqual(
            sources["ustc"]["sha256"],
            generator.DATASETS[
                "ustc"
            ][
                "config_sha256"
            ],
        )

        self.assertEqual(
            sources[
                "cicmaldroid"
            ][
                "sha256"
            ],
            generator.DATASETS[
                "cicmaldroid"
            ][
                "config_sha256"
            ],
        )

        for dataset_key, dataset in (
            generator.DATASETS.items()
        ):
            expected_commit = (
                generator.resolve_git_commit(
                    REPO_ROOT,
                    dataset[
                        "config_git_anchor"
                    ],
                )
            )

            self.assertEqual(
                sources[
                    dataset_key
                ][
                    "git_commit"
                ],
                expected_commit,
            )

    def test_committed_generated_outputs_are_current(
        self,
    ):
        outputs = generator.build_outputs(
            REPO_ROOT
        )

        if not (
            STUDY_ROOT
            / "experiment_index.yaml"
        ).exists():
            self.skipTest(
                "Generated Paper 1 experiment "
                "contracts have not been created yet."
            )

        issues = generator.check_outputs(
            output_root=STUDY_ROOT,
            outputs=outputs,
        )

        self.assertEqual(
            issues,
            [],
        )


if __name__ == "__main__":
    unittest.main()
