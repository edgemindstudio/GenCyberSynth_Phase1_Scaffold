#!/usr/bin/env python3
"""Generate frozen TrustForge experiment contracts for Paper 1.

This generator represents the scientific experiment grid of the historical
GenCyberSynth Phase-1 benchmark inside TrustForge.

Historical scientific configuration is loaded from explicit Git objects,
not from working-tree file bytes. This prevents checkout-specific newline
conversion or later working-tree changes from altering frozen provenance.

The generator does not:

- execute scientific workloads;
- modify historical configurations;
- modify historical artifacts;
- modify historical logs;
- read or write ~/gencys;
- retroactively classify historical executions as TrustForge-native runs.

The generated contracts describe scientific identities. Historical
execution-specific provenance is attached only when independently recovered
and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


STUDY_ID = "paper01_benchmark"

MODELS = (
    "gan",
    "vae",
    "diffusion",
    "autoregressive",
    "restrictedboltzmann",
    "gaussianmixture",
    "maskedautoflow",
)

SEEDS = (42, 43, 44)

PRIMARY_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "macro_auprc",
    "gen_precision",
    "gen_recall",
)

SUPPORTING_METRICS = (
    "kid",
    "ms_ssim",
)

ALL_METRICS = PRIMARY_METRICS + SUPPORTING_METRICS

REPOSITORY = "edgemindstudio/GenCyberSynth_Phase1_Scaffold"

DATASETS: dict[str, dict[str, Any]] = {
    "ustc": {
        "dataset_id": "ustc_tfc2016_malware_nhwc",
        "source_dataset": "USTC-TFC2016",
        "representation": "malware_image_nhwc",
        "preparation_stage": "paper1_ready",
        "config_source": "configs/paper1_final_rerun.yaml",
        "config_git_anchor": (
            "37c60929b3545e5ff68d9107280a63ece0b62f45"
        ),
        "config_sha256": (
            "559d2f47bd35c72bda6dc0b06a55d38312dfec6c88b4e2f20591118fa5492800"
        ),
        "image_shape": [40, 40, 1],
        "num_classes": 9,
        "input_scale": [-1, 1],
        "num_real": 90000,
        "samples_per_class": 2000,
        "total_samples": 18000,
        "classes": [f"class_{index}" for index in range(9)],
        "authoritative_result_sha256": (
            "c3c731f355f4cecfb930d024dd8ee42d97ce952d5abfca73c717af6b2a7a6031"
        ),
    },
    "cicmaldroid": {
        "dataset_id": "cicmaldroid2020_paper1",
        "source_dataset": "CICMalDroid2020",
        "representation": "malware_image_nhwc",
        "preparation_stage": "paper1_ready",
        "config_source": "configs/paper1_cicmaldroid.yaml",
        "config_git_anchor": "b75fce9",
        "config_sha256": (
            "1a0c328293875f59af243e0d7b9d0840d30b2656bed36b4739fb57f4e7e32e75"
        ),
        "image_shape": [12, 12, 1],
        "num_classes": 5,
        "input_scale": [0, 1],
        "num_real": 11598,
        "samples_per_class": 2000,
        "total_samples": 10000,
        "classes": [f"class_{index}" for index in range(5)],
        "authoritative_result_sha256": (
            "63fc2cd7b2cde81f2b964cc4407c6efdb1c50ee8bd61bfbbd4384d7eff8fc4f4"
        ),
    },
}


class Paper01GenerationError(RuntimeError):
    """Raised when Paper 1 experiment generation cannot proceed safely."""


def sha256_bytes(data: bytes) -> str:
    """Return the SHA256 digest of exact bytes."""

    return hashlib.sha256(data).hexdigest()


def run_git_bytes(
    repo_root: Path,
    *args: str,
) -> bytes:
    """Run Git and return exact stdout bytes."""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise Paper01GenerationError(
            "Git executable is required to reconstruct Paper 1 contracts."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()

        raise Paper01GenerationError(
            f"Git command failed: git {' '.join(args)}"
            + (f": {stderr}" if stderr else "")
        ) from exc

    return completed.stdout


def resolve_git_commit(
    repo_root: Path,
    revision: str,
) -> str:
    """Resolve a Git revision to a full commit identifier."""

    value = run_git_bytes(
        repo_root,
        "rev-parse",
        f"{revision}^{{commit}}",
    ).decode("ascii").strip()

    if len(value) != 40:
        raise Paper01GenerationError(
            f"Expected full Git commit for {revision!r}, got {value!r}."
        )

    return value


def git_blob_bytes(
    repo_root: Path,
    revision: str,
    relative_path: Path,
) -> bytes:
    """Return exact canonical Git blob bytes for a path at a revision."""

    spec = f"{revision}:{relative_path.as_posix()}"

    return run_git_bytes(
        repo_root,
        "cat-file",
        "blob",
        spec,
    )


def load_yaml_bytes_mapping(
    data: bytes,
    *,
    source: str,
) -> dict[str, Any]:
    """Parse YAML bytes and require a mapping at the root."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Paper01GenerationError(
            f"Historical YAML source is not UTF-8: {source}"
        ) from exc

    value = yaml.safe_load(text)

    if not isinstance(value, dict):
        raise Paper01GenerationError(
            f"Expected YAML mapping at {source}, "
            f"got {type(value).__name__}."
        )

    return value


def require_mapping(
    mapping: Mapping[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, Any]:
    """Return a required nested mapping."""

    value = mapping.get(key)

    if not isinstance(value, dict):
        raise Paper01GenerationError(
            f"Expected mapping {key!r} in historical config {source}."
        )

    return dict(value)


def require_value(
    mapping: Mapping[str, Any],
    key: str,
    *,
    source: str,
) -> Any:
    """Return a required value from a historical mapping."""

    if key not in mapping:
        raise Paper01GenerationError(
            f"Missing required key {key!r} in historical config {source}."
        )

    return mapping[key]


def model_implementation(model_id: str) -> str:
    """Return the historical implementation identity."""

    return f"historical_gencys_{model_id}"


def normalized_training(
    *,
    model_id: str,
    config: Mapping[str, Any],
    config_source: str,
) -> dict[str, Any]:
    """Translate historical model settings into the experiment contract."""

    block = require_mapping(
        config,
        model_id,
        source=config_source,
    )

    if model_id == "gan":
        train = require_mapping(
            config,
            "train",
            source=config_source,
        )

        ema = require_mapping(
            block,
            "EMA",
            source=config_source,
        )

        return {
            "epochs": int(
                require_value(
                    train,
                    "epochs",
                    source=config_source,
                )
            ),
            "batch_size": int(
                require_value(
                    train,
                    "batch_size",
                    source=config_source,
                )
            ),
            "learning_rate": float(
                require_value(
                    block,
                    "lr",
                    source=config_source,
                )
            ),
            "parameters": {
                "latent_dim": int(
                    require_value(
                        block,
                        "latent_dim",
                        source=config_source,
                    )
                ),
                "beta_1": float(
                    require_value(
                        block,
                        "beta_1",
                        source=config_source,
                    )
                ),
                "lr_g": float(
                    require_value(
                        block,
                        "LR_G",
                        source=config_source,
                    )
                ),
                "lr_d": float(
                    require_value(
                        block,
                        "LR_D",
                        source=config_source,
                    )
                ),
                "betas": list(
                    require_value(
                        block,
                        "BETAS",
                        source=config_source,
                    )
                ),
                "discriminator_updates_per_generator": int(
                    require_value(
                        block,
                        "D_UPDATES_PER_G",
                        source=config_source,
                    )
                ),
                "batch": int(
                    require_value(
                        block,
                        "BATCH",
                        source=config_source,
                    )
                ),
                "total_steps": int(
                    require_value(
                        block,
                        "TOTAL_STEPS",
                        source=config_source,
                    )
                ),
                "regularization": str(
                    require_value(
                        block,
                        "REG",
                        source=config_source,
                    )
                ),
                "spectral_norm": bool(
                    require_value(
                        block,
                        "SPECTRAL_NORM",
                        source=config_source,
                    )
                ),
                "ema_enabled": bool(
                    require_value(
                        ema,
                        "ENABLED",
                        source=config_source,
                    )
                ),
                "ema_decay": float(
                    require_value(
                        ema,
                        "DECAY",
                        source=config_source,
                    )
                ),
            },
        }

    if model_id == "vae":
        return {
            "batch_size": int(
                require_value(
                    block,
                    "BATCH",
                    source=config_source,
                )
            ),
            "learning_rate": float(
                require_value(
                    block,
                    "LR",
                    source=config_source,
                )
            ),
            "parameters": {
                "beta": float(
                    require_value(
                        block,
                        "BETA",
                        source=config_source,
                    )
                ),
                "kl_warmup_steps": int(
                    require_value(
                        block,
                        "KL_WARMUP_STEPS",
                        source=config_source,
                    )
                ),
                "total_steps": int(
                    require_value(
                        block,
                        "TOTAL_STEPS",
                        source=config_source,
                    )
                ),
            },
        }

    if model_id == "diffusion":
        return {
            "batch_size": int(
                require_value(
                    block,
                    "BATCH",
                    source=config_source,
                )
            ),
            "learning_rate": float(
                require_value(
                    block,
                    "LR",
                    source=config_source,
                )
            ),
            "parameters": {
                "ema": float(
                    require_value(
                        block,
                        "EMA",
                        source=config_source,
                    )
                ),
                "schedule": str(
                    require_value(
                        block,
                        "SCHEDULE",
                        source=config_source,
                    )
                ),
                "timesteps": int(
                    require_value(
                        block,
                        "TIMESTEPS",
                        source=config_source,
                    )
                ),
                "base_filters": int(
                    require_value(
                        block,
                        "BASE_FILTERS",
                        source=config_source,
                    )
                ),
                "depth": int(
                    require_value(
                        block,
                        "DEPTH",
                        source=config_source,
                    )
                ),
                "time_embedding_dim": int(
                    require_value(
                        block,
                        "TIME_EMB_DIM",
                        source=config_source,
                    )
                ),
                "total_steps": int(
                    require_value(
                        block,
                        "TOTAL_STEPS",
                        source=config_source,
                    )
                ),
                "sampler_steps": int(
                    require_value(
                        block,
                        "SAMPLER_STEPS",
                        source=config_source,
                    )
                ),
                "guidance_weight": float(
                    require_value(
                        block,
                        "GUIDANCE_WEIGHT",
                        source=config_source,
                    )
                ),
            },
        }

    if model_id == "autoregressive":
        sampling = require_mapping(
            block,
            "SAMPLING",
            source=config_source,
        )

        return {
            "batch_size": int(
                require_value(
                    block,
                    "BATCH",
                    source=config_source,
                )
            ),
            "learning_rate": float(
                require_value(
                    block,
                    "LR",
                    source=config_source,
                )
            ),
            "parameters": {
                "total_steps": int(
                    require_value(
                        block,
                        "TOTAL_STEPS",
                        source=config_source,
                    )
                ),
                "sampling": {
                    "temperature": float(
                        require_value(
                            sampling,
                            "TEMPERATURE",
                            source=config_source,
                        )
                    ),
                    "top_k": int(
                        require_value(
                            sampling,
                            "TOP_K",
                            source=config_source,
                        )
                    ),
                    "top_p": float(
                        require_value(
                            sampling,
                            "TOP_P",
                            source=config_source,
                        )
                    ),
                },
            },
        }

    if model_id == "restrictedboltzmann":
        return {
            "epochs": int(
                require_value(
                    block,
                    "EPOCHS",
                    source=config_source,
                )
            ),
            "batch_size": int(
                require_value(
                    block,
                    "BATCH",
                    source=config_source,
                )
            ),
            "learning_rate": float(
                require_value(
                    block,
                    "LR",
                    source=config_source,
                )
            ),
            "parameters": {
                "cd_k": int(
                    require_value(
                        block,
                        "CD_K",
                        source=config_source,
                    )
                ),
                "hidden_units": int(
                    require_value(
                        block,
                        "HIDDEN",
                        source=config_source,
                    )
                ),
                "train_per_class_cap": int(
                    require_value(
                        block,
                        "TRAIN_PER_CLASS_CAP",
                        source=config_source,
                    )
                ),
            },
        }

    if model_id == "gaussianmixture":
        return {
            "parameters": {
                "components": int(
                    require_value(
                        block,
                        "GMM_COMPONENTS",
                        source=config_source,
                    )
                ),
                "covariance": str(
                    require_value(
                        block,
                        "GMM_COVARIANCE",
                        source=config_source,
                    )
                ),
                "reg_covar": float(
                    require_value(
                        block,
                        "GMM_REG_COVAR",
                        source=config_source,
                    )
                ),
                "max_iter": int(
                    require_value(
                        block,
                        "GMM_MAX_ITER",
                        source=config_source,
                    )
                ),
                "train_global": bool(
                    require_value(
                        block,
                        "GMM_TRAIN_GLOBAL",
                        source=config_source,
                    )
                ),
            },
        }

    if model_id == "maskedautoflow":
        return {
            "batch_size": int(
                require_value(
                    block,
                    "BATCH",
                    source=config_source,
                )
            ),
            "learning_rate": float(
                require_value(
                    block,
                    "LR",
                    source=config_source,
                )
            ),
            "parameters": {
                "depth": int(
                    require_value(
                        block,
                        "DEPTH",
                        source=config_source,
                    )
                ),
                "width": int(
                    require_value(
                        block,
                        "WIDTH",
                        source=config_source,
                    )
                ),
                "total_steps": int(
                    require_value(
                        block,
                        "TOTAL_STEPS",
                        source=config_source,
                    )
                ),
                "dequant_noise": float(
                    require_value(
                        block,
                        "DEQUANT_NOISE",
                        source=config_source,
                    )
                ),
            },
        }

    raise Paper01GenerationError(
        f"Unsupported model_id: {model_id}"
    )


def validate_dataset_config(
    *,
    dataset_key: str,
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    config_source: str,
) -> None:
    """Verify frozen dataset-level controls against historical Git content."""

    expected_seeds = list(SEEDS)

    checks = {
        "random_seeds": expected_seeds,
        "SAMPLES_PER_CLASS": dataset["samples_per_class"],
        "IMG_SHAPE": dataset["image_shape"],
        "NUM_CLASSES": dataset["num_classes"],
        "scale": dataset["input_scale"],
    }

    for key, expected in checks.items():
        actual = config.get(key)

        if actual != expected:
            raise Paper01GenerationError(
                f"{dataset_key}: historical config "
                f"{config_source} has {key}={actual!r}; "
                f"expected {expected!r}."
            )

    synth = require_mapping(
        config,
        "synth",
        source=config_source,
    )

    if synth.get("n_per_class") != dataset["samples_per_class"]:
        raise Paper01GenerationError(
            f"{dataset_key}: synth.n_per_class does not match "
            "the frozen Paper 1 budget."
        )

    evaluator = require_mapping(
        config,
        "evaluator",
        source=config_source,
    )

    evaluator_checks = {
        "per_class_cap": 200,
        "feature_extractor": "domain",
        "fid_split": "val",
        "domain_encoder": "malware_encoder_v1",
    }

    for key, expected in evaluator_checks.items():
        actual = evaluator.get(key)

        if actual != expected:
            raise Paper01GenerationError(
                f"{dataset_key}: evaluator.{key}={actual!r}; "
                f"expected {expected!r}."
            )

    run_meta = require_mapping(
        config,
        "run_meta",
        source=config_source,
    )

    if run_meta.get("num_real") != dataset["num_real"]:
        raise Paper01GenerationError(
            f"{dataset_key}: run_meta.num_real does not match "
            "the frozen Paper 1 study map."
        )


def experiment_id(
    dataset_key: str,
    model_id: str,
    seed: int,
) -> str:
    """Return the stable scientific experiment identifier."""

    return f"{dataset_key}_{model_id}_b2000_seed{seed}"


def historical_run_id(
    model_id: str,
    seed: int,
) -> str:
    """Return the run identifier used by accepted Paper 1 score tables."""

    return f"{model_id}_s{seed}"


def build_experiment(
    *,
    dataset_key: str,
    dataset: Mapping[str, Any],
    model_id: str,
    seed: int,
    config: Mapping[str, Any],
    config_path: Path,
    config_git_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Build one frozen Paper 1 experiment contract."""

    exp_id = experiment_id(
        dataset_key,
        model_id,
        seed,
    )

    run_id = historical_run_id(
        model_id,
        seed,
    )

    config_source = (
        f"{config_git_commit}:{config_path.as_posix()}"
    )

    return {
        "schema_version": "1.0",
        "schema_type": "experiment",
        "study_id": STUDY_ID,
        "experiment_id": exp_id,
        "status": "frozen",
        "description": (
            f"Historical Paper 1 {model_id} experiment on "
            f"{dataset['source_dataset']} using seed {seed} and a "
            "synthetic budget of 2000 samples per class."
        ),
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "source_dataset": dataset["source_dataset"],
            "representation": dataset["representation"],
            "preparation_stage": dataset["preparation_stage"],
        },
        "model": {
            "model_id": model_id,
            "family": model_id,
            "implementation": model_implementation(model_id),
        },
        "seed": seed,
        "training": normalized_training(
            model_id=model_id,
            config=config,
            config_source=config_source,
        ),
        "generation": {
            "samples_per_class": dataset["samples_per_class"],
            "total_samples": dataset["total_samples"],
            "classes": list(dataset["classes"]),
        },
        "evaluation": {
            "evaluator_id": "historical_paper1_evaluator",
            "metrics": list(ALL_METRICS),
            "primary_metrics": list(PRIMARY_METRICS),
            "supporting_metrics": list(SUPPORTING_METRICS),
            "evaluation_split": "val",
            "parameters": {
                "per_class_cap": 200,
                "feature_extractor": "domain",
                "domain_encoder": "malware_encoder_v1",
            },
        },
        "scientific_parameters": {
            "image_shape": list(dataset["image_shape"]),
            "num_classes": dataset["num_classes"],
            "input_scale": list(dataset["input_scale"]),
            "num_real": dataset["num_real"],
            "historical_run_id": run_id,
            "historical_config_git_commit": config_git_commit,
            "authoritative_result_sha256": (
                dataset["authoritative_result_sha256"]
            ),
        },
        "execution_requirements": {
            "accelerator": "gpu",
            "gpu_count": 1,
            "cpu_count": 8,
            "memory_gb": 48,
            "walltime": "27-00:00:00",
        },
        "provenance": {
            "repository": REPOSITORY,
            "config_source": config_path.as_posix(),
            "config_sha256": config_sha256,
        },
        "acceptance": {
            "required": True,
            "criteria": [
                "historical_run_id_present_in_authoritative_result_table",
                "authoritative_result_hash_preserved",
                "historical_scientific_definition_preserved",
            ],
        },
        "notes": [
            (
                "Historical experiment reconstructed from the frozen "
                "Paper 1 scientific protocol."
            ),
            (
                "The historical configuration is loaded from the recorded "
                "Git commit rather than from working-tree file bytes."
            ),
            (
                "This contract represents scientific identity and does not "
                "retroactively classify the historical execution as a "
                "TrustForge-native execution."
            ),
            (
                "Execution-specific Git provenance is intentionally omitted "
                "unless independently recovered and verified for that "
                "historical execution."
            ),
        ],
    }


def render_yaml(
    value: Mapping[str, Any],
) -> str:
    """Render deterministic YAML text."""

    return yaml.safe_dump(
        dict(value),
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
        width=100,
    )


def build_outputs(
    repo_root: Path,
) -> dict[Path, str]:
    """Build all generated Paper 1 experiment files in memory."""

    repo_root = repo_root.resolve()

    outputs: dict[Path, str] = {}
    index_entries: list[dict[str, Any]] = []
    config_sources: dict[str, dict[str, str]] = {}

    for dataset_key, dataset in DATASETS.items():
        relative_config = Path(
            dataset["config_source"]
        )

        configured_anchor = str(
            dataset["config_git_anchor"]
        )

        resolved_commit = resolve_git_commit(
            repo_root,
            configured_anchor,
        )

        blob = git_blob_bytes(
            repo_root,
            resolved_commit,
            relative_config,
        )

        config_digest = sha256_bytes(blob)

        expected_digest = str(
            dataset["config_sha256"]
        )

        if config_digest != expected_digest:
            raise Paper01GenerationError(
                f"{dataset_key}: historical config digest mismatch. "
                f"Expected {expected_digest}, got {config_digest} "
                f"for {resolved_commit}:{relative_config.as_posix()}."
            )

        config_source = (
            f"{resolved_commit}:"
            f"{relative_config.as_posix()}"
        )

        config = load_yaml_bytes_mapping(
            blob,
            source=config_source,
        )

        validate_dataset_config(
            dataset_key=dataset_key,
            dataset=dataset,
            config=config,
            config_source=config_source,
        )

        config_sources[dataset_key] = {
            "dataset_id": str(
                dataset["dataset_id"]
            ),
            "config_path": relative_config.as_posix(),
            "git_commit": resolved_commit,
            "sha256": config_digest,
        }

        for model_id in MODELS:
            for seed in SEEDS:
                exp_id = experiment_id(
                    dataset_key,
                    model_id,
                    seed,
                )

                relative_output = (
                    Path("experiments")
                    / dataset_key
                    / f"{exp_id}.yaml"
                )

                contract = build_experiment(
                    dataset_key=dataset_key,
                    dataset=dataset,
                    model_id=model_id,
                    seed=seed,
                    config=config,
                    config_path=relative_config,
                    config_git_commit=resolved_commit,
                    config_sha256=config_digest,
                )

                outputs[relative_output] = render_yaml(
                    contract
                )

                index_entries.append(
                    {
                        "experiment_id": exp_id,
                        "dataset_id": dataset["dataset_id"],
                        "model_id": model_id,
                        "seed": seed,
                        "historical_run_id": (
                            historical_run_id(
                                model_id,
                                seed,
                            )
                        ),
                        "contract": (
                            relative_output.as_posix()
                        ),
                    }
                )

    index = {
        "experiment_index_version": "1.0",
        "study_id": STUDY_ID,
        "status": "frozen",
        "generation_rule": (
            "2 datasets x 7 model families x 3 seeds = "
            "42 frozen scientific experiment identities"
        ),
        "axes": {
            "datasets": [
                DATASETS["ustc"]["dataset_id"],
                DATASETS["cicmaldroid"]["dataset_id"],
            ],
            "models": list(MODELS),
            "seeds": list(SEEDS),
            "samples_per_class": 2000,
        },
        "counts": {
            "datasets": 2,
            "models": 7,
            "seeds": 3,
            "experiments": 42,
        },
        "historical_config_sources": config_sources,
        "entries": index_entries,
        "notes": [
            (
                "Experiment identities are machine-independent even where "
                "historical execution paths were machine-specific."
            ),
            (
                "Historical configuration digests are computed from "
                "canonical Git blob bytes, not checkout-specific "
                "working-tree bytes."
            ),
            (
                "Historical run IDs bind these contracts to rows in the "
                "accepted Paper 1 result tables."
            ),
            (
                "Exact execution provenance is not inferred from scientific "
                "grid membership."
            ),
        ],
    }

    outputs[
        Path("experiment_index.yaml")
    ] = render_yaml(index)

    return outputs


def write_outputs(
    *,
    output_root: Path,
    outputs: Mapping[Path, str],
) -> None:
    """Write generated outputs without deleting unrelated files."""

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for relative_path, text in outputs.items():
        destination = (
            output_root / relative_path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_text(
            text,
            encoding="utf-8",
        )


def check_outputs(
    *,
    output_root: Path,
    outputs: Mapping[Path, str],
) -> list[str]:
    """Return differences between expected and existing generated output."""

    issues: list[str] = []

    for relative_path, expected in outputs.items():
        destination = (
            output_root / relative_path
        )

        if not destination.is_file():
            issues.append(
                f"missing: {relative_path}"
            )
            continue

        actual = destination.read_text(
            encoding="utf-8"
        )

        if actual != expected:
            issues.append(
                f"content differs: {relative_path}"
            )

    expected_contracts = {
        path
        for path in outputs
        if path.parts
        and path.parts[0] == "experiments"
    }

    experiments_root = (
        output_root / "experiments"
    )

    if experiments_root.exists():
        actual_contracts = {
            path.relative_to(output_root)
            for path in experiments_root.rglob(
                "*.yaml"
            )
            if path.is_file()
        }

        unexpected = sorted(
            actual_contracts
            - expected_contracts
        )

        for relative_path in unexpected:
            issues.append(
                "unexpected generated contract: "
                f"{relative_path}"
            )

    return issues


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify frozen TrustForge "
            "Paper 1 experiment contracts."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Repository root. Defaults to the "
            "current directory."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "studies/paper01_benchmark"
        ),
        help=(
            "Generated study output root. "
            "Defaults to studies/paper01_benchmark."
        ),
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify committed generated files "
            "without modifying them."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Generate or check the Paper 1 experiment representation."""

    args = parse_args()

    repo_root = (
        args.repo_root.resolve()
    )

    output_root = args.output_root

    if not output_root.is_absolute():
        output_root = (
            repo_root / output_root
        )

    outputs = build_outputs(
        repo_root
    )

    contract_count = sum(
        1
        for path in outputs
        if path.parts
        and path.parts[0] == "experiments"
    )

    if contract_count != 42:
        raise Paper01GenerationError(
            "Expected 42 experiment contracts, "
            f"built {contract_count}."
        )

    if args.check:
        issues = check_outputs(
            output_root=output_root,
            outputs=outputs,
        )

        if issues:
            print(
                "Paper 1 experiment generation "
                "check: FAIL"
            )

            for issue in issues:
                print(f" - {issue}")

            return 1

        print(
            "Paper 1 experiment generation "
            "check: PASS"
        )
        print(
            f"Experiment contracts: "
            f"{contract_count}"
        )

        return 0

    write_outputs(
        output_root=output_root,
        outputs=outputs,
    )

    print(
        "Paper 1 experiment generation: PASS"
    )
    print(
        f"Experiment contracts: "
        f"{contract_count}"
    )
    print(
        f"Output root: {output_root}"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Paper01GenerationError as exc:
        print(
            "Paper 1 experiment generation: "
            f"FAIL: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
