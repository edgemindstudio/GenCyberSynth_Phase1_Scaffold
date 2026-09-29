"""
Canonical Paper 1 consumer projections.

M6.5.3A defines a narrow downstream-consumption view over the canonical
Paper 1 execution/evidence linkage materialization.

The projection requires explicit scientific identity:
    dataset + seed

It never selects "latest" evidence and never reads or modifies historical
artifact files. The canonical linkage layer decides which historical
timestamped summary is authoritative; this module only exposes that validated
selection to downstream consumers.

Permanent distinctions preserved:
    SCIENTIFIC EXPERIMENT IDENTITY != HISTORICAL EXECUTION IDENTITY
    CANONICAL AUTHORITY SELECTION != HISTORICAL METRIC PAYLOAD
    latest.json != AUTHORITATIVE EVIDENCE
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Any

from trustforge.paper01_execution_evidence import (
    Paper01LinkageRecord,
    Paper01LinkageStudy,
    load_paper01_linkage_study,
)


EXPECTED_DATASETS = frozenset(
    {
        "ustc_tfc2016",
        "cicmaldroid2020",
    }
)

EXPECTED_SEEDS = frozenset({42, 43, 44})

EXPECTED_FAMILIES = frozenset(
    {
        "gan",
        "vae",
        "diffusion",
        "autoregressive",
        "restrictedboltzmann",
        "gaussianmixture",
        "maskedautoflow",
    }
)

EXPECTED_RECORDS_PER_VIEW = 7


class Paper01ConsumerViewError(RuntimeError):
    """Raised when a requested canonical consumer projection is invalid."""


@dataclass(frozen=True)
class Paper01ConsumerRecord:
    """One canonical Paper 1 experiment projected for downstream use."""

    experiment_id: str
    dataset: str
    family: str
    seed: int
    budget_per_class: int
    historical_run_id: str
    accepted_summary_path: Path
    linkage_record: Paper01LinkageRecord

    @property
    def linkage_data(self) -> Mapping[str, Any]:
        """Return the validated canonical linkage payload."""
        return self.linkage_record.data


@dataclass(frozen=True)
class Paper01ConsumerView:
    """Exactly seven canonical Paper 1 model-family records for one dataset/seed."""

    dataset: str
    seed: int
    records: tuple[Paper01ConsumerRecord, ...]

    def by_family(self) -> dict[str, Paper01ConsumerRecord]:
        return {
            record.family: record
            for record in self.records
        }

    def get(self, family: str) -> Paper01ConsumerRecord:
        try:
            return self.by_family()[family]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Paper 1 family for "
                f"dataset={self.dataset!r}, seed={self.seed}: {family!r}"
            ) from exc

    def __iter__(self) -> Iterable[Paper01ConsumerRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


def _normalize_dataset(dataset: str) -> str:
    value = str(dataset).strip().lower()
    if value not in EXPECTED_DATASETS:
        raise Paper01ConsumerViewError(
            f"Unsupported Paper 1 dataset {dataset!r}; "
            f"expected one of {sorted(EXPECTED_DATASETS)}"
        )
    return value


def _normalize_seed(seed: int) -> int:
    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise Paper01ConsumerViewError(
            f"Paper 1 seed must be an integer; got {seed!r}"
        ) from exc

    if value not in EXPECTED_SEEDS:
        raise Paper01ConsumerViewError(
            f"Unsupported Paper 1 seed {value!r}; "
            f"expected one of {sorted(EXPECTED_SEEDS)}"
        )
    return value


def _project_record(
    linkage_record: Paper01LinkageRecord,
) -> Paper01ConsumerRecord:
    data = linkage_record.data

    experiment = data.get("experiment")
    if not isinstance(experiment, Mapping):
        raise Paper01ConsumerViewError(
            f"{linkage_record.path}: experiment must be a mapping"
        )

    accepted_evaluation = data.get("accepted_evaluation")
    if not isinstance(accepted_evaluation, Mapping):
        raise Paper01ConsumerViewError(
            f"{linkage_record.path}: accepted_evaluation must be a mapping"
        )

    accepted_summary = accepted_evaluation.get("accepted_summary")
    if not isinstance(accepted_summary, Mapping):
        raise Paper01ConsumerViewError(
            f"{linkage_record.path}: accepted_summary must be a mapping"
        )

    summary_path_text = accepted_summary.get("path")
    if not isinstance(summary_path_text, str) or not summary_path_text:
        raise Paper01ConsumerViewError(
            f"{linkage_record.path}: accepted summary path is missing"
        )

    accepted_summary_path = Path(summary_path_text)

    if accepted_summary_path.name == "latest.json":
        raise Paper01ConsumerViewError(
            f"{linkage_record.path}: latest.json cannot be authoritative"
        )

    budget_per_class = experiment.get("budget_per_class")
    try:
        budget_per_class_int = int(budget_per_class)
    except (TypeError, ValueError) as exc:
        raise Paper01ConsumerViewError(
            f"{linkage_record.path}: invalid budget_per_class="
            f"{budget_per_class!r}"
        ) from exc

    seed = experiment.get("seed")
    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as exc:
        raise Paper01ConsumerViewError(
            f"{linkage_record.path}: invalid seed={seed!r}"
        ) from exc

    return Paper01ConsumerRecord(
        experiment_id=str(experiment["experiment_id"]),
        dataset=str(experiment["dataset"]),
        family=str(experiment["family"]),
        seed=seed_int,
        budget_per_class=budget_per_class_int,
        historical_run_id=str(experiment["historical_run_id"]),
        accepted_summary_path=accepted_summary_path,
        linkage_record=linkage_record,
    )


def project_paper01_consumer_view(
    study: Paper01LinkageStudy,
    *,
    dataset: str,
    seed: int,
) -> Paper01ConsumerView:
    """
    Project one validated canonical study into one explicit dataset/seed view.

    The result must contain exactly one record for each of the seven expected
    Paper 1 model families.
    """
    normalized_dataset = _normalize_dataset(dataset)
    normalized_seed = _normalize_seed(seed)

    projected: list[Paper01ConsumerRecord] = []

    for linkage_record in study.records:
        experiment = linkage_record.data["experiment"]

        if experiment["dataset"] != normalized_dataset:
            continue

        if int(experiment["seed"]) != normalized_seed:
            continue

        projected.append(_project_record(linkage_record))

    projected.sort(key=lambda record: record.family)

    if len(projected) != EXPECTED_RECORDS_PER_VIEW:
        raise Paper01ConsumerViewError(
            f"Paper 1 consumer view dataset={normalized_dataset!r}, "
            f"seed={normalized_seed} must contain exactly "
            f"{EXPECTED_RECORDS_PER_VIEW} records; found {len(projected)}"
        )

    families = [record.family for record in projected]
    family_set = set(families)

    if len(families) != len(family_set):
        raise Paper01ConsumerViewError(
            f"Paper 1 consumer view dataset={normalized_dataset!r}, "
            f"seed={normalized_seed} contains duplicate model families"
        )

    if family_set != EXPECTED_FAMILIES:
        missing = sorted(EXPECTED_FAMILIES - family_set)
        unexpected = sorted(family_set - EXPECTED_FAMILIES)
        raise Paper01ConsumerViewError(
            f"Paper 1 consumer view family-set mismatch for "
            f"dataset={normalized_dataset!r}, seed={normalized_seed}: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for record in projected:
        if record.dataset != normalized_dataset:
            raise Paper01ConsumerViewError(
                f"{record.experiment_id}: dataset projection mismatch"
            )

        if record.seed != normalized_seed:
            raise Paper01ConsumerViewError(
                f"{record.experiment_id}: seed projection mismatch"
            )

    return Paper01ConsumerView(
        dataset=normalized_dataset,
        seed=normalized_seed,
        records=tuple(projected),
    )


def load_paper01_consumer_view(
    repo_root: Path | str,
    *,
    dataset: str,
    seed: int,
) -> Paper01ConsumerView:
    """
    Load canonical Paper 1 linkage and return one explicit dataset/seed view.

    This function:
    - requires dataset and seed;
    - fully validates canonical linkage through the existing loader;
    - returns exactly seven model-family records;
    - exposes the accepted timestamped summary path selected by canonical
      authority;
    - never opens the historical summary file itself;
    - never treats latest.json as authoritative.
    """
    study = load_paper01_linkage_study(repo_root)

    return project_paper01_consumer_view(
        study,
        dataset=dataset,
        seed=seed,
    )
