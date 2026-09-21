#!/usr/bin/env python3
"""
M6.4.1 — Paper 1 historical execution-evidence inventory (v2).

READ-ONLY HISTORICAL EVIDENCE AUDITOR.

This tool inventories the evidence shape of the 42 Paper 1 scientific
experiments. It does NOT create canonical TrustForge linkage records and
must never modify historical Paper 1 artifacts.

Permanent distinctions:

    SCIENTIFIC EXPERIMENT IDENTITY != HISTORICAL EXECUTION IDENTITY
    ARTIFACT PRODUCER != ACCEPTED EVALUATOR != AUTHORITATIVE RESULT ROW
    MISSING SEED MANIFEST != MISSING EXPERIMENT
    MULTIPLE COMPLETED HISTORICAL EXECUTIONS != ONE AUTHORITATIVE EXECUTION
    OBSERVED EVIDENCE != PROVEN LINEAGE

v2 design change
----------------
Historical Slurm logs are associated to experiments ONLY from explicit
execution identity fields found in the log itself, principally:

    [map] MODEL=<family> SEED=<seed>
    [cfg_effective] model=<family> seed=<seed>
    [slurm] job_id=<id> task_id=<id> host=<host>
    [cfg_task] sha1 <sha1>
    [cfg_task] git <commit>

Summary JSON files are discovered from the model's summaries directory and
are linked to explicitly identified executions through config SHA1 whenever
possible.

No fuzzy "family name + seed appears somewhere in free text" relationship
is used for canonical inventory association.

The only files written by this program are NEW audit reports under
--output-dir, which must be inside the repository and outside every
historical evidence root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

AUDIT_VERSION = "M6.4.1-v3.2"

EXPECTED_DATASETS = ("ustc_tfc2016", "cicmaldroid2020")
EXPECTED_FAMILIES = (
    "gan",
    "vae",
    "diffusion",
    "autoregressive",
    "restrictedboltzmann",
    "gaussianmixture",
    "maskedautoflow",
)
EXPECTED_SEEDS = (42, 43, 44)

FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "gan": ("gan",),
    "vae": ("vae",),
    "diffusion": ("diffusion",),
    "autoregressive": ("autoregressive", "ar"),
    "restrictedboltzmann": (
        "restrictedboltzmann",
        "restricted_boltzmann",
        "restricted-boltzmann",
        "rbm",
    ),
    "gaussianmixture": (
        "gaussianmixture",
        "gaussian_mixture",
        "gaussian-mixture",
        "gmm",
    ),
    "maskedautoflow": (
        "maskedautoflow",
        "masked_auto_flow",
        "masked-auto-flow",
        "maf",
    ),
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "run_id": ("run_id", "run", "id"),
    "seed": ("seed", "random_seed"),
    "model": (
        "model",
        "family",
        "model_family",
        "generator",
        "generator_family",
        "method",
    ),
}

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "kid": ("kid", "kid_mean", "kernel_inception_distance"),
    "ms_ssim": ("ms_ssim", "msssim", "ms-ssim", "ms ssim"),
    "balanced_accuracy": (
        "balanced_accuracy",
        "balanced_acc",
        "bal_acc",
        "utility_balanced_accuracy",
        "util_balanced_accuracy",
    ),
    "macro_f1": (
        "macro_f1",
        "f1_macro",
        "macro-f1",
        "utility_macro_f1",
        "util_macro_f1",
    ),
    "macro_auprc": (
        "macro_auprc",
        "macro_ap",
        "auprc_macro",
        "utility_macro_auprc",
        "util_macro_auprc",
    ),
    "generative_precision": ("generative_precision", "gen_precision", "precision"),
    "generative_recall": ("generative_recall", "gen_recall", "recall"),
}

SUMMARY_SHA_KEYS = (
    "config_sha1",
    "effective_config_sha1",
    "config_hash",
    "configuration_sha1",
)
SUMMARY_GIT_KEYS = ("git_commit", "git_sha", "commit_sha", "commit")
SUMMARY_JOB_KEYS = ("job_id", "slurm_job_id", "slurm_job")
SUMMARY_TASK_KEYS = ("task_id", "array_task_id", "slurm_array_task_id")
SUMMARY_HOST_KEYS = ("host", "hostname", "node", "nodename")

CHECKPOINT_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".pkl",
    ".pickle",
    ".joblib",
    ".h5",
    ".keras",
    ".safetensors",
)
SYNTHETIC_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz", ".parquet")
LOG_SUFFIXES = (".out", ".err", ".log", ".txt")

MAP_RE = re.compile(r"\[map\]\s+MODEL=([A-Za-z0-9_.-]+)\s+SEED=([0-9]+)", re.IGNORECASE)
CFG_EFFECTIVE_RE = re.compile(r"\[cfg_effective\]\s+model=([A-Za-z0-9_.-]+)\s+seed=([0-9]+)", re.IGNORECASE)
SLURM_RE = re.compile(r"\[slurm\]\s+job_id=([0-9]+)\s+task_id=([0-9]+)\s+host=([A-Za-z0-9_.-]+)", re.IGNORECASE)
CFG_SHA_RE = re.compile(r"\[cfg_task\]\s+sha1\s+([0-9a-fA-F]{40})", re.IGNORECASE)
CFG_GIT_RE = re.compile(r"\[cfg_task\]\s+git\s+([0-9a-fA-F]{7,40})", re.IGNORECASE)
CFG_PATH_RE = re.compile(r"\[cfg\]\s+CFG=([^\s]+)", re.IGNORECASE)
ARTS_RE = re.compile(r"(?:\[cfg\]\s+ARTS=|\[check\]\s+cfg\.paths\.artifacts\s*=\s*)([^\s]+)", re.IGNORECASE)
TOGGLES_RE = re.compile(r"\[toggles\]\s+DO_TRAIN=([01])\s+DO_SYNTH=([01])\s+DO_EVAL=([01])", re.IGNORECASE)
SUMMARY_PATH_RE = re.compile(r"(?:appended summary|Saved evaluation summary)\s*[→>-]*\s*([^\s]+\.json)", re.IGNORECASE)
STAGE_LINE_RE = re.compile(r"\[stage\]\s+([A-Za-z0-9_.-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ExperimentKey:
    dataset: str
    family: str
    seed: int
    budget: int = 2000

    @property
    def experiment_id(self) -> str:
        return f"{self.dataset}_{self.family}_b{self.budget}_seed{self.seed}"


@dataclass
class ScoreEvidence:
    source: str
    row_number: int
    run_id: str
    metrics: dict[str, float | None]


@dataclass
class ManifestEvidence:
    path: str
    kind: str
    mtime_utc: str
    size_bytes: int
    declared_paths: int | None
    embedded_seed: int | None
    embedded_model: str | None
    embedded_git_commit: str | None
    embedded_config_sha1: str | None
    parse_status: str


@dataclass
class SummaryEvidence:
    path: str
    mtime_utc: str
    config_sha1: str | None
    git_commit: str | None
    job_id: str | None
    task_id: str | None
    host: str | None
    metrics: dict[str, float | None]
    metric_match_status: str
    metric_match_count: int
    metric_compared_count: int
    metric_mismatches: list[str]
    linked_log_paths: list[str] = field(default_factory=list)


@dataclass
class LogEvidence:
    path: str
    mtime_utc: str
    dataset: str | None
    family: str
    seed: int
    job_id: str | None
    task_id: str | None
    host: str | None
    config_path: str | None
    artifact_root: str | None
    config_sha1: str | None
    git_commit: str | None
    do_train: bool | None
    do_synth: bool | None
    do_eval: bool | None
    stages: list[str]
    summary_paths_reported: list[str]
    identity_source: str


@dataclass
class CheckpointEvidence:
    path: str
    mtime_utc: str
    size_bytes: int


@dataclass
class SyntheticEvidence:
    layout: str
    file_count: int
    layout_counts: dict[str, int]
    earliest_mtime_utc: str | None
    latest_mtime_utc: str | None
    example_paths: list[str]


@dataclass
class ExperimentInventory:
    experiment_id: str
    dataset: str
    family: str
    seed: int
    budget: int
    historical_run_id: str
    accepted_score_row: ScoreEvidence
    seed_manifest_present: bool
    shared_manifest_present: bool
    manifests: list[ManifestEvidence]
    synthetic: SyntheticEvidence
    checkpoints: list[CheckpointEvidence]
    summaries: list[SummaryEvidence]
    latest_aliases: list[SummaryEvidence]
    slurm_logs: list[LogEvidence]
    multiple_execution_waves: bool
    observed_job_ids: list[str]
    observed_effective_config_sha1s: list[str]
    observed_git_commits: list[str]
    exact_accepted_summary_count: int
    sha1_bound_summary_log_count: int
    model_specific_layout_anomalies: list[str]
    notes: list[str]


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip()
        return int(float(text)) if text else None
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        number = float(text)
        return None if math.isnan(number) else number
    except (TypeError, ValueError):
        return None


def read_text_limited(path: Path, limit: int = 16_000_000) -> str:
    with path.open("rb") as handle:
        raw = handle.read(limit)
    return raw.decode("utf-8", errors="replace")


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}

    def visit(value: Any, key: str) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                next_key = f"{key}.{child_key}" if key else str(child_key)
                visit(child_value, next_key)
        elif isinstance(value, list):
            for i, child_value in enumerate(value):
                visit(child_value, f"{key}[{i}]")
        else:
            out[key] = value

    visit(obj, prefix)
    return out


def leaf_lookup(flat: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    targets = {norm(alias) for alias in aliases}
    for key, value in flat.items():
        leaf = key.split(".")[-1]
        leaf = re.sub(r"\[[0-9]+\]$", "", leaf)
        if norm(leaf) in targets:
            return value
    return None


def first_sha1(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"\b[0-9a-fA-F]{40}\b", str(value))
    return match.group(0).lower() if match else None


def first_git_sha(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"\b[0-9a-fA-F]{7,40}\b", str(value))
    return match.group(0).lower() if match else None


def all_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def resolve_column(fieldnames: Sequence[str], aliases: Iterable[str]) -> str | None:
    mapping = {norm(name): name for name in fieldnames}
    for alias in aliases:
        if norm(alias) in mapping:
            return mapping[norm(alias)]
    return None


def canonical_family(raw: str) -> str | None:
    raw_norm = norm(raw)
    for family, aliases in FAMILY_ALIASES.items():
        if any(norm(alias) == raw_norm for alias in aliases):
            return family
    return None


def infer_family_from_run_id(run_id: str) -> str | None:
    raw = re.sub(r"[_-]?s(?:eed)?\d+$", "", run_id, flags=re.IGNORECASE)
    return canonical_family(raw)


def infer_seed_from_run_id(run_id: str) -> int | None:
    match = re.search(r"(?:^|[_-])s(?:eed)?(\d+)$", run_id, re.IGNORECASE)
    return int(match.group(1)) if match else None


def family_dir(artifact_root: Path, family: str) -> Path | None:
    aliases = {norm(x) for x in FAMILY_ALIASES[family]}
    for child in artifact_root.iterdir():
        if child.is_dir() and norm(child.name) in aliases:
            return child
    return None


def load_score_table(path: Path, dataset: str, repo_root: Path) -> dict[ExperimentKey, ScoreEvidence]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"No CSV header in {path}")

        fields = list(reader.fieldnames)
        run_col = resolve_column(fields, COLUMN_ALIASES["run_id"])
        seed_col = resolve_column(fields, COLUMN_ALIASES["seed"])
        model_col = resolve_column(fields, COLUMN_ALIASES["model"])

        if run_col is None:
            raise RuntimeError(f"Could not identify run_id column in {path}")

        metric_columns = {
            metric: resolve_column(fields, aliases)
            for metric, aliases in METRIC_ALIASES.items()
        }

        result: dict[ExperimentKey, ScoreEvidence] = {}

        for row_number, row in enumerate(reader, start=2):
            run_id = str(row.get(run_col, "")).strip()
            if not run_id:
                continue

            family = None
            if model_col is not None:
                family = canonical_family(str(row.get(model_col, "")))
            if family is None:
                family = infer_family_from_run_id(run_id)

            seed = parse_int(row.get(seed_col)) if seed_col is not None else None
            if seed is None:
                seed = infer_seed_from_run_id(run_id)

            if family is None or seed is None:
                raise RuntimeError(
                    f"Could not resolve family/seed from {path}:{row_number} "
                    f"(run_id={run_id!r})"
                )

            key = ExperimentKey(dataset=dataset, family=family, seed=seed)
            metrics = {
                metric: parse_float(row.get(column)) if column is not None else None
                for metric, column in metric_columns.items()
            }

            if key in result:
                raise RuntimeError(f"Duplicate scientific experiment in {path}: {key}")

            result[key] = ScoreEvidence(
                source=rel(path, repo_root),
                row_number=row_number,
                run_id=run_id,
                metrics=metrics,
            )

    return result


def validate_expected_universe(scores: Mapping[ExperimentKey, ScoreEvidence]) -> None:
    expected = {
        ExperimentKey(dataset, family, seed)
        for dataset in EXPECTED_DATASETS
        for family in EXPECTED_FAMILIES
        for seed in EXPECTED_SEEDS
    }
    actual = set(scores)

    missing = sorted(expected - actual, key=lambda key: (key.dataset, key.family, key.seed))
    unexpected = sorted(actual - expected, key=lambda key: (key.dataset, key.family, key.seed))

    if missing or unexpected:
        lines = ["Authoritative score universe is not exactly the expected 42."]
        if missing:
            lines.append("Missing:")
            lines.extend(f"  {key.experiment_id}" for key in missing)
        if unexpected:
            lines.append("Unexpected:")
            lines.extend(f"  {key.experiment_id}" for key in unexpected)
        raise RuntimeError("\n".join(lines))


def parse_manifest(path: Path, kind: str, repo_root: Path) -> ManifestEvidence:
    declared_paths = None
    embedded_seed = None
    embedded_model = None
    embedded_git_commit = None
    embedded_config_sha1 = None
    parse_status = "parsed"

    try:
        if path.suffix.lower() == ".json":
            obj = json.loads(read_text_limited(path))
            flat = flatten(obj)

            if isinstance(obj, list):
                declared_paths = len(obj)
            elif isinstance(obj, Mapping):
                for key in ("paths", "files", "samples", "items"):
                    value = obj.get(key)
                    if isinstance(value, list):
                        declared_paths = len(value)
                        break

                if declared_paths is None:
                    declared_paths = parse_int(
                        leaf_lookup(flat, ("count", "file_count", "n_files", "n_samples"))
                    )

            embedded_seed = parse_int(leaf_lookup(flat, ("seed", "random_seed")))
            model_value = leaf_lookup(flat, ("model", "family", "generator", "method"))
            embedded_model = str(model_value) if model_value is not None else None
            embedded_git_commit = first_git_sha(leaf_lookup(flat, SUMMARY_GIT_KEYS))
            embedded_config_sha1 = first_sha1(leaf_lookup(flat, SUMMARY_SHA_KEYS))
        else:
            parse_status = "recognized_non_json_manifest"

    except Exception as exc:
        parse_status = f"parse_error:{type(exc).__name__}"

    stat = path.stat()
    return ManifestEvidence(
        path=rel(path, repo_root),
        kind=kind,
        mtime_utc=utc_iso(stat.st_mtime),
        size_bytes=stat.st_size,
        declared_paths=declared_paths,
        embedded_seed=embedded_seed,
        embedded_model=embedded_model,
        embedded_git_commit=embedded_git_commit,
        embedded_config_sha1=embedded_config_sha1,
        parse_status=parse_status,
    )


def is_seed_dir_name(name: str, seed: int) -> bool:
    value = norm(name)
    return value in {
        norm(f"seed{seed}"),
        norm(f"seed_{seed}"),
        norm(f"s{seed}"),
        norm(str(seed)),
    }


def discover_manifests(model_root: Path, seed: int, repo_root: Path) -> tuple[list[ManifestEvidence], bool, bool]:
    manifests: list[ManifestEvidence] = []
    seed_present = False
    shared_present = False

    synthetic_root = model_root / "synthetic"
    if not synthetic_root.exists():
        return manifests, seed_present, shared_present

    for path in synthetic_root.rglob("manifest.json"):
        try:
            relative = path.relative_to(synthetic_root)
        except ValueError:
            continue

        parents = relative.parts[:-1]

        if any(is_seed_dir_name(part, seed) for part in parents):
            kind = "seed_specific"
            seed_present = True
        elif path.parent == synthetic_root:
            kind = "shared_root_manifest"
            shared_present = True
        else:
            kind = "other_manifest"

        manifests.append(parse_manifest(path, kind, repo_root))

    return manifests, seed_present, shared_present


def classify_seed_path(path: Path, synthetic_root: Path, seed: int) -> tuple[bool, str | None]:
    try:
        relative = path.relative_to(synthetic_root)
    except ValueError:
        return False, None

    dirs = relative.parts[:-1]
    if not dirs:
        return False, None

    if is_seed_dir_name(dirs[0], seed):
        return True, "seed_directory"

    if len(dirs) >= 2 and is_seed_dir_name(dirs[1], seed):
        return True, "class_then_seed"

    return False, None


def discover_synthetic(model_root: Path, seed: int, repo_root: Path) -> SyntheticEvidence:
    synthetic_root = model_root / "synthetic"
    if not synthetic_root.exists():
        return SyntheticEvidence("absent", 0, {}, None, None, [])

    candidates: list[Path] = []
    layout_counts: dict[str, int] = {}

    for path in all_files(synthetic_root):
        if path.name == "manifest.json":
            continue
        if path.suffix.lower() not in SYNTHETIC_SUFFIXES:
            continue

        belongs, layout = classify_seed_path(path, synthetic_root, seed)
        if belongs:
            candidates.append(path)
            if layout:
                layout_counts[layout] = layout_counts.get(layout, 0) + 1

    layouts = set(layout_counts)

    if not candidates:
        layout_name = "unresolved_or_absent"
    elif len(layouts) == 1:
        layout_name = next(iter(layouts))
    else:
        layout_name = "mixed:" + ",".join(sorted(layouts))

    mtimes = [path.stat().st_mtime for path in candidates]

    return SyntheticEvidence(
        layout=layout_name,
        file_count=len(candidates),
        layout_counts=dict(sorted(layout_counts.items())),
        earliest_mtime_utc=utc_iso(min(mtimes)) if mtimes else None,
        latest_mtime_utc=utc_iso(max(mtimes)) if mtimes else None,
        example_paths=[rel(path, repo_root) for path in sorted(candidates)[:5]],
    )


def discover_checkpoints(model_root: Path, seed: int, repo_root: Path) -> list[CheckpointEvidence]:
    result: list[CheckpointEvidence] = []
    checkpoint_root = model_root / "checkpoints"
    if not checkpoint_root.exists():
        return result

    for path in all_files(checkpoint_root):
        if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
            continue

        try:
            relative = path.relative_to(checkpoint_root)
        except ValueError:
            continue

        dirs = relative.parts[:-1]
        if not dirs or not any(is_seed_dir_name(part, seed) for part in dirs):
            continue

        stat = path.stat()
        result.append(
            CheckpointEvidence(
                path=rel(path, repo_root),
                mtime_utc=utc_iso(stat.st_mtime),
                size_bytes=stat.st_size,
            )
        )

    return sorted(result, key=lambda item: item.path)


def infer_dataset_from_artifact_root(artifact_path: str | None, artifact_roots: Mapping[str, Path]) -> str | None:
    if not artifact_path:
        return None

    observed_text = str(Path(artifact_path).expanduser())
    matches = [
        dataset
        for dataset, root in artifact_roots.items()
        if observed_text == str(root) or observed_text.startswith(str(root) + "/")
    ]
    return matches[0] if len(matches) == 1 else None


def parse_log_identity(path: Path, text: str, artifact_roots: Mapping[str, Path], repo_root: Path) -> LogEvidence | None:
    identity_source = None
    family = None
    seed = None

    map_match = MAP_RE.search(text)
    if map_match:
        family = canonical_family(map_match.group(1))
        seed = int(map_match.group(2))
        identity_source = "map"

    if family is None or seed is None:
        cfg_match = CFG_EFFECTIVE_RE.search(text)
        if cfg_match:
            family = canonical_family(cfg_match.group(1))
            seed = int(cfg_match.group(2))
            identity_source = "cfg_effective"

    if family is None or seed is None:
        return None

    slurm_match = SLURM_RE.search(text)
    config_sha_match = CFG_SHA_RE.search(text)
    git_match = CFG_GIT_RE.search(text)
    cfg_path_match = CFG_PATH_RE.search(text)
    toggles_match = TOGGLES_RE.search(text)

    arts_values = [match.group(1) for match in ARTS_RE.finditer(text)]
    artifact_root = arts_values[0] if arts_values else None
    dataset = infer_dataset_from_artifact_root(artifact_root, artifact_roots)

    stages = sorted({match.group(1).lower() for match in STAGE_LINE_RE.finditer(text)})
    summary_paths = sorted({match.group(1) for match in SUMMARY_PATH_RE.finditer(text)})

    do_train = do_synth = do_eval = None
    if toggles_match:
        do_train = toggles_match.group(1) == "1"
        do_synth = toggles_match.group(2) == "1"
        do_eval = toggles_match.group(3) == "1"

    return LogEvidence(
        path=rel(path, repo_root),
        mtime_utc=utc_iso(path.stat().st_mtime),
        dataset=dataset,
        family=family,
        seed=seed,
        job_id=slurm_match.group(1) if slurm_match else None,
        task_id=slurm_match.group(2) if slurm_match else None,
        host=slurm_match.group(3) if slurm_match else None,
        config_path=cfg_path_match.group(1) if cfg_path_match else None,
        artifact_root=artifact_root,
        config_sha1=config_sha_match.group(1).lower() if config_sha_match else None,
        git_commit=git_match.group(1).lower() if git_match else None,
        do_train=do_train,
        do_synth=do_synth,
        do_eval=do_eval,
        stages=stages,
        summary_paths_reported=summary_paths,
        identity_source=identity_source or "unknown",
    )


def discover_all_logs(log_roots: Sequence[Path], artifact_roots: Mapping[str, Path], repo_root: Path) -> list[LogEvidence]:
    result: list[LogEvidence] = []
    seen: set[Path] = set()

    for root in log_roots:
        if not root.exists():
            continue

        for path in all_files(root):
            resolved = path.resolve()
            if resolved in seen or path.suffix.lower() not in LOG_SUFFIXES:
                continue

            try:
                text = read_text_limited(path)
            except OSError:
                continue

            parsed = parse_log_identity(path, text, artifact_roots, repo_root)
            if parsed is None:
                continue

            seen.add(resolved)
            result.append(parsed)

    return sorted(result, key=lambda item: (item.mtime_utc, item.path))


def exact_flat_value(flat: Mapping[str, Any], *keys: str) -> Any:
    """
    Return the first exact flattened-path match.

    Historical Paper 1 summaries can contain both utility_real_only and
    utility_real_plus_synth metrics with identical leaf names. Leaf-only
    lookup is therefore unsafe for accepted-result comparison.
    """
    for key in keys:
        if key in flat:
            return flat[key]
    return None


def summary_metrics(flat: Mapping[str, Any]) -> dict[str, float | None]:
    """
    Extract the exact metric semantics used by phase1_scores_dedup.csv.

    Accepted utility rows represent REAL + SYNTHETIC downstream utility,
    not the real-only baseline. Prefer metrics.downstream.* fields when
    present, then utility_real_plus_synth.*. KID/MS-SSIM come from the
    generative metrics.
    """
    return {
        "kid": parse_float(
            exact_flat_value(flat, "metrics.kid", "generative.kid")
        ),
        "ms_ssim": parse_float(
            exact_flat_value(flat, "metrics.ms_ssim", "generative.ms_ssim")
        ),
        "balanced_accuracy": parse_float(
            exact_flat_value(
                flat,
                "metrics.downstream.balanced_acc",
                "metrics.downstream.bal_acc",
                "utility_real_plus_synth.balanced_acc",
                "utility_real_plus_synth.bal_acc",
            )
        ),
        "macro_f1": parse_float(
            exact_flat_value(
                flat,
                "metrics.downstream.macro_f1",
                "utility_real_plus_synth.macro_f1",
            )
        ),
        "macro_auprc": parse_float(
            exact_flat_value(
                flat,
                "metrics.downstream.macro_auprc",
                "utility_real_plus_synth.macro_auprc",
            )
        ),
        "generative_precision": parse_float(
            exact_flat_value(
                flat,
                "metrics.gen_precision",
                "metrics.downstream.precision",
                "utility_real_plus_synth.macro_precision",
            )
        ),
        "generative_recall": parse_float(
            exact_flat_value(
                flat,
                "metrics.gen_recall",
                "metrics.downstream.recall",
                "utility_real_plus_synth.macro_recall",
            )
        ),
    }


def score_summary_metrics(
    accepted: Mapping[str, float | None],
    observed: Mapping[str, float | None],
    abs_tol: float,
    rel_tol: float,
) -> tuple[str, int, int, list[str]]:
    compared = 0
    matched = 0
    mismatches: list[str] = []

    for metric in METRIC_ALIASES:
        expected = accepted.get(metric)
        actual = observed.get(metric)

        if expected is None or actual is None:
            continue

        compared += 1
        if math.isclose(expected, actual, abs_tol=abs_tol, rel_tol=rel_tol):
            matched += 1
        else:
            mismatches.append(metric)

    if compared == 0:
        status = "unknown"
    elif mismatches:
        status = "conflicting"
    elif matched == compared == len(METRIC_ALIASES):
        status = "exact_all_accepted_metrics"
    elif matched == compared:
        status = "exact_available_metrics_only"
    else:
        status = "partial"

    return status, matched, compared, mismatches


def parse_summary_file(
    path: Path,
    accepted_metrics: Mapping[str, float | None],
    repo_root: Path,
    abs_tol: float,
    rel_tol: float,
) -> SummaryEvidence | None:
    try:
        obj = json.loads(read_text_limited(path))
    except Exception:
        return None

    if not isinstance(obj, (Mapping, list)):
        return None

    flat = flatten(obj)
    metrics = summary_metrics(flat)
    if not any(value is not None for value in metrics.values()):
        return None

    config_sha1 = first_sha1(leaf_lookup(flat, SUMMARY_SHA_KEYS))
    git_commit = first_git_sha(leaf_lookup(flat, SUMMARY_GIT_KEYS))
    job_value = leaf_lookup(flat, SUMMARY_JOB_KEYS)
    task_value = leaf_lookup(flat, SUMMARY_TASK_KEYS)
    host_value = leaf_lookup(flat, SUMMARY_HOST_KEYS)

    status, matched, compared, mismatches = score_summary_metrics(
        accepted_metrics,
        metrics,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

    return SummaryEvidence(
        path=rel(path, repo_root),
        mtime_utc=utc_iso(path.stat().st_mtime),
        config_sha1=config_sha1,
        git_commit=git_commit,
        job_id=str(job_value) if job_value is not None else None,
        task_id=str(task_value) if task_value is not None else None,
        host=str(host_value) if host_value is not None else None,
        metrics=metrics,
        metric_match_status=status,
        metric_match_count=matched,
        metric_compared_count=compared,
        metric_mismatches=mismatches,
    )


def discover_summaries_for_experiment(
    model_root: Path,
    accepted_metrics: Mapping[str, float | None],
    experiment_logs: Sequence[LogEvidence],
    repo_root: Path,
    abs_tol: float,
    rel_tol: float,
) -> tuple[list[SummaryEvidence], list[SummaryEvidence]]:
    """
    Discover explicitly linked historical summaries.

    Timestamped summary JSON files are historical evidence.

    latest.json is a compatibility/convenience alias and MUST NOT be counted
    as an independent historical evaluation. It is still inventoried
    separately so its presence remains observable.

    Permanent rule:
        latest.json IS NOT AUTHORITATIVE EVIDENCE.
    """
    summaries_root = model_root / "summaries"
    if not summaries_root.exists():
        return [], []

    log_sha_to_paths: dict[str, list[str]] = {}
    explicit_summary_paths: set[str] = set()

    for log in experiment_logs:
        if log.config_sha1:
            log_sha_to_paths.setdefault(log.config_sha1, []).append(log.path)
        explicit_summary_paths.update(log.summary_paths_reported)

    historical: list[SummaryEvidence] = []
    latest_aliases: list[SummaryEvidence] = []

    for path in all_files(summaries_root):
        if path.suffix.lower() != ".json":
            continue

        parsed = parse_summary_file(
            path,
            accepted_metrics,
            repo_root,
            abs_tol,
            rel_tol,
        )
        if parsed is None:
            continue

        linked_logs: set[str] = set()

        if parsed.config_sha1 and parsed.config_sha1 in log_sha_to_paths:
            linked_logs.update(log_sha_to_paths[parsed.config_sha1])

        absolute_text = str(path.resolve())
        if absolute_text in explicit_summary_paths:
            linked_logs.update(
                log.path
                for log in experiment_logs
                if absolute_text in log.summary_paths_reported
            )

        if not linked_logs:
            continue

        parsed.linked_log_paths = sorted(linked_logs)

        if path.name == "latest.json":
            latest_aliases.append(parsed)
        else:
            historical.append(parsed)

    historical.sort(key=lambda item: (item.mtime_utc, item.path))
    latest_aliases.sort(key=lambda item: (item.mtime_utc, item.path))
    return historical, latest_aliases

def derive_layout_anomalies(
    family: str,
    synthetic: SyntheticEvidence,
    seed_manifest_present: bool,
    shared_manifest_present: bool,
) -> list[str]:
    anomalies: list[str] = []

    if synthetic.file_count > 0 and not seed_manifest_present:
        anomalies.append("synthetic_files_present_without_seed_specific_manifest")
    if shared_manifest_present:
        anomalies.append("shared_root_manifest_present")
    if synthetic.layout in {"class_then_seed", "unresolved_or_absent"}:
        anomalies.append(f"artifact_layout:{synthetic.layout}")
    if len(synthetic.layout_counts) > 1:
        anomalies.append("multiple_historical_artifact_layouts_coexist")

    if family == "gaussianmixture" and "class_then_seed" in synthetic.layout_counts:
        anomalies.append("historical_gmm_class_then_seed_layout")

    return anomalies


def build_inventory(
    scores: Mapping[ExperimentKey, ScoreEvidence],
    artifact_roots: Mapping[str, Path],
    all_logs: Sequence[LogEvidence],
    repo_root: Path,
    abs_tol: float,
    rel_tol: float,
) -> list[ExperimentInventory]:
    inventories: list[ExperimentInventory] = []

    for key in sorted(scores, key=lambda k: (k.dataset, k.family, k.seed)):
        score = scores[key]
        root = artifact_roots[key.dataset]
        model_root = family_dir(root, key.family)
        notes: list[str] = []

        experiment_logs = [
            log
            for log in all_logs
            if log.dataset == key.dataset and log.family == key.family and log.seed == key.seed
        ]

        dataset_unknown_logs = [
            log
            for log in all_logs
            if log.dataset is None and log.family == key.family and log.seed == key.seed
        ]
        if dataset_unknown_logs:
            notes.append(f"unassigned_logs_with_matching_family_seed:{len(dataset_unknown_logs)}")

        if model_root is None:
            manifests: list[ManifestEvidence] = []
            seed_manifest_present = False
            shared_manifest_present = False
            synthetic = SyntheticEvidence("model_directory_absent", 0, {}, None, None, [])
            checkpoints: list[CheckpointEvidence] = []
            summaries: list[SummaryEvidence] = []
            latest_aliases: list[SummaryEvidence] = []
            notes.append("model_artifact_directory_not_found")
        else:
            manifests, seed_manifest_present, shared_manifest_present = discover_manifests(
                model_root, key.seed, repo_root
            )
            synthetic = discover_synthetic(model_root, key.seed, repo_root)
            checkpoints = discover_checkpoints(model_root, key.seed, repo_root)
            summaries, latest_aliases = discover_summaries_for_experiment(
                model_root,
                score.metrics,
                experiment_logs,
                repo_root,
                abs_tol,
                rel_tol,
            )

        job_ids = sorted(
            {log.job_id for log in experiment_logs if log.job_id is not None},
            key=lambda value: int(value) if value.isdigit() else value,
        )

        config_sha1s = sorted(
            {log.config_sha1 for log in experiment_logs if log.config_sha1}
            | {summary.config_sha1 for summary in summaries if summary.config_sha1}
        )
        git_commits = sorted(
            {log.git_commit for log in experiment_logs if log.git_commit}
            | {summary.git_commit for summary in summaries if summary.git_commit}
        )

        exact_summaries = [
            summary
            for summary in summaries
            if summary.metric_match_status == "exact_all_accepted_metrics"
        ]
        sha1_bound_pairs = sum(
            len(summary.linked_log_paths)
            for summary in summaries
            if summary.config_sha1
        )

        anomalies = derive_layout_anomalies(
            key.family,
            synthetic,
            seed_manifest_present,
            shared_manifest_present,
        )

        if len(job_ids) > 1:
            notes.append("multiple_historical_execution_jobs_observed")
        if exact_summaries:
            notes.append(f"accepted_metric_exact_summary_candidates:{len(exact_summaries)}")
        if sha1_bound_pairs:
            notes.append(f"summary_log_sha1_bound_pairs:{sha1_bound_pairs}")
        if latest_aliases:
            notes.append(f"latest_json_aliases_observed:{len(latest_aliases)}")

        inventories.append(
            ExperimentInventory(
                experiment_id=key.experiment_id,
                dataset=key.dataset,
                family=key.family,
                seed=key.seed,
                budget=key.budget,
                historical_run_id=score.run_id,
                accepted_score_row=score,
                seed_manifest_present=seed_manifest_present,
                shared_manifest_present=shared_manifest_present,
                manifests=manifests,
                synthetic=synthetic,
                checkpoints=checkpoints,
                summaries=summaries,
                latest_aliases=latest_aliases,
                slurm_logs=experiment_logs,
                multiple_execution_waves=len(job_ids) > 1,
                observed_job_ids=job_ids,
                observed_effective_config_sha1s=config_sha1s,
                observed_git_commits=git_commits,
                exact_accepted_summary_count=len(exact_summaries),
                sha1_bound_summary_log_count=sha1_bound_pairs,
                model_specific_layout_anomalies=anomalies,
                notes=notes,
            )
        )

    return inventories


def json_report(path: Path, inventories: Sequence[ExperimentInventory], metadata: Mapping[str, Any]) -> None:
    payload = {
        "audit_version": AUDIT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_linkage_records_created": False,
        "historical_artifacts_modified": False,
        "metadata": dict(metadata),
        "principles": [
            "SCIENTIFIC EXPERIMENT IDENTITY != HISTORICAL EXECUTION IDENTITY",
            "ARTIFACT PRODUCER != ACCEPTED EVALUATOR != AUTHORITATIVE RESULT ROW",
            "MISSING SEED MANIFEST != MISSING EXPERIMENT",
            "MULTIPLE COMPLETED HISTORICAL EXECUTIONS != ONE AUTHORITATIVE EXECUTION",
            "OBSERVED EVIDENCE != PROVEN LINEAGE",
            "latest.json IS NOT AUTHORITATIVE EVIDENCE",
        ],
        "experiments": [asdict(item) for item in inventories],
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def csv_report(path: Path, inventories: Sequence[ExperimentInventory]) -> None:
    fields = [
        "experiment_id",
        "dataset",
        "family",
        "seed",
        "historical_run_id",
        "accepted_score_row_present",
        "seed_manifest_present",
        "shared_manifest_present",
        "synthetic_file_count",
        "synthetic_layout",
        "synthetic_layout_counts",
        "checkpoint_count",
        "summary_candidate_count",
        "latest_alias_count",
        "exact_accepted_summary_count",
        "slurm_log_count",
        "sha1_bound_summary_log_count",
        "observed_job_ids",
        "multiple_execution_waves",
        "observed_effective_config_sha1s",
        "observed_git_commits",
        "layout_anomalies",
        "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for item in inventories:
            writer.writerow(
                {
                    "experiment_id": item.experiment_id,
                    "dataset": item.dataset,
                    "family": item.family,
                    "seed": item.seed,
                    "historical_run_id": item.historical_run_id,
                    "accepted_score_row_present": True,
                    "seed_manifest_present": item.seed_manifest_present,
                    "shared_manifest_present": item.shared_manifest_present,
                    "synthetic_file_count": item.synthetic.file_count,
                    "synthetic_layout": item.synthetic.layout,
                    "synthetic_layout_counts": ";".join(f"{k}={v}" for k, v in item.synthetic.layout_counts.items()),
                    "checkpoint_count": len(item.checkpoints),
                    "summary_candidate_count": len(item.summaries),
                    "latest_alias_count": len(item.latest_aliases),
                    "exact_accepted_summary_count": item.exact_accepted_summary_count,
                    "slurm_log_count": len(item.slurm_logs),
                    "sha1_bound_summary_log_count": item.sha1_bound_summary_log_count,
                    "observed_job_ids": ";".join(item.observed_job_ids),
                    "multiple_execution_waves": item.multiple_execution_waves,
                    "observed_effective_config_sha1s": ";".join(item.observed_effective_config_sha1s),
                    "observed_git_commits": ";".join(item.observed_git_commits),
                    "layout_anomalies": ";".join(item.model_specific_layout_anomalies),
                    "notes": ";".join(item.notes),
                }
            )


def markdown_report(path: Path, inventories: Sequence[ExperimentInventory]) -> None:
    lines = [
        "# M6.4.1 — Paper 1 Historical Execution-Evidence Inventory",
        "",
        f"Audit version: `{AUDIT_VERSION}`",
        "",
        "> Audit-only. No canonical execution linkage records are created.",
        "",
        "> **ARTIFACT PRODUCER ≠ ACCEPTED EVALUATOR ≠ AUTHORITATIVE RESULT ROW**",
        "",
        "> **OBSERVED EVIDENCE ≠ PROVEN LINEAGE**",
        "",
        f"Experiments inventoried: **{len(inventories)}**",
        "",
        "## Coverage",
        "",
        "| Experiment | Run ID | Seed manifest | Shared root manifest | Synthetic | Layout | Layout counts | Checkpoints | Summaries | latest.json aliases | Exact accepted summary | Logs | SHA1-bound summary/log | Multi-wave |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for item in inventories:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.experiment_id}`",
                    f"`{item.historical_run_id}`",
                    "YES" if item.seed_manifest_present else "NO",
                    "YES" if item.shared_manifest_present else "NO",
                    str(item.synthetic.file_count),
                    f"`{item.synthetic.layout}`",
                    "`" + ";".join(f"{k}={v}" for k, v in item.synthetic.layout_counts.items()) + "`",
                    str(len(item.checkpoints)),
                    str(len(item.summaries)),
                    str(len(item.latest_aliases)),
                    str(item.exact_accepted_summary_count),
                    str(len(item.slurm_logs)),
                    str(item.sha1_bound_summary_log_count),
                    "YES" if item.multiple_execution_waves else "NO",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Experiment observations", ""])

    for item in inventories:
        if not item.model_specific_layout_anomalies and not item.notes and not item.slurm_logs:
            continue

        lines.append(f"### `{item.experiment_id}`")
        lines.append("")

        if item.observed_job_ids:
            lines.append("- observed jobs: " + ", ".join(f"`{job}`" for job in item.observed_job_ids))
        if item.observed_effective_config_sha1s:
            lines.append(
                "- config SHA1s: "
                + ", ".join(f"`{sha}`" for sha in item.observed_effective_config_sha1s)
            )
        if item.observed_git_commits:
            lines.append(
                "- Git commits: "
                + ", ".join(f"`{commit}`" for commit in item.observed_git_commits)
            )
        for anomaly in item.model_specific_layout_anomalies:
            lines.append(f"- layout/anomaly: `{anomaly}`")
        for note in item.notes:
            lines.append(f"- observation: `{note}`")
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This report inventories evidence candidates. It does not declare an artifact producer, accepted evaluator, or end-to-end execution unless a later governed M6.4 linkage stage establishes that relationship from evidence.",
            "",
        ]
    )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_output_location(output_dir: Path, repo_root: Path, historical_roots: Sequence[Path]) -> None:
    if not is_within(output_dir, repo_root):
        raise RuntimeError("--output-dir must be inside the TrustForge repository.")

    for root in historical_roots:
        if is_within(output_dir, root):
            raise RuntimeError(f"--output-dir must not be inside historical root: {root}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Read-only M6.4.1 inventory of Paper 1 historical execution "
            "and evidence candidates."
        )
    )

    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--ustc-scores", type=Path, required=True)
    p.add_argument("--cic-scores", type=Path, required=True)
    p.add_argument("--ustc-artifacts", type=Path, required=True)
    p.add_argument("--cic-artifacts", type=Path, required=True)
    p.add_argument(
        "--log-root",
        type=Path,
        action="append",
        required=True,
        help="Historical log search root. May be repeated.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metric-abs-tol", type=float, default=1e-12)
    p.add_argument("--metric-rel-tol", type=float, default=1e-9)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)

    repo_root = args.repo_root.expanduser().resolve()
    ustc_scores = args.ustc_scores.expanduser().resolve()
    cic_scores = args.cic_scores.expanduser().resolve()
    ustc_artifacts = args.ustc_artifacts.expanduser().resolve()
    cic_artifacts = args.cic_artifacts.expanduser().resolve()
    log_roots = [path.expanduser().resolve() for path in args.log_root]
    output_dir = args.output_dir.expanduser().resolve()

    required_files = (ustc_scores, cic_scores)
    required_dirs = (repo_root, ustc_artifacts, cic_artifacts, *log_roots)

    for path in required_files:
        if not path.is_file():
            raise RuntimeError(f"Required file not found: {path}")
    for path in required_dirs:
        if not path.is_dir():
            raise RuntimeError(f"Required directory not found: {path}")

    historical_roots = [ustc_artifacts, cic_artifacts, *log_roots]
    validate_output_location(output_dir, repo_root, historical_roots)

    artifact_roots = {
        "ustc_tfc2016": ustc_artifacts,
        "cicmaldroid2020": cic_artifacts,
    }

    print("[M6.4.1] Loading authoritative Paper 1 result tables...")

    scores: dict[ExperimentKey, ScoreEvidence] = {}
    scores.update(load_score_table(ustc_scores, "ustc_tfc2016", repo_root))
    scores.update(load_score_table(cic_scores, "cicmaldroid2020", repo_root))
    validate_expected_universe(scores)

    print(f"[M6.4.1] Authoritative experiments: {len(scores)}")
    print("[M6.4.1] Parsing historical execution logs by explicit identity...")

    all_logs = discover_all_logs(log_roots, artifact_roots, repo_root)

    print(f"[M6.4.1] Explicitly identified execution logs: {len(all_logs)}")
    print("[M6.4.1] Inventorying historical evidence read-only...")

    inventories = build_inventory(
        scores=scores,
        artifact_roots=artifact_roots,
        all_logs=all_logs,
        repo_root=repo_root,
        abs_tol=args.metric_abs_tol,
        rel_tol=args.metric_rel_tol,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "repo_root": str(repo_root),
        "ustc_scores": str(ustc_scores),
        "cic_scores": str(cic_scores),
        "ustc_artifacts": str(ustc_artifacts),
        "cic_artifacts": str(cic_artifacts),
        "log_roots": [str(path) for path in log_roots],
        "experiment_count": len(inventories),
        "explicit_execution_log_count": len(all_logs),
    }

    json_path = output_dir / "paper01_execution_evidence_inventory.json"
    csv_path = output_dir / "paper01_execution_evidence_inventory.csv"
    md_path = output_dir / "paper01_execution_evidence_inventory.md"

    json_report(json_path, inventories, metadata)
    csv_report(csv_path, inventories)
    markdown_report(md_path, inventories)

    print("[M6.4.1] Audit complete.")
    print(f"[M6.4.1] JSON: {json_path}")
    print(f"[M6.4.1] CSV:  {csv_path}")
    print(f"[M6.4.1] MD:   {md_path}")
    print("[M6.4.1] Historical artifacts modified: NO")
    print("[M6.4.1] Canonical linkage records created: NO")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
