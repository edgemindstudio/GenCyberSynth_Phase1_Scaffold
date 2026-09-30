#!/usr/bin/env python3
"""
scripts/summaries_to_jsonl.py

Paper 1 JSONL compatibility producer.

Two modes are intentionally supported during TrustForge migration.

Canonical mode
--------------
Requires explicit scientific identity:

    --canonical --dataset <dataset> --seed <seed>

Authority is selected only through:

    trustforge.paper01_exports
        -> trustforge.paper01_compat

No glob, newest-file selection, latest.json, or paper1.json chooses authority.

Legacy compatibility mode
-------------------------
Preserves the historical glob-based collector used by CI smoke workflows and
older local pipelines. This mode is intentionally separate from canonical
Paper 1 authority and remains migration compatibility only.

The output JSONL format preserves the legacy top-level metric/provenance shims
expected by downstream scripts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--canonical",
        action="store_true",
        help="Use canonical Paper 1 authority instead of legacy glob discovery.",
    )
    p.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used by canonical Paper 1 linkage.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Canonical Paper 1 dataset identity.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Canonical Paper 1 seed identity.",
    )
    p.add_argument(
        "--glob",
        default="artifacts/*/summaries/summary_*.json",
        help="Legacy-mode glob for input summary JSON files.",
    )
    p.add_argument(
        "--out",
        default="artifacts/summaries/phase1_summaries.jsonl",
        help="Output JSONL path.",
    )
    p.add_argument(
        "--schema",
        default=None,
        help="Optional path to a JSON schema file for validation.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Overwrite the output JSONL instead of appending.",
    )
    return p.parse_args()


def load_schema(schema_path: str | None):
    if not schema_path:
        return None
    try:
        import jsonschema  # type: ignore  # noqa: F401
    except Exception:
        print("jsonschema not installed; skipping validation.", file=sys.stderr)
        return None
    try:
        return (
            "jsonschema",
            json.loads(Path(schema_path).read_text(encoding="utf-8")),
        )
    except Exception as exc:
        print(f"Failed to load schema: {exc}", file=sys.stderr)
        return None


def existing_sources(out_path: Path) -> set[str]:
    seen: set[str] = set()
    if not out_path.exists():
        return seen

    with out_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            source_path = obj.get("source_path")
            if isinstance(source_path, str):
                seen.add(str(Path(source_path)))

    return seen


def _schema_validator(schema_path: str | None):
    schema_bundle = load_schema(schema_path)
    if not schema_bundle:
        return None, None

    try:
        import jsonschema  # type: ignore
    except Exception:
        return None, None

    return jsonschema.validate, schema_bundle[1]


def _validate_row(
    obj: dict[str, Any],
    *,
    validator,
    schema,
    source_label: str,
) -> None:
    if not validator or schema is None:
        return

    try:
        validator(instance=obj, schema=schema)
    except Exception as exc:
        print(
            f"Validation failed for {source_label}: {exc}",
            file=sys.stderr,
        )
        obj["_schema_error"] = str(exc)


def write_rows(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    reset: bool,
    validator=None,
    schema=None,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set() if reset else existing_sources(out_path)

    if reset and out_path.exists():
        out_path.unlink()

    written = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for obj in rows:
            source_path = obj.get("source_path")
            normalized_source = (
                str(Path(source_path))
                if isinstance(source_path, str)
                else None
            )

            if normalized_source and normalized_source in seen:
                continue

            _validate_row(
                obj,
                validator=validator,
                schema=schema,
                source_label=normalized_source or "<unknown>",
            )

            handle.write(
                json.dumps(
                    obj,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1

            if normalized_source:
                seen.add(normalized_source)

    return written


def canonical_rows(
    repo_root: Path | str,
    *,
    dataset: str,
    seed: int,
) -> list[dict[str, Any]]:
    from trustforge.paper01_compat import (
        paper01_export_view_to_legacy_jsonl,
    )
    from trustforge.paper01_exports import (
        load_paper01_export_view,
    )

    export_view = load_paper01_export_view(
        repo_root,
        dataset=dataset,
        seed=seed,
    )
    return list(
        paper01_export_view_to_legacy_jsonl(export_view)
    )


def make_run_id(model: str, src: Path) -> str:
    """
    Legacy-mode fallback only.

    Canonical mode never needs a wall-clock fallback because canonical
    compatibility records use accepted historical execution identity.
    """
    ts = None
    try:
        parts = src.stem.split("_")
        if len(parts) >= 3:
            ymd, hms = parts[-2], parts[-1]
            ts = datetime.strptime(
                ymd + hms,
                "%Y%m%d%H%M%S",
            )
    except Exception:
        ts = None

    if not ts:
        ts = datetime.now(timezone.utc)

    return f"{model}_{ts.strftime('%Y%m%dT%H%M%SZ')}"


def _dig(d: Dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first(*vals):
    for value in vals:
        if value is not None:
            return value
    return None


def _set_if_missing(
    d: Dict[str, Any],
    key: str,
    value: Any,
) -> None:
    if value is not None and d.get(key) is None:
        d[key] = value


_AUDIT_SHIMS = (
    "config_path",
    "config_sha1",
    "git_commit",
    "caps",
    "budget_per_class",
)


def _attach_audit_fields(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        return

    run_meta = obj.get("run_meta")
    has_run_meta = (
        isinstance(run_meta, dict)
        and bool(run_meta)
    )

    if has_run_meta:
        for key in _AUDIT_SHIMS:
            _set_if_missing(
                obj,
                key,
                run_meta.get(key),
            )
        return

    if any(
        obj.get(key) is not None
        for key in _AUDIT_SHIMS
    ):
        obj["run_meta"] = {
            key: obj.get(key)
            for key in _AUDIT_SHIMS
        }


def promote_top_level_metrics(obj: Dict[str, Any]) -> None:
    metrics = (
        obj.get("metrics")
        if isinstance(obj.get("metrics"), dict)
        else {}
    )
    counts = (
        obj.get("counts")
        if isinstance(obj.get("counts"), dict)
        else {}
    )
    meta = (
        obj.get("metrics_meta")
        if isinstance(obj.get("metrics_meta"), dict)
        else {}
    )
    downstream = _dig(metrics, "downstream")
    downstream = (
        downstream
        if isinstance(downstream, dict)
        else {}
    )

    num_real = _first(
        counts.get("num_real"),
        obj.get("counts.num_real"),
        obj.get("num_real"),
        _dig(obj, "images", "train_real"),
    )
    num_fake = _first(
        counts.get("num_fake"),
        obj.get("counts.num_fake"),
        obj.get("num_fake"),
        _dig(obj, "images", "synthetic"),
    )
    _set_if_missing(obj, "num_real", num_real)
    _set_if_missing(obj, "num_fake", num_fake)

    kid = _first(
        _dig(metrics, "kid"),
        obj.get("metrics.kid"),
        _dig(obj, "generative", "kid"),
    )
    fid = _first(
        _dig(metrics, "fid_macro"),
        _dig(metrics, "fid"),
        obj.get("metrics.fid_macro"),
        _dig(obj, "generative", "fid_macro"),
        _dig(obj, "generative", "fid"),
    )
    cfid = _first(
        _dig(metrics, "cfid"),
        _dig(metrics, "cfid_macro"),
        obj.get("metrics.cfid"),
        obj.get("metrics.cfid_macro"),
        _dig(obj, "generative", "cfid_macro"),
    )
    ms_ssim = _first(
        _dig(metrics, "ms_ssim"),
        obj.get("metrics.ms_ssim"),
        _dig(obj, "generative", "diversity"),
    )
    _set_if_missing(obj, "kid", kid)
    _set_if_missing(obj, "fid", fid)
    _set_if_missing(obj, "cfid", cfid)
    _set_if_missing(obj, "ms_ssim", ms_ssim)

    macro_f1 = _first(
        _dig(downstream, "macro_f1"),
        obj.get("metrics.downstream.macro_f1"),
        _dig(
            obj,
            "utility_real_plus_synth",
            "macro_f1",
        ),
    )
    macro_auprc = _first(
        _dig(downstream, "macro_auprc"),
        obj.get("metrics.downstream.macro_auprc"),
        _dig(
            obj,
            "utility_real_plus_synth",
            "macro_auprc",
        ),
    )
    balanced_acc = _first(
        _dig(downstream, "balanced_acc"),
        obj.get("metrics.downstream.balanced_acc"),
        _dig(
            obj,
            "utility_real_plus_synth",
            "balanced_accuracy",
        ),
        _dig(
            obj,
            "utility_real_plus_synth",
            "balanced_acc",
        ),
        _dig(
            obj,
            "utility_real_plus_synth",
            "bal_acc",
        ),
    )
    _set_if_missing(obj, "macro_f1", macro_f1)
    _set_if_missing(obj, "macro_auprc", macro_auprc)
    _set_if_missing(obj, "balanced_acc", balanced_acc)

    gen_precision = _first(
        _dig(metrics, "gen_precision"),
        obj.get("metrics.gen_precision"),
        _dig(
            obj,
            "utility_real_plus_synth",
            "macro_precision",
        ),
        _dig(
            obj,
            "utility_real_plus_synth",
            "per_class",
            "macro_precision",
        ),
    )
    gen_recall = _first(
        _dig(metrics, "gen_recall"),
        obj.get("metrics.gen_recall"),
        _dig(
            obj,
            "utility_real_plus_synth",
            "macro_recall",
        ),
        _dig(
            obj,
            "utility_real_plus_synth",
            "per_class",
            "macro_recall",
        ),
    )
    _set_if_missing(obj, "gen_precision", gen_precision)
    _set_if_missing(obj, "gen_recall", gen_recall)

    _set_if_missing(
        obj,
        "computed_at",
        _first(
            meta.get("computed_at"),
            obj.get("computed_at"),
        ),
    )
    _set_if_missing(
        obj,
        "metrics_version",
        _first(
            meta.get("metrics_version"),
            obj.get("metrics_version"),
        ),
    )
    _set_if_missing(
        obj,
        "kid_mode",
        _first(
            meta.get("kid_mode"),
            obj.get("kid_mode"),
        ),
    )


def legacy_rows(glob_pattern: str) -> list[dict[str, Any]]:
    files = sorted(Path(".").glob(glob_pattern))
    if not files:
        return []

    rows: list[dict[str, Any]] = []

    for path in files:
        source_path = str(Path(path))

        try:
            obj = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            print(
                f"Skip {path}: {exc}",
                file=sys.stderr,
            )
            continue

        parts = path.parts
        model = obj.get("model")
        if not isinstance(model, str):
            if (
                len(parts) >= 4
                and parts[0] == "artifacts"
            ):
                model = parts[1]
            else:
                model = "unknown"

        obj.setdefault("model", model)
        obj.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
        )
        obj.setdefault(
            "run_id",
            make_run_id(model, path),
        )
        obj.setdefault(
            "source_path",
            source_path,
        )

        _attach_audit_fields(obj)
        promote_top_level_metrics(obj)
        rows.append(obj)

    return rows


def _validate_identity_args(
    *,
    canonical: bool,
    dataset: str | None,
    seed: int | None,
) -> None:
    if canonical:
        if not dataset:
            raise SystemExit(
                "--dataset is required with --canonical"
            )
        if seed is None:
            raise SystemExit(
                "--seed is required with --canonical"
            )
        return

    if dataset is not None or seed is not None:
        raise SystemExit(
            "--dataset/--seed require --canonical"
        )


def main() -> int:
    args = parse_args()
    _validate_identity_args(
        canonical=args.canonical,
        dataset=args.dataset,
        seed=args.seed,
    )

    validator, schema = _schema_validator(
        args.schema
    )

    if args.canonical:
        rows = canonical_rows(
            Path(args.repo_root).resolve(),
            dataset=args.dataset,
            seed=args.seed,
        )
        mode = (
            f"canonical dataset={args.dataset} "
            f"seed={args.seed}"
        )
    else:
        rows = legacy_rows(args.glob)
        if not rows:
            print(
                "No summary_*.json files found.",
                file=sys.stderr,
            )
            return 1
        mode = f"legacy glob={args.glob}"

    written = write_rows(
        rows,
        out_path=Path(args.out),
        reset=args.reset,
        validator=validator,
        schema=schema,
    )

    print(
        f"Wrote {written} new line(s) → "
        f"{args.out} [{mode}]"
    )
    return 0 if written > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
