#!/usr/bin/env python3
"""
scripts/summaries_to_jsonl.py

Consolidate per-model JSON summaries into a single JSONL.

What it does
------------
- Scans for summary_*.json under artifacts/*/summaries/ (configurable).
- Appends one JSON object per line to an output .jsonl.
- Idempotent: skips files that were already ingested (tracked via source_path).
- Optionally validates each JSON object against a JSON Schema.

Why JSONL?
----------
JSONL (one JSON object per line) is:
- easy to stream/process
- easy to grep
- easy to load into pandas
- robust for large artifact collections

IMPORTANT (Phase-1 pipeline requirement)
----------------------------------------
Downstream scripts (jsonl_to_csv.py, quick greps, README tables) benefit from having
key metrics at the TOP LEVEL in the JSONL, e.g.:

  kid, balanced_acc, gen_precision, gen_recall,
  computed_at, metrics_version, kid_mode,
  macro_f1, macro_auprc,
  num_real, num_fake,
  fid, cfid, ms_ssim

Some summaries store these nested under:
  metrics_meta.*, metrics.downstream.*, metrics.*, counts.*, utility_real_plus_synth.*, generative.*

So, before writing each record, we "promote" (best-effort) these keys to top-level
WITHOUT deleting nested fields (safe + backwards compatible).

NEW (Provenance / audit requirement)
------------------------------------
Reviewers need evidence of:
- exactly which config file was used,
- its hash,
- the git commit,
- evaluation caps,
- synthetic budget per class.

The CLI (app/main.py) injects cfg["run_meta"], and eval/runner.py persists it into
summary_*.json / latest.json. This script ensures those audit fields are preserved in
phase1_summaries.jsonl by:

- keeping canonical nested "run_meta"
- ensuring top-level shims exist:
    config_path, config_sha1, git_commit, caps, budget_per_class
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict


# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--glob",
        default="artifacts/*/summaries/summary_*.json",
        help="Glob for input summary JSON files.",
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


# -----------------------------------------------------------------------------
# Optional JSON Schema validation
# -----------------------------------------------------------------------------
def load_schema(schema_path: str | None):
    """
    Load an optional JSON schema. If jsonschema isn't installed or schema missing,
    we skip validation without failing the build.
    """
    if not schema_path:
        return None
    try:
        import jsonschema  # type: ignore  # noqa: F401
    except Exception:
        print("jsonschema not installed; skipping validation.", file=sys.stderr)
        return None
    try:
        return ("jsonschema", json.loads(Path(schema_path).read_text(encoding="utf-8")))
    except Exception as e:
        print(f"Failed to load schema: {e}", file=sys.stderr)
        return None


# -----------------------------------------------------------------------------
# Idempotency: avoid duplicating lines for files we've already ingested
# -----------------------------------------------------------------------------
def existing_sources(out_path: Path) -> set[str]:
    """Return set of source_path values already in JSONL (for idempotency)."""
    seen: set[str] = set()
    if not out_path.exists():
        return seen

    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sp = obj.get("source_path")
                if isinstance(sp, str):
                    # Normalize so "./artifacts/..." and "artifacts/..." match
                    sp_norm = str(Path(sp))
                    seen.add(sp_norm)
            except Exception:
                continue
    return seen


# -----------------------------------------------------------------------------
# Helpers for missing fields / normalization
# -----------------------------------------------------------------------------
def make_run_id(model: str, src: Path) -> str:
    """
    Derive a stable-ish run_id if filename embeds a timestamp; else fallback to UTC now.

    Expected filename pattern:
      summary_YYYYMMDD_HHMMSS.json
    """
    ts = None
    try:
        stem = src.stem
        parts = stem.split("_")
        if len(parts) >= 3:
            ymd, hms = parts[-2], parts[-1]
            ts = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
    except Exception:
        ts = None

    if not ts:
        ts = datetime.now(timezone.utc)

    return f"{model}_{ts.strftime('%Y%m%dT%H%M%SZ')}"


def _dig(d: Dict[str, Any], *path: str) -> Any:
    """Safely fetch nested dict values by path; returns None if missing."""
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _first(*vals):
    """Return the first non-None value from a list of candidates."""
    for v in vals:
        if v is not None:
            return v
    return None


def _set_if_missing(d: Dict[str, Any], k: str, v: Any) -> None:
    """Set d[k]=v only if v is not None and d[k] is missing/None."""
    if v is None:
        return
    if d.get(k) is None:
        d[k] = v


# -----------------------------------------------------------------------------
# NEW: Audit/provenance preservation
# -----------------------------------------------------------------------------
_AUDIT_SHIMS = ("config_path", "config_sha1", "git_commit", "caps", "budget_per_class")


def _attach_audit_fields(obj: Dict[str, Any]) -> None:
    """
    Ensure provenance/audit fields survive into the consolidated JSONL.

    Canonical source (preferred):
      obj["run_meta"] = {
        "config_path": ...,
        "config_sha1": ...,
        "git_commit": ...,
        "caps": {...},
        "budget_per_class": ...
      }

    Compatibility shims (top-level; useful for older tools/greps):
      obj["config_path"], obj["config_sha1"], obj["git_commit"], obj["caps"], obj["budget_per_class"]

    Behavior:
      - If run_meta exists → populate missing shims from run_meta.
      - Else if shims exist → reconstruct run_meta (best effort).
      - Else → do nothing (older summaries may not have provenance).
    """
    if not isinstance(obj, dict):
        return

    rm = obj.get("run_meta")
    has_rm = isinstance(rm, dict) and bool(rm)

    # Case 1: canonical run_meta exists → ensure shims exist too
    if has_rm:
        for k in _AUDIT_SHIMS:
            _set_if_missing(obj, k, rm.get(k))
        return

    # Case 2: no run_meta, but top-level shims exist → reconstruct run_meta
    if any(obj.get(k) is not None for k in _AUDIT_SHIMS):
        obj["run_meta"] = {
            "config_path": obj.get("config_path"),
            "config_sha1": obj.get("config_sha1"),
            "git_commit": obj.get("git_commit"),
            "caps": obj.get("caps"),
            "budget_per_class": obj.get("budget_per_class"),
        }


# -----------------------------------------------------------------------------
# Metric promotion: keep your existing behavior (unchanged)
# -----------------------------------------------------------------------------
def promote_top_level_metrics(obj: Dict[str, Any]) -> None:
    """
    Promote frequently-used fields to top-level keys so JSONL -> CSV remains simple
    and consistent across runtime vs backfill summaries.
    """
    metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else {}
    counts = obj.get("counts") if isinstance(obj.get("counts"), dict) else {}
    meta = obj.get("metrics_meta") if isinstance(obj.get("metrics_meta"), dict) else {}
    downstream = _dig(metrics, "downstream")
    downstream = downstream if isinstance(downstream, dict) else {}

    # --- counts ---
    num_real = _first(counts.get("num_real"), obj.get("counts.num_real"), obj.get("num_real"), _dig(obj, "images", "train_real"))
    num_fake = _first(counts.get("num_fake"), obj.get("counts.num_fake"), obj.get("num_fake"), _dig(obj, "images", "synthetic"))
    _set_if_missing(obj, "num_real", num_real)
    _set_if_missing(obj, "num_fake", num_fake)

    # --- generative ---
    kid = _first(_dig(metrics, "kid"), obj.get("metrics.kid"), _dig(obj, "generative", "kid"))
    fid = _first(_dig(metrics, "fid_macro"), _dig(metrics, "fid"), obj.get("metrics.fid_macro"), _dig(obj, "generative", "fid_macro"), _dig(obj, "generative", "fid"))
    cfid = _first(_dig(metrics, "cfid"), _dig(metrics, "cfid_macro"), obj.get("metrics.cfid"), obj.get("metrics.cfid_macro"), _dig(obj, "generative", "cfid_macro"))
    ms_ssim = _first(_dig(metrics, "ms_ssim"), obj.get("metrics.ms_ssim"), _dig(obj, "generative", "diversity"))
    _set_if_missing(obj, "kid", kid)
    _set_if_missing(obj, "fid", fid)
    _set_if_missing(obj, "cfid", cfid)
    _set_if_missing(obj, "ms_ssim", ms_ssim)

    # --- downstream / utility (R+S block) ---
    macro_f1 = _first(_dig(downstream, "macro_f1"), obj.get("metrics.downstream.macro_f1"), _dig(obj, "utility_real_plus_synth", "macro_f1"))
    macro_auprc = _first(_dig(downstream, "macro_auprc"), obj.get("metrics.downstream.macro_auprc"), _dig(obj, "utility_real_plus_synth", "macro_auprc"))
    balanced_acc = _first(
        _dig(downstream, "balanced_acc"),
        obj.get("metrics.downstream.balanced_acc"),
        _dig(obj, "utility_real_plus_synth", "balanced_accuracy"),
        _dig(obj, "utility_real_plus_synth", "balanced_acc"),
        _dig(obj, "utility_real_plus_synth", "bal_acc"),
    )
    _set_if_missing(obj, "macro_f1", macro_f1)
    _set_if_missing(obj, "macro_auprc", macro_auprc)
    _set_if_missing(obj, "balanced_acc", balanced_acc)

    # --- downstream macro precision/recall (Phase-1 uses gen_precision/gen_recall columns) ---
    gen_precision = _first(
        _dig(metrics, "gen_precision"),
        obj.get("metrics.gen_precision"),
        _dig(obj, "utility_real_plus_synth", "macro_precision"),
        _dig(obj, "utility_real_plus_synth", "per_class", "macro_precision"),
    )
    gen_recall = _first(
        _dig(metrics, "gen_recall"),
        obj.get("metrics.gen_recall"),
        _dig(obj, "utility_real_plus_synth", "macro_recall"),
        _dig(obj, "utility_real_plus_synth", "per_class", "macro_recall"),
    )
    _set_if_missing(obj, "gen_precision", gen_precision)
    _set_if_missing(obj, "gen_recall", gen_recall)

    # --- provenance/meta ---
    computed_at = _first(meta.get("computed_at"), obj.get("computed_at"))
    metrics_version = _first(meta.get("metrics_version"), obj.get("metrics_version"))
    kid_mode = _first(meta.get("kid_mode"), obj.get("kid_mode"))
    _set_if_missing(obj, "computed_at", computed_at)
    _set_if_missing(obj, "metrics_version", metrics_version)
    _set_if_missing(obj, "kid_mode", kid_mode)


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build idempotency set (unless --reset)
    seen = set()
    if not args.reset:
        seen = existing_sources(out_path)

    # Reset output if requested
    if args.reset and out_path.exists():
        out_path.unlink()

    # Optional schema validation setup
    schema_bundle = load_schema(args.schema)
    validator = None
    schema = None
    if schema_bundle:
        try:
            import jsonschema  # type: ignore
            validator = jsonschema.validate
            schema = schema_bundle[1]
        except Exception:
            validator = None
            schema = None

    # Discover files
    files = sorted(Path(".").glob(args.glob))
    if not files:
        print("No summary_*.json files found.", file=sys.stderr)
        return 1

    written = 0
    with out_path.open("a", encoding="utf-8") as out:
        for f in files:
            # Normalize so "./artifacts/..." and "artifacts/..." match
            src = str(Path(f))

            # Idempotency: skip files already ingested
            if src in seen:
                continue

            # Load JSON summary
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Skip {f}: {e}", file=sys.stderr)
                continue

            # Infer model from JSON or path artifacts/<model>/summaries/...
            parts = f.parts
            model = obj.get("model")
            if not isinstance(model, str):
                if len(parts) >= 4 and parts[0] == "artifacts":
                    model = parts[1]
                else:
                    model = "unknown"

            # Ensure required fields exist (non-destructive defaults)
            obj.setdefault("model", model)
            obj.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
            obj.setdefault("run_id", make_run_id(model, f))
            obj.setdefault("source_path", src)

            # NEW: ensure audit/provenance fields survive into JSONL
            _attach_audit_fields(obj)

            # Promote key metrics to top-level (critical for jsonl_to_csv simplicity)
            promote_top_level_metrics(obj)

            # Optional validation (note: schema may disallow new fields if additionalProperties=false)
            if validator and schema:
                try:
                    validator(instance=obj, schema=schema)
                except Exception as e:
                    print(f"Validation failed for {f}: {e}", file=sys.stderr)
                    obj["_schema_error"] = str(e)

            # Write compact JSON (1 object per line)
            out.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    print(f"Wrote {written} new line(s) → {out_path}")
    return 0 if written > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
