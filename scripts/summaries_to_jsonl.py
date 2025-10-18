# scripts/summaries_to_jsonl.py
#!/usr/bin/env python3
"""
Consolidate per-model JSON summaries into a single JSONL.

- Scans for summary_*.json under artifacts/*/summaries/ (configurable).
- Appends one JSON object per line to an output .jsonl.
- Idempotent: skips files that were already ingested (tracked via source_path).
- Optionally validates each JSON against a JSON Schema.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="artifacts/*/summaries/summary_*.json",
                   help="Glob for input summary JSON files.")
    p.add_argument("--out", default="artifacts/summaries/phase1_summaries.jsonl",
                   help="Output JSONL path.")
    p.add_argument("--schema", default=None,
                   help="Optional path to a JSON schema file for validation.")
    p.add_argument("--reset", action="store_true",
                   help="Overwrite the output JSONL instead of appending.")
    return p.parse_args()

def load_schema(schema_path: str | None):
    if not schema_path:
        return None
    try:
        import jsonschema  # type: ignore
    except Exception:
        print("jsonschema not installed; skipping validation.", file=sys.stderr)
        return None
    try:
        return ("jsonschema", json.loads(Path(schema_path).read_text(encoding="utf-8")))
    except Exception as e:
        print(f"Failed to load schema: {e}", file=sys.stderr)
        return None

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
                    seen.add(sp)
            except Exception:
                # If a line is malformed, ignore but don't crash.
                continue
    return seen

def make_run_id(model: str, src: Path) -> str:
    # Derive a stable-ish id if filename embeds a timestamp; else fallback to UTC now.
    ts = None
    try:
        # Try to find YYYYMMDD_HHMMSS in the filename
        stem = src.stem
        # common pattern: summary_YYYYMMDD_HHMMSS
        parts = stem.split("_")
        if len(parts) >= 3:
            ymd, hms = parts[-2], parts[-1]
            ts = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
    except Exception:
        ts = None
    if not ts:
        ts = datetime.now(timezone.utc)
    return f"{model}_{ts.strftime('%Y%m%dT%H%M%SZ')}"

def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build idempotency set
    seen = set()
    if not args.reset:
        seen = existing_sources(out_path)

    # Reset output if requested
    if args.reset and out_path.exists():
        out_path.unlink()

    # Optional schema validation
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

    files = sorted(Path(".").glob(args.glob))
    if not files:
        print("No summary_*.json files found.", file=sys.stderr)
        return 1

    written = 0
    with out_path.open("a", encoding="utf-8") as out:
        for f in files:
            src = str(f)
            if src in seen:
                # already ingested
                continue
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Skip {f}: {e}", file=sys.stderr)
                continue

            # Infer model from path artifacts/<model>/summaries/...
            parts = f.parts
            model = obj.get("model")
            if not isinstance(model, str):
                if len(parts) >= 4 and parts[0] == "artifacts":
                    model = parts[1]
                else:
                    model = "unknown"

            obj.setdefault("model", model)
            # obj.setdefault("timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z")
            obj.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
            obj.setdefault("run_id", make_run_id(model, f))
            obj.setdefault("source_path", src)

            # Optional validation
            if validator and schema:
                try:
                    validator(instance=obj, schema=schema)
                except Exception as e:
                    print(f"Validation failed for {f}: {e}", file=sys.stderr)
                    # Still write it, but mark invalid for visibility
                    obj["_schema_error"] = str(e)

            out.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    print(f"Wrote {written} new line(s) → {out_path}")
    return 0 if written > 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
