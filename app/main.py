# app/main.py

"""
GenCyberSynth Scaffold CLI
==========================

A small, production-friendly CLI that wires together:
- Adapters (via `adapters.registry`) for per-model synthesis,
- Evaluator runner (via `eval.runner`) for metrics on generated samples,
- A simple YAML config loader with sane defaults.

Subcommands
-----------
- synth : Generate synthetic images using a registered adapter.
- eval  : Run evaluation using gcs-core on the latest manifest (optionally skip synth).
- list  : Show registered adapters and any skipped adapter imports.

Typical usage
-------------
python -m app.main synth --model diffusion --config configs/config.yaml
python -m app.main eval  --model diffusion --config configs/config.yaml
python -m app.main list

Config expectations (minimal)
-----------------------------
paths:
  artifacts: "artifacts"           # root to write model/{synthetic,summaries}/...

evaluator:
  per_class_cap: 200               # (optional) cap per-class images loaded for metrics

Notes
-----
- Adapters are responsible for writing a manifest JSON to:
    {paths.artifacts}/{model}/synthetic/manifest.json
  The evaluator looks for that path by default.
- Keep adapters pure: accept a `config: dict`, return a manifest dict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

# Local modules (kept light so the CLI starts even if some deps are missing)
from adapters.registry import make_adapter, list_adapters, SKIPPED_IMPORTS
from eval.runner import evaluate_model_suite

# Optional dependency (loaded lazily but we check here for a friendlier error)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    yaml = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _info(msg: str) -> None:
    print(f"[info] {msg}")


def _warn(msg: str) -> None:
    print(f"[warn] {msg}")


def _err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


def load_config(path: str | None) -> Dict[str, Any]:
    """
    Load a YAML config or return an empty dict if the path is None/nonexistent.

    Raises a clean error if PyYAML is missing and a config path is provided.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        _warn(f"Config file not found: {path} (continuing with defaults)")
        return {}
    if yaml is None:
        _err("PyYAML is required to read config files. Install with: pip install pyyaml")
        raise SystemExit(2)
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def artifacts_root(cfg: Dict[str, Any], override: str | None = None) -> str:
    """
    Resolve artifacts root in priority:
    1) explicit override (--artifacts),
    2) cfg['paths']['artifacts'],
    3) 'artifacts' (default).
    """
    if override:
        return override
    return cfg.get("paths", {}).get("artifacts", "artifacts")


def _manifest_path(model_name: str, arts_root: str) -> str:
    return os.path.join(arts_root, model_name, "synthetic", "manifest.json")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
def cmd_synth(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.setdefault("paths", {})
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts

    _info(f"Adapter: {args.model}")
    _info(f"Config : {args.config or '<defaults>'}")

    try:
        adapter = make_adapter(args.model)
    except KeyError as e:
        _err(str(e))
        _info(f"Registered adapters: {', '.join(list_adapters()) or '<none>'}")
        if SKIPPED_IMPORTS:
            _warn("Some adapters failed to import:\n  - " + "\n  - ".join(SKIPPED_IMPORTS))
        return 2

    manifest = adapter.synth(cfg)  # Adapter is responsible for writing the manifest

    # Ensure the manifest exists at the conventional location (helpful for tooling)
    arts_root = artifacts_root(cfg, args.artifacts)
    expected_path = _manifest_path(args.model, arts_root)
    if not os.path.exists(expected_path):
        # Write a convenience copy if the adapter returned a manifest but didn't write it
        try:
            os.makedirs(os.path.dirname(expected_path), exist_ok=True)
            with open(expected_path, "w") as f:
                json.dump(manifest, f, indent=2)
            _warn(f"The adapter did not write the conventional manifest; "
                  f"a copy was saved to: {expected_path}")
        except Exception as e:
            _warn(f"Could not save manifest copy to {expected_path}: {e}")

    _info(f"Synthesis complete. Manifest: {expected_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.setdefault("paths", {})
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts

    _info(f"Evaluate model: {args.model}")
    _info(f"Config        : {args.config or '<defaults>'}")
    _info(f"No-synth flag : {args.no_synth}")

    try:
        # If no_synth is False, we *do not* auto-generate here; we assume you ran synth first.
        # (If you prefer an auto-flow, invoke the adapter here before evaluate_model_suite)
        evaluate_model_suite(cfg, model_name=args.model, no_synth=args.no_synth)
    except FileNotFoundError as e:
        _err(str(e))
        return 2
    except Exception as e:
        _err(f"Evaluation failed: {e.__class__.__name__}: {e}")
        return 1

    return 0


def cmd_list(_: argparse.Namespace) -> int:
    names = list_adapters()
    if not names:
        _info("No adapters registered.")
    else:
        _info("Registered adapters:")
        for n in names:
            print(f"  - {n}")
    if SKIPPED_IMPORTS:
        _warn("Adapters skipped during import (non-fatal):")
        for note in SKIPPED_IMPORTS:
            print(f"  * {note}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gencs",
        description="GenCyberSynth – unified CLI for synthesis & evaluation",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # synth
    p_s = sub.add_parser("synth", help="Generate synthetic images via an adapter")
    p_s.add_argument("--model", required=True, help="Adapter name (e.g., diffusion, gan, vae, ...)")
    p_s.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_s.add_argument("--artifacts", default=None, help="Override artifacts root directory")
    p_s.set_defaults(func=cmd_synth)

    # eval
    p_e = sub.add_parser("eval", help="Run evaluation (uses gcs-core) on latest manifest")
    p_e.add_argument("--model", required=True, help="Adapter name (e.g., diffusion, gan, vae, ...)")
    p_e.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_e.add_argument("--artifacts", default=None, help="Override artifacts root directory")
    p_e.add_argument("--no-synth", action="store_true", help="Skip metrics that require synthetic images")
    p_e.set_defaults(func=cmd_eval)

    # list
    p_l = sub.add_parser("list", help="List registered adapters")
    p_l.set_defaults(func=cmd_list)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Dispatch
    return args.func(args)  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())
