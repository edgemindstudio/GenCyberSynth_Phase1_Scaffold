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
- train : (optional) Route to a model repo's trainer if present.
- synth : Generate synthetic images using a registered adapter.
- eval  : Run evaluation using gcs-core on the latest manifest (optionally skip synth).
- list  : Show registered adapters and any skipped adapter imports.

Typical usage
-------------
python -m app.main synth --model diffusion --config configs/config.yaml
python -m app.main eval  --model diffusion --config configs/config.yaml
python -m app.main train --model gan        --config configs/config.yaml
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
from typing import Any, Dict, Callable
from pathlib import Path

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
# Audit / provenance metadata (Option 1)
# ---------------------------------------------------------------------------
def attach_run_meta(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Attach audit metadata to cfg so downstream summaries can prove which config was used.

    Safe:
    - Never raises outward.
    - Stores None when a value cannot be computed.

    Writes/updates:
      cfg["run_meta"] = {
        "config_path": <absolute path or None>,
        "config_sha1": <sha1 hex or None>,
        "git_commit": <commit hex or None>,
        "caps": {...},
        "budget_per_class": <int or None>,
      }
    """
    def _sha1(path: str) -> str | None:
        try:
            b = Path(path).expanduser().read_bytes()
            return hashlib.sha1(b).hexdigest()
        except Exception:
            return None

    def _git_commit(repo_root: str) -> str | None:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
        except Exception:
            return None

    # Resolve config path (absolute) if provided
    cfg_path: str | None = None
    try:
        if getattr(args, "config", None):
            cfg_path = str(Path(args.config).expanduser().resolve())
    except Exception:
        cfg_path = None

    # app/main.py is expected at: <repo_root>/app/main.py -> parents[1] = repo_root
    repo_root = str(Path(__file__).resolve().parents[1])

    # Extract commonly-audited knobs from config (best-effort)
    try:
        per_class_cap = cfg.get("evaluator", {}).get("per_class_cap", None)
    except Exception:
        per_class_cap = None

    try:
        budget_per_class = cfg.get("synth", {}).get("n_per_class", None)
    except Exception:
        budget_per_class = None

    # Merge with any existing run_meta (Slurm merged config already injects useful fields)
    existing = cfg.get("run_meta")
    existing = existing if isinstance(existing, dict) else {}

    rm = dict(existing)  # copy
    rm.setdefault("config_path", cfg_path)
    rm.setdefault("config_sha1", _sha1(cfg_path) if cfg_path else None)
    rm.setdefault("git_commit", _git_commit(repo_root))

    # caps always refresh (safe)
    rm["caps"] = {
        "manifest_cap_per_class": per_class_cap,
        "fid_cap_per_class": per_class_cap,
    }

    # budget: DO NOT overwrite if already set by overrides/slurm
    if rm.get("budget_per_class") is None:
        rm["budget_per_class"] = budget_per_class

    cfg["run_meta"] = rm



# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    """
    Train a model (best-effort routing).

    This CLI does NOT implement training logic directly. Instead it tries to import:
        <model>.train
    and then calls either:
        - main(argv)   if present
        - train(cfg)   if present

    IMPORTANT:
      If overrides are provided, we must pass the MERGED config to the trainer.
      For trainers that expose main(argv) (like gan.train), we write a temp YAML
      containing the merged cfg and call main(['--config', tmp_yaml]).
    """
    cfg = load_config(args.config)

    # Merge overrides (base + overrides)
    if getattr(args, "overrides", None):
        ov = load_config(args.overrides)
        deep_update(cfg, ov)

    # Ensure cfg has paths key and apply artifacts override (so training can write consistently)
    cfg.setdefault("paths", {})
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts

    # Attach audit metadata (useful even for training logs/checkpoints)
    attach_run_meta(cfg, args)

    _info(f"Train model : {args.model}")
    _info(f"Config      : {args.config or '<defaults>'}")
    _info(f"Artifacts   : {artifacts_root(cfg, args.artifacts)}")
    if getattr(args, "overrides", None):
        _info(f"Overrides   : {args.overrides}")

    module_name = f"{args.model}.train"
    try:
        mod = __import__(module_name, fromlist=["*"])
    except Exception as e:
        _warn(f"No trainer module found at '{module_name}' ({e.__class__.__name__}: {e}).")
        _info(
            "Tip: add a train.py to your model package (expose `main(argv)` or `train(config)`), "
            "or skip training and just run synth/eval."
        )
        return 1

    has_main = hasattr(mod, "main") and callable(getattr(mod, "main"))
    has_train = hasattr(mod, "train") and callable(getattr(mod, "train"))

    if not (has_main or has_train):
        _warn(f"Trainer module '{module_name}' has no callable main()/train(). Nothing to do.")
        return 1

    # Prefer main(argv) if present, else train(cfg)
    if has_main:
        try:
            # Write merged cfg (base + overrides + artifacts + run_meta) to a temp YAML
            import tempfile
            if yaml is None:
                _err("PyYAML is required to pass merged config to trainers via temp YAML. Install with: pip install pyyaml")
                return 2

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
                tmp_cfg = f.name

            _info(f"Calling {module_name}.main(['--config', '{tmp_cfg}'])")
            ret = mod.main(["--config", tmp_cfg])  # type: ignore[attr-defined]
            return int(ret) if isinstance(ret, int) else 0

        except TypeError:
            # Fallback: maybe it actually wants a dict
            try:
                _info(f"Falling back: {module_name}.main(config_dict)")
                ret = mod.main(cfg)  # type: ignore[attr-defined]
                return int(ret) if isinstance(ret, int) else 0
            except Exception as e:
                _err(f"Training failed: {e.__class__.__name__}: {e}")
                return 1
        except Exception as e:
            _err(f"Training failed: {e.__class__.__name__}: {e}")
            return 1

    # Else: has_train
    try:
        _info(f"Calling {module_name}.train(config_dict)")
        ret = mod.train(cfg)  # type: ignore[attr-defined]
        return int(ret) if isinstance(ret, int) else 0
    except TypeError:
        # Signature mismatch: maybe it expects argv-style input
        try:
            import tempfile
            if yaml is None:
                _err("PyYAML is required to pass merged config to trainers via temp YAML. Install with: pip install pyyaml")
                return 2

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
                tmp_cfg = f.name

            _info(f"Falling back: {module_name}.train(['--config', '{tmp_cfg}'])")
            ret = mod.train(["--config", tmp_cfg])  # type: ignore[attr-defined]
            return int(ret) if isinstance(ret, int) else 0
        except Exception as e:
            _err(f"Training failed: {e.__class__.__name__}: {e}")
            return 1
    except Exception as e:
        _err(f"Training failed: {e.__class__.__name__}: {e}")
        return 1


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
        description="GenCyberSynth – unified CLI for training, synthesis & evaluation",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # train (optional wire-through to per-model trainer)
    p_t = sub.add_parser("train", help="Train the model (routes into <model>.train if available)")
    p_t.add_argument("--model", required=True, help="Model family (e.g., gan, diffusion, vae, ...)")
    p_t.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_t.add_argument("--artifacts", default=None, help="Override artifacts root directory")
    p_t.set_defaults(func=cmd_train)

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
