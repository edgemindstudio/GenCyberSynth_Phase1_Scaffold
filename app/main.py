# app/main.py
"""
GenCyberSynth Scaffold CLI
==========================

This module defines a small, production-friendly command-line interface (CLI) that
wires together three major pieces of the GenCyberSynth scaffold:

1) Adapters (via `adapters.registry`)
   - Responsible for synthesis: generating synthetic samples and writing a manifest.

2) Evaluator (via `eval.runner`)
   - Responsible for evaluation: computing metrics from REAL, SYNTH, or REAL+SYNTH.

3) Config loader (YAML)
   - Provides a single place to define paths, budgets, caps, etc.

Why this file exists
--------------------
You want a single command entry point so your Slurm jobs can run consistent commands:
  - train (optional)
  - synth
  - eval
  - list

Typical usage
-------------
python -m app.main synth --model diffusion --config configs/config.yaml
python -m app.main eval  --model diffusion --config configs/config.yaml
python -m app.main train --model gan        --config configs/config.yaml
python -m app.main list

Config expectations (minimal)
-----------------------------
paths:
  artifacts: "artifacts"

evaluator:
  per_class_cap: 200

synth:
  n_per_class: 2000

Audit / Tuning proof (Option 1)
-------------------------------
This CLI attaches `cfg["run_meta"]` at runtime. That metadata includes:
  - config_path (absolute)
  - config_sha1 (content hash)
  - git_commit (repo commit hash)
  - caps (e.g., per-class cap)
  - budget_per_class (synth.n_per_class)

IMPORTANT:
  Attaching cfg["run_meta"] here only guarantees it exists in the config passed
  into your evaluation pipeline. You still need to ensure the summary writer
  (inside eval/runner.py or gcs_core/val_common.py) copies that `run_meta`
  into the final summary JSON.

Design choices
--------------
- This CLI is intentionally lightweight: it imports model adapters through a registry,
  and it calls the evaluator through a single function.
- It is defensive: missing config file => continues with defaults; missing adapters => helpful error.
- It is audit-friendly: adds run_meta without crashing if git/hash cannot be computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# Local modules (kept light so the CLI starts even if some deps are missing)
from adapters.registry import SKIPPED_IMPORTS, list_adapters, make_adapter
from eval.runner import evaluate_model_suite

# Optional dependency (we fail nicely if YAML is requested but PyYAML is missing)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Logging helpers (simple, consistent formatting)
# ---------------------------------------------------------------------------
def _info(msg: str) -> None:
    """Standard informational log."""
    print(f"[info] {msg}")


def _warn(msg: str) -> None:
    """Standard warning log."""
    print(f"[warn] {msg}")


def _err(msg: str) -> None:
    """Standard error log (stderr)."""
    print(f"[error] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------
def load_config(path: str | None) -> Dict[str, Any]:
    """
    Load a YAML config file into a Python dict.

    Behavior:
    - If `path` is None/empty: return {}.
    - If the file does not exist: warn and return {}.
    - If PyYAML is missing and a file path was provided: exit with code 2.

    Notes:
    - This function does not validate schema; downstream code should handle missing keys.
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

    # Ensure dict return type even if YAML file contains unexpected types
    if not isinstance(data, dict):
        _warn(f"Config file loaded but top-level YAML is not a mapping/dict: {path}. Using defaults.")
        return {}

    return data


def artifacts_root(cfg: Dict[str, Any], override: str | None = None) -> str:
    """
    Resolve artifacts root directory in priority order:
      1) explicit override (--artifacts)
      2) cfg['paths']['artifacts']
      3) 'artifacts' (default)

    Returned value is a string path (not created here).
    """
    if override:
        return override
    return str(cfg.get("paths", {}).get("artifacts", "artifacts"))
    

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _manifest_path(model_name: str, arts_root: str) -> str:
    """
    Conventional location for the synthesis manifest.
    Adapters SHOULD write here:
        <artifacts>/<model>/synthetic/manifest.json
    """
    return os.path.join(arts_root, model_name, "synthetic", "manifest.json")


def _infer_config_variant(cfg: Dict[str, Any]) -> str | None:
    """
    Best-effort inference of config variant letter (A/B/...) from run_meta.

    Priority:
      1) cfg["run_meta"]["config_variant"]  (preferred)
      2) cfg["run_meta"]["config_id"]       (e.g., "gan_A" -> "A")
    """
    rm = cfg.get("run_meta")
    if not isinstance(rm, dict):
        return None

    v = rm.get("config_variant")
    if isinstance(v, str) and v:
        return v

    cid = rm.get("config_id")
    if isinstance(cid, str) and "_" in cid:
        maybe = cid.split("_")[-1]
        return maybe if maybe else None

    return None


def _per_run_manifest_path(model_name: str, arts_root: str, cfg: Dict[str, Any]) -> str | None:
    """
    Seed/config-specific manifest path:
      <arts>/<model>/synthetic/<model>_<CFG>_seed<SEED>/manifest.json

    Returns None if we cannot determine CFG and SEED.
    """
    cfg_variant = _infer_config_variant(cfg)
    seed = cfg.get("SEED")

    if not isinstance(cfg_variant, str) or not cfg_variant:
        return None

    # YAML usually loads SEED as int, but allow string digits too.
    if not isinstance(seed, int):
        if isinstance(seed, str) and seed.isdigit():
            seed = int(seed)
        else:
            return None

    run_dir = os.path.join(
        arts_root, model_name, "synthetic", f"{model_name}_{cfg_variant}_seed{seed}"
    )
    return os.path.join(run_dir, "manifest.json")


# ---------------------------------------------------------------------------
# Audit / provenance metadata (Option 1)
# ---------------------------------------------------------------------------
def attach_run_meta(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Attach audit metadata to cfg so downstream summaries can prove which config was used.

    This function is deliberately safe:
    - It never raises an exception outward.
    - If it cannot compute a value, it stores None.

    Output:
      cfg["run_meta"] = {
        "config_path": <absolute path or None>,
        "config_sha1": <sha1 hex or None>,
        "git_commit": <commit hex or None>,
        "caps": {...},
        "budget_per_class": <int or None>,
      }
    """

    def _sha1(path: str) -> str | None:
        """SHA1 hash of file contents (used as an immutable config fingerprint)."""
        try:
            b = Path(path).expanduser().read_bytes()
            return hashlib.sha1(b).hexdigest()
        except Exception:
            return None

    def _git_commit(repo_root: str) -> str | None:
        """Current git commit hash for repo_root (if repo_root is a git repo)."""
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
    per_class_cap = None
    try:
        per_class_cap = cfg.get("evaluator", {}).get("per_class_cap", None)
    except Exception:
        per_class_cap = None

    budget_per_class = None
    try:
        budget_per_class = cfg.get("synth", {}).get("n_per_class", None)
    except Exception:
        budget_per_class = None

    # cfg["run_meta"] = {
        # "config_path": cfg_path,
        # "config_sha1": _sha1(cfg_path) if cfg_path else None,
        # "git_commit": _git_commit(repo_root),
        # "caps": {
            # # If you later split these caps, update here accordingly.
            # "manifest_cap_per_class": per_class_cap,
            # "fid_cap_per_class": per_class_cap,
        # },
        # "budget_per_class": budget_per_class,
    # }
    
    # Updated from above block to below
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
    
    # budget: DO NOT overwrite if already set by overrides
    if rm.get("budget_per_class") is None:
        rm["budget_per_class"] = budget_per_class

    cfg["run_meta"] = rm



# ---------------------------------------------------------------------------
# Helper Functions for Deep Merging Dicts
# ---------------------------------------------------------------------------
def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base
    

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

    This lets each model family own its training code without complicating the CLI.
    """
    cfg = load_config(args.config)
        
    if args.overrides:
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

    # Prefer `main(argv)` if available, else `train(cfg)`
    if has_main:
        try:
            _info(f"Calling {module_name}.main(['--config', '{args.config}'])")
            ret = mod.main(["--config", args.config])  # type: ignore[attr-defined]
            return int(ret) if isinstance(ret, int) else 0
        except TypeError:
            # Signature mismatch: maybe it expects a dict
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
            _info(f"Falling back: {module_name}.train(['--config', '{args.config}'])")
            ret = mod.train(["--config", args.config])  # type: ignore[attr-defined]
            return int(ret) if isinstance(ret, int) else 0
        except Exception as e:
            _err(f"Training failed: {e.__class__.__name__}: {e}")
            return 1
    except Exception as e:
        _err(f"Training failed: {e.__class__.__name__}: {e}")
        return 1


def cmd_synth(args: argparse.Namespace) -> int:
    """
    Synthesize (generate) synthetic samples for a given model adapter.

    Flow:
      1) Load config
      2) Build adapter from registry
      3) adapter.synth(cfg) -> returns manifest dict (and should write manifest file)
      4) Ensure a conventional manifest path exists (write copy if adapter didn't)
      5) ALSO write a per-run manifest (seed/config-specific) for tuning safety
    """
    cfg = load_config(args.config)

    if args.overrides:
        ov = load_config(args.overrides)
        deep_update(cfg, ov)

    cfg.setdefault("paths", {})
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts

    # Attach audit metadata so adapter can optionally record provenance in manifests
    attach_run_meta(cfg, args)

    _info(f"Adapter     : {args.model}")
    _info(f"Config      : {args.config or '<defaults>'}")
    _info(f"Artifacts   : {artifacts_root(cfg, args.artifacts)}")

    try:
        adapter = make_adapter(args.model)
    except KeyError as e:
        _err(str(e))
        _info(f"Registered adapters: {', '.join(list_adapters()) or '<none>'}")
        if SKIPPED_IMPORTS:
            _warn("Some adapters failed to import (non-fatal for others):\n  - " + "\n  - ".join(SKIPPED_IMPORTS))
        return 2

    # Adapter is responsible for writing the manifest. We still return/handle it defensively.
    manifest = adapter.synth(cfg)

    # -----------------------------------------------------------------------
    # Manifest paths
    # -----------------------------------------------------------------------
    arts_root = artifacts_root(cfg, args.artifacts)

    # Backwards-compatible (shared) manifest path
    expected_path = _manifest_path(args.model, arts_root)

    # New: per-run (seed/config-specific) manifest path
    per_run_path = _per_run_manifest_path(args.model, arts_root, cfg)

    # Optional: expose these paths for adapters/tooling (does not break anything)
    cfg.setdefault("paths", {})
    cfg["paths"]["shared_manifest_path"] = expected_path
    if per_run_path:
        cfg["paths"]["per_run_manifest_path"] = per_run_path
        cfg["paths"]["per_run_synth_dir"] = os.path.dirname(per_run_path)

    # -----------------------------------------------------------------------
    # Ensure shared manifest exists (tooling expects this)
    # -----------------------------------------------------------------------
    if not os.path.exists(expected_path):
        try:
            os.makedirs(os.path.dirname(expected_path), exist_ok=True)
            with open(expected_path, "w") as f:
                json.dump(manifest, f, indent=2)
            _warn(
                "The adapter did not write the conventional manifest; "
                f"a copy was saved to: {expected_path}"
            )
        except Exception as e:
            _warn(f"Could not save manifest copy to {expected_path}: {e}")

    # -----------------------------------------------------------------------
    # Also write per-run manifest (seed/config-specific)
    # -----------------------------------------------------------------------
    if per_run_path and not os.path.exists(per_run_path):
        try:
            os.makedirs(os.path.dirname(per_run_path), exist_ok=True)
            with open(per_run_path, "w") as f:
                json.dump(manifest, f, indent=2)
            _info(f"Saved per-run manifest: {per_run_path}")
        except Exception as e:
            _warn(f"Could not save per-run manifest copy to {per_run_path}: {e}")

    _info(f"Synthesis complete. Manifest: {expected_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """
    Evaluate a model using the evaluator runner.

    This command assumes:
      - you already ran synth and produced a manifest at:
            <artifacts>/<model>/synthetic/manifest.json
        unless you use --no-synth (then only REAL-only metrics may be computed).

    IMPORTANT:
      - This is where summary JSON files are written by the evaluator code.
      - For auditability, ensure eval summary writer copies cfg["run_meta"].
    """
    cfg = load_config(args.config)
    
    if args.overrides:
        ov = load_config(args.overrides)
        deep_update(cfg, ov)

    cfg.setdefault("paths", {})
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts

    # Attach run_meta so evaluation pipeline can embed provenance into summary JSON
    attach_run_meta(cfg, args)

    _info(f"Evaluate model: {args.model}")
    _info(f"Config        : {args.config or '<defaults>'}")
    _info(f"Artifacts     : {artifacts_root(cfg, args.artifacts)}")
    _info(f"No-synth flag : {args.no_synth}")

    try:
        # If no_synth is False, we do not auto-generate here; run synth first.
        evaluate_model_suite(cfg, model_name=args.model, no_synth=args.no_synth)
    except FileNotFoundError as e:
        _err(str(e))
        return 2
    except Exception as e:
        _err(f"Evaluation failed: {e.__class__.__name__}: {e}")
        return 1

    return 0


def cmd_list(_: argparse.Namespace) -> int:
    """
    List registered adapters and any adapters that were skipped due to import errors.
    """
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
# CLI parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser and subcommands.

    Subcommands:
      - train : optional training router to <model>.train
      - synth : synthesis via adapter registry
      - eval  : evaluation via evaluator runner
      - list  : list registered adapters
    """
    p = argparse.ArgumentParser(
        prog="gencs",
        description="GenCyberSynth – unified CLI for training, synthesis & evaluation",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    p_t = sub.add_parser("train", help="Train the model (routes into <model>.train if available)")
    p_t.add_argument("--overrides", default=None, help="Path to a YAML overrides file")
    p_t.add_argument("--model", required=True, help="Model family (e.g., gan, diffusion, vae, ...)")
    p_t.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_t.add_argument("--artifacts", default=None, help="Override artifacts root directory")
    p_t.set_defaults(func=cmd_train)

    # synth
    p_s = sub.add_parser("synth", help="Generate synthetic images via an adapter")
    p_s.add_argument("--overrides", default=None, help="Path to a YAML overrides file")
    p_s.add_argument("--model", required=True, help="Adapter name (e.g., diffusion, gan, vae, ...)")
    p_s.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_s.add_argument("--artifacts", default=None, help="Override artifacts root directory")
    p_s.set_defaults(func=cmd_synth)

    # eval
    p_e = sub.add_parser("eval", help="Run evaluation (uses gcs-core) on latest manifest")
    p_e.add_argument("--overrides", default=None, help="Path to a YAML overrides file")
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
    """
    CLI entry point.

    Returns:
      int exit code:
        0 = success
        1 = general failure
        2 = bad config / missing resource / adapter not found
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())