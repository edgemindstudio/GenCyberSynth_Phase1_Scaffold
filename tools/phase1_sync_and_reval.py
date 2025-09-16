#!/usr/bin/env python3
"""
Sync the canonical eval/val_common.py into each Phase-1 repo and re-run eval.

What this does
--------------
1) Copies GenCyberSynth_Phase1_Scaffold/eval/val_common.py into <REPO>/eval/val_common.py
   (only if content differs; hash-checked).
2) Re-runs `python -m app.main eval` in each repo using the best Python:
   - Prefer <repo>/.venv/bin/python (or venv/bin/python) when --prefer-repo-venv is set.
   - Otherwise fall back to the current interpreter (sys.executable).
   - Sets MPLBACKEND=Agg so matplotlib never needs a display.
3) Prints, for each repo, whether RS and FID are present after re-eval.

Typical usage
-------------
# Use each repo's own venvs for eval (recommended):
python tools/phase1_sync_and_reval.py --reval --prefer-repo-venv

# If you want to keep using the scaffold env, first install deps:
python -m pip install -U scikit-learn matplotlib
python tools/phase1_sync_and_reval.py --reval
"""

from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
from typing import Dict, List, Optional, Tuple

DEFAULT_BASE = Path.home() / "PycharmProjects"
DEFAULT_REPOS = [
    "GAN",
    "VAEs",
    "AUTOREGRESSIVE",
    "MASKEDAUTOFLOW",
    "RESTRICTEDBOLTZMANN",
    "GAUSSIANMIXTURE",
    "DIFFUSION",
]

SUMMARY_PATTERNS = [
    "artifacts/**/summaries/*_eval_summary_seed*.json",
    "artifacts/*/summaries/*_eval_summary_seed*.json",
]

# --------------------------------------------------------------------------------------
# Small utils
# --------------------------------------------------------------------------------------

def sha256_of_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text()
    except Exception:
        return None

def write_text(p: Path, s: str) -> bool:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s)
        return True
    except Exception as e:
        print(f"[err] write failed: {p} ({e})")
        return False

def newest_summary(repo_dir: Path) -> Optional[Path]:
    cands: List[Path] = []
    for pat in SUMMARY_PATTERNS:
        cands.extend(repo_dir.glob(pat))
    if not cands:
        return None
    return max(cands, key=lambda q: q.stat().st_mtime)

def read_json(p: Path) -> Optional[Dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def has_rs_and_fid(data: Dict) -> Tuple[bool, bool, str]:
    """
    Return (has_RS, has_FID, reason_or_ok).
    RS block must be "utility_real_plus_synth" with all required fields.
    """
    gen = data.get("generative") or {}
    fid = gen.get("fid")
    has_fid = (fid is not None)

    rs = data.get("utility_real_plus_synth")
    if not isinstance(rs, dict):
        return (False, has_fid, "util_RS missing")

    needed = ["accuracy","macro_f1","balanced_accuracy","macro_auprc","recall_at_1pct_fpr","ece","brier"]
    missing = [k for k in needed if rs.get(k) is None]
    if missing:
        return (False, has_fid, f"util_RS missing keys: {', '.join(missing)}")
    return (True, has_fid, "ok")

def detect_repo_python(repo_dir: Path) -> Optional[str]:
    """
    Return a repo-local Python if a venv exists, else None.
    Checks .venv and venv, typical macOS/Linux layouts.
    """
    candidates = [
        repo_dir / ".venv" / "bin" / "python",
        repo_dir / "venv"  / "bin" / "python",
        repo_dir / ".venv" / "Scripts" / "python.exe",  # Windows
        repo_dir / "venv"  / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

def check_deps(py_exec: str) -> Tuple[bool, str]:
    """
    Quick dependency probe for sklearn + matplotlib in the chosen interpreter.
    Returns (ok, hint).
    """
    code = "import sys; import importlib; mods=['sklearn','matplotlib'];" \
           "missing=[m for m in mods if importlib.util.find_spec(m) is None];" \
           "sys.stdout.write(','.join(missing))"
    try:
        proc = subprocess.run([py_exec, "-c", code], capture_output=True, text=True)
        missing = (proc.stdout or "").strip()
        if proc.returncode != 0:
            return False, "dependency probe failed"
        if missing:
            hint = f"missing: {missing} (try: {py_exec} -m pip install -U {missing.replace(',', ' ')})"
            return False, hint
        return True, "ok"
    except Exception as e:
        return False, f"probe error: {e}"

def run_eval(repo_dir: Path, python_exec: str, dry_run: bool=False) -> int:
    """
    Run `python -m app.main eval` in repo_dir with a safe MPL backend.
    """
    cmd = [python_exec, "-m", "app.main", "eval"]
    if dry_run:
        print(f"[dry] would run: (cd {repo_dir}) {' '.join(cmd)}")
        return 0
    env = os.environ.copy()
    # Ensure matplotlib never needs a GUI/display during import in sample.py files.
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(cmd, cwd=str(repo_dir), env=env)
    return proc.returncode

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Sync canonical eval/val_common.py into repos and re-evaluate summaries.")
    ap.add_argument("--base", default=str(DEFAULT_BASE), help="Base directory containing the 7 repos")
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS, help="Subset of repos to process")
    ap.add_argument("--val-common", default=str(Path(__file__).resolve().parents[1] / "eval" / "val_common.py"),
                    help="Path to canonical eval/val_common.py to copy into each repo")
    ap.add_argument("--reval", action="store_true", help="Re-run eval in each repo after syncing")
    ap.add_argument("--skip-sync", action="store_true", help="Do not copy the file; only re-eval")
    ap.add_argument("--prefer-repo-venv", action="store_true",
                    help="Use <repo>/.venv (or venv) Python if available for running eval")
    ap.add_argument("--python", default=None,
                    help="Explicit python interpreter to use for eval (overrides prefer-repo-venv).")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without writing/subprocess")
    args = ap.parse_args()

    base = Path(args.base)
    canon_path = Path(args.val_common)
    canon_text = read_text(canon_path)
    if canon_text is None:
        print(f"[fatal] cannot read canonical file: {canon_path}")
        sys.exit(2)
    canon_hash = sha256_of_text(canon_text)

    print(f"[sync] base={base}  repos={', '.join(args.repos)}")
    print(f"[sync] canonical: {canon_path}  sha256={canon_hash[:16]}  "
          f"reval={args.reval}  skip_sync={args.skip_sync}  prefer_repo_venv={args.prefer_repo_venv}  dry={args.dry_run}")

    for name in args.repos:
        repo_dir = base / name
        tag = f"[{name:<18}]"
        dst = repo_dir / "eval" / "val_common.py"

        # 1) Sync file if requested
        if not args.skip_sync:
            dst_text = read_text(dst)
            dst_hash = sha256_of_text(dst_text) if dst_text is not None else None
            if dst_hash == canon_hash:
                print(f"{tag} eval/val_common.py already up-to-date ({dst_hash[:16]})")
            else:
                if args.dry_run:
                    print(f"{tag} would update eval/val_common.py -> {dst_hash or 'NONE'} -> {canon_hash[:16]}")
                else:
                    ok = write_text(dst, canon_text)
                    print(f"{tag} {'updated' if ok else 'FAILED to write'} eval/val_common.py -> {canon_hash[:16]}")

        # 2) Choose interpreter for eval
        py_exec = args.python
        if not py_exec and args.prefer_repo_venv:
            py_exec = detect_repo_python(repo_dir)
        if not py_exec:
            py_exec = sys.executable

        # 3) Optional: dependency probe before reval
        ok_deps, hint = check_deps(py_exec)
        if not ok_deps:
            print(f"{tag} dependency check: {hint}")

        # 4) Re-run eval if requested
        if args.reval:
            rc = run_eval(repo_dir, py_exec, dry_run=args.dry_run)
            if rc != 0:
                print(f"{tag} eval returned non-zero code: {rc}")

        # 5) Inspect newest summary
        sp = newest_summary(repo_dir)
        if sp is None:
            print(f"{tag} no summary found")
            continue
        data = read_json(sp)
        if data is None:
            print(f"{tag} could not parse JSON: {sp}")
            continue

        has_rs, has_fid, reason = has_rs_and_fid(data)
        gen = data.get("generative") or {}
        fid = gen.get("fid")
        cfid = gen.get("cfid_macro")
        synth = (data.get("images") or {}).get("synthetic")
        print(f"{tag} {sp.relative_to(repo_dir)} :: RS={has_rs} ({reason})  "
              f"FID={'ok' if has_fid else 'missing'}  synth={synth}  fid={fid}  cfid_macro={cfid}")

    print("\nNext:")
    print("  • Re-run the aggregator:")
    print("      python tools/aggregate_phase1.py --diagnose")
    print("  • Inspect phase1_table.csv for Δ* columns and FID/cFID.")
    print("  • If some repos still miss deps, either:")
    print("      - Install centrally:  python -m pip install -U scikit-learn matplotlib")
    print("      - Or re-run with repo venvs:  --reval --prefer-repo-venv")
    print("      - Or specify interpreter explicitly:  --python /path/to/python")
    print()

if __name__ == "__main__":
    main()
