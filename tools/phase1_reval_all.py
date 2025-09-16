# tools/phase1_reval_all.py
"""
Re-run Phase-1 evaluations for repos that are missing REAL+SYNTH (RS) metrics.

Why?
-----
If your JSON summaries were produced before you standardized `eval/val_common.py`,
they may lack the 'utility_real_plus_synth' block. This script:
  1) Scans each repo for the newest summary.
  2) Detects whether RS metrics are present.
  3) Optionally re-runs:  python -m app.main eval  (ensuring synth is included)
     using the repo's local virtualenv if available.
  4) Re-checks that RS now exists and reports status.

Usage
-----
python tools/phase1_reval_all.py --diagnose
python tools/phase1_reval_all.py --fix-missing-rs
  [--base ~/PycharmProjects]
  [--repos GAN VAEs AUTOREGRESSIVE MASKEDAUTOFLOW RESTRICTEDBOLTZMANN GAUSSIANMIXTURE DIFFUSION]

Notes
-----
- This DOES NOT modify code in repos; it simply runs the 'eval' step.
- It assumes each repo writes summaries to artifacts/*/summaries/.
- It will prefer <repo>/.venv/bin/python if present; otherwise uses current 'python'.
- After running --fix-missing-rs, re-run your aggregator.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
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

SUMMARY_PATTERNS = (
    "artifacts/**/summaries/*_eval_summary_seed*.json",
    "artifacts/*/summaries/*_eval_summary_seed*.json",
)

# ---------------------------------------------------------------------------

@dataclass
class RepoSummary:
    repo: str
    summary_path: Optional[Path]
    has_rs: bool
    reason: str

# ---------------------------------------------------------------------------

def find_newest_summary(repo_dir: Path) -> Optional[Path]:
    cand: List[Path] = []
    for pat in SUMMARY_PATTERNS:
        cand.extend(repo_dir.glob(pat))
    if not cand:
        return None
    return max(cand, key=lambda p: p.stat().st_mtime)

def load_json(p: Path) -> Optional[Dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def rs_block(data: Dict) -> Tuple[Optional[Dict], str]:
    # accept several schema variants
    for k in (
        "utility_real_plus_synth",
        "utility_real_plus_synthetic",
        "utility_RS",
        "utility_real_and_synth",
        "util_real_plus_synth",
    ):
        v = data.get(k)
        if isinstance(v, dict):
            return v, k
    util = data.get("utility")
    if isinstance(util, dict):
        for k in ("real_plus_synth", "RS"):
            v = util.get(k)
            if isinstance(v, dict):
                return v, f"utility.{k}"
    return None, "missing"

def rs_status(util_rs: Optional[Dict]) -> Tuple[bool, str]:
    if not isinstance(util_rs, dict):
        return False, "util_RS missing"
    keys = ["accuracy", "macro_f1", "balanced_accuracy", "macro_auprc",
            "recall_at_1pct_fpr", "ece", "brier"]
    miss = [k for k in keys if k not in util_rs]
    if miss:
        return False, f"util_RS missing keys: {', '.join(miss)}"
    # if all values are None/empty, it's effectively missing (REAL-only run)
    if all(util_rs.get(k) in (None, "", []) for k in keys):
        return False, "util_RS present but empty (REAL-only?)"
    return True, ""

def detect_repo_python(repo_dir: Path) -> List[str]:
    """
    Prefer <repo>/.venv/bin/python if it exists; else fallback to sys.executable or 'python'.
    Returns an argv prefix, e.g. ['/path/to/python'].
    """
    venv_py = repo_dir / ".venv" / "bin" / "python"
    if venv_py.exists():
        return [str(venv_py)]
    # sys.executable gives current interpreter; useful if you installed deps globally
    if sys.executable:
        return [sys.executable]
    return ["python"]

def run_eval(repo_dir: Path, config: Optional[str] = None) -> Tuple[int, str]:
    """
    Run: python -m app.main eval  [--config <config>]  (with synth enabled)
    Returns (returncode, combined_stdout_stderr).
    """
    py = detect_repo_python(repo_dir)
    cmd = py + ["-m", "app.main", "eval"]
    if config:
        cmd += ["--config", config]
    # Important: DO NOT pass --no-synth (we want RS metrics)
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = proc.communicate()
    return proc.returncode, out

def check_repo(repo: str, base: Path) -> RepoSummary:
    repo_dir = base / repo
    summ = find_newest_summary(repo_dir)
    if not summ:
        return RepoSummary(repo, None, False, "no summaries found")
    data = load_json(summ)
    if not isinstance(data, dict):
        return RepoSummary(repo, summ, False, "failed to parse JSON")
    util_rs, _ = rs_block(data)
    ok, reason = rs_status(util_rs)
    return RepoSummary(repo, summ, ok, reason)

# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-run eval for repos missing RS metrics")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Base folder with all repos")
    p.add_argument("--repos", nargs="*", default=DEFAULT_REPOS, help="Subset of repos to process")
    p.add_argument("--diagnose", action="store_true", help="Only print current RS status")
    p.add_argument("--fix-missing-rs", action="store_true", help="Re-run eval for repos missing RS")
    p.add_argument("--config", default=None, help="Optional --config path to pass to each repo")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    base: Path = args.base

    # 1) Initial scan
    summaries: List[RepoSummary] = []
    for repo in args.repos:
        res = check_repo(repo, base)
        summaries.append(res)

    print("\nCurrent RS status:")
    print(f"{'repo':22s} {'has_RS':>7s}  reason / summary")
    for r in summaries:
        sp = str(r.summary_path.relative_to(base / r.repo)) if r.summary_path else "(none)"
        print(f"{r.repo:22s} {str(r.has_rs):>7s}  {r.reason}  [{sp}]")

    if args.diagnose and not args.fix_missing_rs:
        return

    # 2) Re-run eval for missing RS
    to_fix = [r for r in summaries if not r.has_rs]
    if not to_fix:
        print("\nAll repos already have RS metrics. Nothing to do.")
        return

    print("\nRe-running eval for repos missing RS metrics...")
    for r in to_fix:
        repo_dir = base / r.repo
        rc, out = run_eval(repo_dir, config=args.config)
        print(f"\n=== {r.repo} :: return_code={rc} ===")
        print(out.strip() or "(no output)")

    # 3) Re-check and report
    print("\nRe-checking summaries...")
    summaries2: List[RepoSummary] = []
    for repo in args.repos:
        res = check_repo(repo, base)
        summaries2.append(res)

    print(f"\n{'repo':22s} {'has_RS':>7s}  reason / summary (after re-eval)")
    for r in summaries2:
        sp = str(r.summary_path.relative_to(base / r.repo)) if r.summary_path else "(none)"
        print(f"{r.repo:22s} {str(r.has_rs):>7s}  {r.reason}  [{sp}]")

    missing_after = [r.repo for r in summaries2 if not r.has_rs]
    if missing_after:
        print(
            "\nStill missing RS metrics for: " + ", ".join(missing_after) +
            "\n- Ensure synthetic files exist under artifacts/<repo>/synthetic/"
            "\n  (either x_synth.npy/y_synth.npy OR per-class gen_class_<k>.npy + labels_class_<k>.npy)."
            "\n- Make sure each repo is importing the UPDATED eval/val_common.py."
            "\n  Then re-run: python -m app.main eval"
        )
    else:
        print("\nSuccess: RS metrics present for all processed repos. Now re-run the aggregator:\n"
              "  python tools/aggregate_phase1.py --diagnose")

if __name__ == "__main__":
    main()
