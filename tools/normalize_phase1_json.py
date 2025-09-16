#!/usr/bin/env python3
"""
Normalize/repair Phase-1 summary JSONs so aggregate_phase1.py can read
RS metrics and FID/cFID consistently.

What this does per repo:
- Picks the newest *_eval_summary_seed*.json under artifacts/**/summaries/
- Ensures an RS block under "utility_real_plus_synth"
  - Accepts older keys: "real_plus_synth", "utility_real_plus_", "util_RS", etc.
- Normalizes utility metric names:
  - "bal_acc" -> "balanced_accuracy"
- Ensures generative.fid exists (fallback to generative.fid_macro if needed)
- (Optional) infers images.synthetic from x_synth.npy or per-class dumps
- Recomputes deltas_RS_minus_R when both sides are numeric

Usage:
  python tools/normalize_phase1_json.py --write            # apply changes
  python tools/normalize_phase1_json.py --dry-run          # show diffs only
  python tools/normalize_phase1_json.py --infer-synth      # also fill images.synthetic
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, glob
from typing import Dict, Optional, Tuple, Any, List

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
GLOBS = [
    "artifacts/**/summaries/*_eval_summary_seed*.json",
    "artifacts/*/summaries/*_eval_summary_seed*.json",
]

REQ_RS_KEYS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "macro_auprc",
    "recall_at_1pct_fpr",
    "ece",
    "brier",
]

def _find_newest_summary(repo_dir: Path) -> Optional[Path]:
    cands: List[Path] = []
    for pat in GLOBS:
        cands.extend(repo_dir.glob(pat))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def _first_key(d: Dict[str, Any], prefixes: Tuple[str, ...]) -> Optional[str]:
    for k in d.keys():
        for pf in prefixes:
            if k.startswith(pf):
                return k
    return None

def _normalize_util_block(blk: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(blk, dict):
        return {}
    out = dict(blk)  # shallow copy
    # normalize balanced accuracy name
    if "balanced_accuracy" not in out and "bal_acc" in out:
        out["balanced_accuracy"] = out.get("bal_acc")
    return out

def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        return None if (a is None or b is None) else float(a - b)
    except Exception:
        return None

def _infer_synth_count(synth_dir: Path) -> Optional[int]:
    if not synth_dir.exists():
        return None
    x_s = synth_dir / "x_synth.npy"
    if x_s.exists():
        try:
            import numpy as np
            return int(np.load(x_s, mmap_mode="r").shape[0])
        except Exception:
            pass
    # fallback: sum per-class arrays
    try:
        import numpy as np
        total = 0
        pcs = list(synth_dir.glob("gen_class_*.npy"))
        if not pcs:
            return None
        for p in pcs:
            try:
                total += int(np.load(p, mmap_mode="r").shape[0])
            except Exception:
                continue
        return total if total > 0 else None
    except Exception:
        return None

def normalize_one(repo_dir: Path, infer_synth: bool=False) -> Tuple[Optional[Path], Dict[str, Tuple[Any, Any]], Dict[str, Any]]:
    """
    Returns:
      (summary_path, changes_dict, new_dict)
      changes_dict maps json_pointer-ish keys -> (old, new)
    """
    spath = _find_newest_summary(repo_dir)
    if spath is None:
        return None, {}, {}

    data = json.loads(spath.read_text())

    changes: Dict[str, Tuple[Any, Any]] = {}
    def setp(ptr: str, old, new):
        if old != new:
            changes[ptr] = (old, new)

    # ---- Ensure images.synthetic is filled (optional) ----
    images = data.get("images", {}) if isinstance(data.get("images"), dict) else {}
    if infer_synth and images.get("synthetic") in (None, 0):
        # guess synth dir under artifacts/*/synthetic
        syn_dirs = list(repo_dir.glob("artifacts/*/synthetic"))
        s_count = None
        for sd in syn_dirs:
            s_count = _infer_synth_count(sd)
            if s_count:
                break
        if s_count is not None:
            old = images.get("synthetic")
            images["synthetic"] = int(s_count)
            setp("/images/synthetic", old, images["synthetic"])
            data["images"] = images

    # ---- Normalize generative ----
    gen = data.get("generative", {}) if isinstance(data.get("generative"), dict) else {}
    if "fid" not in gen and "fid_macro" in gen:
        old = None
        gen["fid"] = gen.get("fid_macro")
        setp("/generative/fid", old, gen["fid"])
    data["generative"] = gen

    # ---- Normalize RS block name and keys ----
    rs_blk = None
    # 1) canonical
    if isinstance(data.get("utility_real_plus_synth"), dict):
        rs_blk = data["utility_real_plus_synth"]
    else:
        # 2) common aliases or accidental truncations
        alt_key = _first_key(data, ("utility_real_plus_", "real_plus_synth", "util_RS"))
        if alt_key and isinstance(data.get(alt_key), dict):
            rs_blk = data[alt_key]
            # move under canonical key
            old = None
            data["utility_real_plus_synth"] = rs_blk
            setp("/utility_real_plus_synth", old, "moved_from:"+alt_key)
            try:
                del data[alt_key]
            except Exception:
                pass

    # map names inside RS
    if isinstance(rs_blk, dict):
        norm = _normalize_util_block(rs_blk)
        if norm != rs_blk:
            setp("/utility_real_plus_synth(keys)", sorted(rs_blk.keys()), sorted(norm.keys()))
        data["utility_real_plus_synth"] = norm

    # normalize REAL-only block for consistency (not strictly required)
    r_blk = data.get("utility_real_only")
    if isinstance(r_blk, dict):
        norm_r = _normalize_util_block(r_blk)
        if norm_r != r_blk:
            setp("/utility_real_only(keys)", sorted(r_blk.keys()), sorted(norm_r.keys()))
        data["utility_real_only"] = norm_r

    # ---- Recompute deltas when both sides present ----
    rs = data.get("utility_real_plus_synth") if isinstance(data.get("utility_real_plus_synth"), dict) else None
    rr = data.get("utility_real_only") if isinstance(data.get("utility_real_only"), dict) else None

    if isinstance(rs, dict) and isinstance(rr, dict):
        deltas = {
            "accuracy":           _delta(rs.get("accuracy"),           rr.get("accuracy")),
            "macro_f1":           _delta(rs.get("macro_f1"),           rr.get("macro_f1")),
            "balanced_accuracy":  _delta(rs.get("balanced_accuracy"),  rr.get("balanced_accuracy")),
            "macro_auprc":        _delta(rs.get("macro_auprc"),        rr.get("macro_auprc")),
            "recall_at_1pct_fpr": _delta(rs.get("recall_at_1pct_fpr"), rr.get("recall_at_1pct_fpr")),
            "ece":                _delta(rs.get("ece"),                rr.get("ece")),
            "brier":              _delta(rs.get("brier"),              rr.get("brier")),
        }
        old = data.get("deltas_RS_minus_R")
        data["deltas_RS_minus_R"] = deltas
        setp("/deltas_RS_minus_R", old, deltas)

    return spath, changes, data

def main():
    ap = argparse.ArgumentParser(description="Normalize Phase-1 summary JSONs for aggregation.")
    ap.add_argument("--base", default=str(DEFAULT_BASE), help="Base directory containing the 7 repos")
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS, help="Subset of repos to process")
    ap.add_argument("--write", action="store_true", help="Apply changes in-place")
    ap.add_argument("--dry-run", action="store_true", help="Show intended changes without writing")
    ap.add_argument("--infer-synth", action="store_true", help="Backfill images.synthetic by counting files")
    args = ap.parse_args()

    base = Path(args.base)
    print(f"[normalize] base={base}  repos={', '.join(args.repos)}  write={args.write}  infer_synth={args.infer_synth}")

    any_changes = False
    for name in args.repos:
        repo_dir = base / name
        spath, changes, new_data = normalize_one(repo_dir, infer_synth=args.infer_synth)
        tag = f"[{name:<18}]"
        if spath is None:
            print(f"{tag} no summary found")
            continue
        if not changes:
            print(f"{tag} {spath.relative_to(repo_dir)}  (ok; no changes)")
            continue

        any_changes = True
        print(f"{tag} {spath.relative_to(repo_dir)}")
        for k, (old, new) in changes.items():
            print(f"   - fix {k}: {old!r} -> {new!r}")

        if args.write and not args.dry_run:
            spath.write_text(json.dumps(new_data, separators=(',', ':'), indent=2))
            print(f"   -> wrote {spath}")

    if not any_changes:
        print("[normalize] nothing to do.")

if __name__ == "__main__":
    main()
