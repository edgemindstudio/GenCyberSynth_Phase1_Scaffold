# tools/aggregate_phase1.py
"""
Aggregate Phase-1 evaluation summaries from all 7 repos into CSV + JSONL.

Outputs
-------
- phase1_table.csv
- phase1_summaries.jsonl

Features
--------
- Robust schema handling for REAL+SYNTH block (accepts older names/variants).
- Coerces number-like strings to floats.
- Recomputes Δ metrics (RS - R) if missing.
- If JSON lacks images.synthetic, infers it by scanning the sibling 'synthetic/' folder.
- --diagnose prints per-repo reasons + filesystem sanity checks.

Usage
-----
python tools/aggregate_phase1.py
  --base     ~/PycharmProjects              (default)
  --outdir   .                              (default: CWD)
  --all                                      (aggregate all summaries, not just newest)
  --diagnose                                 (print detailed reasons / FS checks)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults (edit if your layout differs)
# ---------------------------------------------------------------------------
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

SUMMARY_PATTERNS: Tuple[str, ...] = (
    "artifacts/**/summaries/*_eval_summary_seed*.json",
    "artifacts/*/summaries/*_eval_summary_seed*.json",  # fallback
)

CSV_NAME = "phase1_table.csv"
JSONL_NAME = "phase1_summaries.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pick(d: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """Safely read a nested key path; returns `default` if missing or None anywhere."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def as_float_or_none(x: Any) -> Optional[float]:
    if is_num(x):
        return float(x)
    if isinstance(x, str):
        try:
            v = float(x.strip())
            return v if math.isfinite(v) else None
        except Exception:
            return None
    return None


def coerce_numeric_inplace(obj: Any) -> Any:
    """Recursively coerce number-like strings to floats inside dicts/lists."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = coerce_numeric_inplace(v)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = coerce_numeric_inplace(v)
        return obj
    f = as_float_or_none(obj)
    return f if f is not None else obj


def find_summaries(repo_dir: Path, patterns: Iterable[str]) -> List[Path]:
    """Return absolute paths of all matching summary files under `repo_dir`."""
    out: List[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for p in repo_dir.glob(pat):
            ap = p if p.is_absolute() else (repo_dir / p)
            if ap not in seen:
                seen.add(ap)
                out.append(ap)
    return out


def newest(paths: List[Path]) -> Optional[Path]:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def recompute_deltas(row: Dict[str, Any]) -> None:
    """Ensure Δ fields exist by computing RS - R wherever both sides are numeric."""
    pairs = [
        ("ΔAcc",       "Acc_RS",       "Acc_R"),
        ("ΔMacroF1",   "MacroF1_RS",   "MacroF1_R"),
        ("ΔBalAcc",    "BalAcc_RS",    "BalAcc_R"),
        ("ΔmAUPRC",    "mAUPRC_RS",    "mAUPRC_R"),
        ("ΔR@1%FPR",   "R@1%FPR_RS",   "R@1%FPR_R"),
        ("ΔECE",       "ECE_RS",       "ECE_R"),
        ("ΔBrier",     "Brier_RS",     "Brier_R"),
    ]
    for dkey, rs_key, r_key in pairs:
        if is_num(row.get(dkey)):
            continue
        a, b = as_float_or_none(row.get(rs_key)), as_float_or_none(row.get(r_key))
        row[dkey] = (a - b) if (a is not None and b is not None) else None


def rs_block(data: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], str]:
    """
    Retrieve the REAL+SYNTH metrics block using robust key fallbacks.
    Returns (block or None, key_name_used or 'missing').
    """
    candidates = [
        "utility_real_plus_synth",
        "utility_real_plus_synthetic",
        "utility_RS",
        "utility_real_and_synth",
        "util_real_plus_synth",
    ]
    for name in candidates:
        v = data.get(name)
        if isinstance(v, dict):
            return v, name
    # some writers may nest under "utility" -> "real_plus_synth"
    util = data.get("utility")
    if isinstance(util, dict):
        for name in ("real_plus_synth", "RS"):
            v = util.get(name)
            if isinstance(v, dict):
                return v, f"utility.{name}"
    return None, "missing"


def rs_status(util_rs: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Validate RS block for presence and numeric-ness."""
    if not isinstance(util_rs, dict):
        return False, "util_RS missing"
    keys = ["accuracy", "macro_f1", "balanced_accuracy", "macro_auprc",
            "recall_at_1pct_fpr", "ece", "brier"]
    missing = [k for k in keys if k not in util_rs]
    if missing:
        return False, f"util_RS missing keys: {', '.join(missing)}"
    nonnum = [k for k in keys if as_float_or_none(util_rs.get(k)) is None]
    if nonnum and len(nonnum) == len(keys):
        # likely REAL-only eval wrote an empty or null block
        return False, "util_RS present but empty/non-numeric (REAL-only?)"
    return True, ""


def infer_synth_count_from_fs(summary_path: Path) -> Optional[int]:
    """
    If JSON lacks images.synthetic, infer count by scanning the sibling synthetic/ folder:
      - x_synth.npy
      - y_synth.npy
      - gen_class_*.npy (+ optional labels_class_*.npy)
    """
    # Try to locate sibling synthetic/ based on .../summaries/<file>.json
    if "summaries" not in summary_path.parts:
        return None
    parts = list(summary_path.parts)
    try:
        idx = parts.index("summaries")
    except ValueError:
        return None

    synth_dir = Path(*parts[:idx], "synthetic")
    if not synth_dir.exists():
        return None

    # Combined dumps
    x_all = synth_dir / "x_synth.npy"
    y_all = synth_dir / "y_synth.npy"
    if x_all.exists():
        try:
            import numpy as np
            xs = np.load(x_all, mmap_mode="r")
            return int(xs.shape[0])
        except Exception:
            pass

    # Per-class dumps
    total = 0
    per_class = list(synth_dir.glob("gen_class_*.npy"))
    if per_class:
        try:
            import numpy as np
            for p in per_class:
                xs = np.load(p, mmap_mode="r")
                total += int(xs.shape[0])
            return total if total > 0 else None
        except Exception:
            return None
    return None


def columns() -> List[str]:
    return [
        "repo", "model", "seed",
        "train_real", "val_real", "test_real", "synthetic",
        "FID", "cFID_macro", "JS", "KL", "Diversity",
        "Acc_R", "MacroF1_R", "BalAcc_R", "mAUPRC_R", "R@1%FPR_R", "ECE_R", "Brier_R",
        "Acc_RS", "MacroF1_RS", "BalAcc_RS", "mAUPRC_RS", "R@1%FPR_RS", "ECE_RS", "Brier_RS",
        "ΔAcc", "ΔMacroF1", "ΔBalAcc", "ΔmAUPRC", "ΔR@1%FPR", "ΔECE", "ΔBrier",
        "_has_RS_metrics", "_rs_reason", "_rs_schema",
        "_has_FID", "_synth_fs_count",
        "_summary_relpath", "_summary_path", "_summary_mtime",
    ]


def row_from_summary(repo: str, spath_abs: Path, repo_dir: Path, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map one summary JSON into a flat CSV row with coercion, FS fallbacks, diagnostics."""
    data = coerce_numeric_inplace(raw)

    model   = pick(data, "model") or repo
    seed    = pick(data, "seed")
    counts  = pick(data, "images", default={}) or {}
    gen     = pick(data, "generative", default={}) or {}
    util_R  = pick(data, "utility_real_only", default={}) or {}
    util_RS, rs_schema = rs_block(data)
    dlt     = pick(data, "deltas_RS_minus_R", default={}) or {}

    try:
        rel = spath_abs.relative_to(repo_dir)
    except Exception:
        rel = spath_abs.name

    has_rs, rs_reason = rs_status(util_RS)

    synth_json = counts.get("synthetic")
    synth_fs   = infer_synth_count_from_fs(spath_abs) if synth_json in (None, 0) else None
    synthetic  = synth_json if synth_json not in (None, 0) else synth_fs  # prefer JSON, else FS count

    row = {
        "repo": repo,
        "model": model,
        "seed": seed,
        "train_real": counts.get("train_real"),
        "val_real": counts.get("val_real"),
        "test_real": counts.get("test_real"),
        "synthetic": synthetic,
        # Generative (VAL vs SYNTH)
        "FID": gen.get("fid"),
        "cFID_macro": gen.get("cfid_macro"),
        "JS": gen.get("js"),
        "KL": gen.get("kl"),
        "Diversity": gen.get("diversity"),
        # Utility (REAL only)
        "Acc_R": util_R.get("accuracy"),
        "MacroF1_R": util_R.get("macro_f1"),
        "BalAcc_R": util_R.get("balanced_accuracy"),
        "mAUPRC_R": util_R.get("macro_auprc"),
        "R@1%FPR_R": util_R.get("recall_at_1pct_fpr"),
        "ECE_R": util_R.get("ece"),
        "Brier_R": util_R.get("brier"),
        # Utility (REAL + SYNTH)
        "Acc_RS": None if util_RS is None else util_RS.get("accuracy"),
        "MacroF1_RS": None if util_RS is None else util_RS.get("macro_f1"),
        "BalAcc_RS": None if util_RS is None else util_RS.get("balanced_accuracy"),
        "mAUPRC_RS": None if util_RS is None else util_RS.get("macro_auprc"),
        "R@1%FPR_RS": None if util_RS is None else util_RS.get("recall_at_1pct_fpr"),
        "ECE_RS": None if util_RS is None else util_RS.get("ece"),
        "Brier_RS": None if util_RS is None else util_RS.get("brier"),
        # Deltas as provided (may be None) — recompute below
        "ΔAcc":        pick(dlt, "accuracy"),
        "ΔMacroF1":    pick(dlt, "macro_f1"),
        "ΔBalAcc":     pick(dlt, "balanced_accuracy"),
        "ΔmAUPRC":     pick(dlt, "macro_auprc"),
        "ΔR@1%FPR":    pick(dlt, "recall_at_1pct_fpr"),
        "ΔECE":        pick(dlt, "ece"),
        "ΔBrier":      pick(dlt, "brier"),
        # Diagnostics + breadcrumbs
        "_has_RS_metrics": has_rs,
        "_rs_reason": rs_reason,
        "_rs_schema": rs_schema,
        "_has_FID": is_num(gen.get("fid")),
        "_synth_fs_count": synth_fs,
        "_summary_relpath": str(rel),
        "_summary_path": str(spath_abs),
        "_summary_mtime": spath_abs.stat().st_mtime,
    }

    recompute_deltas(row)
    return row


def fmt_delta(x: Any) -> str:
    return "n/a" if as_float_or_none(x) is None else f"{float(x):+.4f}"


def print_topline(rows: List[Dict[str, Any]]) -> None:
    winners = [r for r in rows if as_float_or_none(r.get("ΔMacroF1")) is not None]
    winners.sort(
        key=lambda r: (
            float(r["ΔMacroF1"]),
            float(as_float_or_none(r.get("ΔBalAcc")) or -1e12),
        ),
        reverse=True,
    )

    print("\nTop-line (by ΔMacroF1):")
    if not winners:
        print("  (no repos had a numeric ΔMacroF1)")
    else:
        for r in winners[:5]:
            print(
                f"  {r['model']:>28s}  "
                f"ΔF1={fmt_delta(r.get('ΔMacroF1'))}  "
                f"ΔBalAcc={fmt_delta(r.get('ΔBalAcc'))}  "
                f"ΔAUPRC={fmt_delta(r.get('ΔmAUPRC'))}  "
                f"ΔECE={fmt_delta(r.get('ΔECE'))}  "
                f"FID={(r.get('FID') if is_num(r.get('FID')) else 'n/a')}"
            )

    missing = [r["model"] for r in rows if as_float_or_none(r.get("ΔMacroF1")) is None]
    if missing:
        print("Note: no synthetic delta for -> " + ", ".join(sorted(set(missing))))


def print_diagnostics(rows: List[Dict[str, Any]]) -> None:
    print("\nDiagnostics:")
    header = (
        f"{'repo/model':34s} {'synthetic':>9s} {'has_RS':>7s} {'has_FID':>8s} "
        f"{'rs_key':>18s}  reason / summary"
    )
    print(header)
    for r in rows:
        synth = r.get("synthetic")
        has_rs = bool(r.get("_has_RS_metrics"))
        has_fid = bool(r.get("_has_FID"))
        rs_key = (r.get("_rs_schema") or "missing")[:18]
        reason = r.get("_rs_reason") or ""
        print(
            f"{(r['repo'] + ' / ' + r['model'])[:34].ljust(34)}"
            f"{str(synth).rjust(9)}{str(has_rs).rjust(7)}{str(has_fid).rjust(8)} "
            f"{rs_key.rjust(18)}  {reason}  [{r['_summary_relpath']}]"
        )
    print(
        "\nIf synthetic==None/0 or has_RS==False, your eval likely ran REAL-only or could not "
        "find per-class dumps (gen_class_<k>.npy + labels_class_<k>.npy) or combined x_synth.npy/y_synth.npy. "
        "Also ensure every repo uses the same updated eval/val_common.py (writes the RS block)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate Phase-1 evaluation results across repos.")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Base directory containing the repos")
    p.add_argument("--outdir", type=Path, default=Path.cwd(), help="Directory to write CSV/JSONL")
    p.add_argument("--all", action="store_true", help="Include all summaries per repo (default: only newest)")
    p.add_argument("--diagnose", action="store_true", help="Print per-repo diagnostics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base: Path = args.base
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    jsonl_lines: List[str] = []

    for repo in DEFAULT_REPOS:
        repo_dir = base / repo
        paths = find_summaries(repo_dir, SUMMARY_PATTERNS)
        if not paths:
            print(f"[skip] No summaries found in {repo_dir}")
            continue

        use_paths = paths if args.all else [newest(paths)]  # type: ignore[list-item]
        for spath_abs in use_paths:
            if spath_abs is None:
                continue
            try:
                raw = json.loads(Path(spath_abs).read_text())
            except Exception as e:
                print(f"[warn] Failed to read {spath_abs}: {e}")
                continue

            row = row_from_summary(repo, spath_abs, repo_dir, raw)
            rows.append(row)

            # Keep original JSON (not coerced) in JSONL for traceability
            jsonl_lines.append(json.dumps({"repo": repo, **raw}, separators=(",", ":")))

    if not rows:
        print("[warn] No summaries found across the listed repos.")
        return

    # Sort by ΔMacroF1 then ΔBalAcc (missing treated as very small)
    def sort_key(r: Dict[str, Any]) -> Tuple[float, float]:
        dmf = as_float_or_none(r.get("ΔMacroF1")) or -1e12
        dba = as_float_or_none(r.get("ΔBalAcc")) or -1e12
        return (dmf, dba)

    rows.sort(key=sort_key, reverse=True)

    # Write CSV
    csv_path = outdir / CSV_NAME
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Write JSONL
    jsonl_path = outdir / JSONL_NAME
    with open(jsonl_path, "w") as f:
        f.write("\n".join(jsonl_lines))

    print(f"[ok] wrote {csv_path} and {jsonl_path}")

    # Console topline + optional diagnostics
    print_topline(rows)
    if args.diagnose:
        print_diagnostics(rows)


if __name__ == "__main__":
    main()
