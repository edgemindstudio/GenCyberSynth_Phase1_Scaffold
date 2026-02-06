#!/usr/bin/env python3
# scripts/tuning_dashboard.py
"""
Compact status dashboard for GenCyberSynth tuning-lite grid.

Checks per (MODEL, CFG, SEED):
  - Phase B manifest exists:
      artifacts/<model>/synthetic/<model>_<CFG>_seed<SEED>/manifest.json
  - Phase C done flag exists:
      artifacts/<model>/summaries/done_<model>_<CFG>_seed<SEED>.txt
  - Latest eval summary (best-effort): finds newest summary_*.json containing run_meta.manifest_path
  - Prints a compact table + totals.

Usage:
  python scripts/tuning_dashboard.py \
    --artifacts /home/bruno.fonkeng/gencys/artifacts \
    --models gan vae diffusion autoregressive restrictedboltzmann gaussianmixture maskedautoflow \
    --cfgs A B \
    --seeds 42 43 44 \
    --show-missing-only

Tip:
  Run periodically while jobs run. It is read-only.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple


@dataclass
class RunStatus:
    model: str
    cfg: str
    seed: int
    manifest_path: Path
    done_flag: Path
    has_manifest: bool
    has_done: bool
    latest_summary: Optional[Path]
    has_summary: bool
    run_id: Optional[str]
    budget_per_class: Optional[int]


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _find_latest_summary_for_model(model_summaries_dir: Path, expected_manifest: Path) -> Optional[Path]:
    """
    Best-effort: among summary_*.json in model summaries dir, return newest file
    whose .run_meta.manifest_path or .manifest_path matches expected_manifest.
    """
    candidates = sorted(model_summaries_dir.glob("summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    expected = str(expected_manifest)
    for p in candidates[:200]:  # cap scanning for speed
        try:
            with p.open("r") as f:
                obj = json.load(f)
            mp = obj.get("manifest_path")
            rmp = None
            rm = obj.get("run_meta")
            if isinstance(rm, dict):
                rmp = rm.get("manifest_path")
            if mp == expected or rmp == expected:
                return p
        except Exception:
            continue
    return None


def _read_summary_fields(p: Path) -> Tuple[Optional[str], Optional[int]]:
    try:
        with p.open("r") as f:
            obj = json.load(f)
        run_id = obj.get("run_id")
        bpc = obj.get("budget_per_class")
        rm = obj.get("run_meta")
        if bpc is None and isinstance(rm, dict):
            bpc = rm.get("budget_per_class")
        return (run_id if isinstance(run_id, str) else None, _safe_int(bpc))
    except Exception:
        return (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="Artifacts root, e.g. /home/bruno.fonkeng/gencys/artifacts")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cfgs", nargs="+", default=["A", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--show-missing-only", action="store_true", help="Show only incomplete runs")
    ap.add_argument("--wide", action="store_true", help="Wider columns (more verbose)")
    args = ap.parse_args()

    arts = Path(args.artifacts).expanduser().resolve()
    models: List[str] = args.models
    cfgs: List[str] = args.cfgs
    seeds: List[int] = args.seeds

    rows: List[RunStatus] = []

    for model in models:
        model_root = arts / model
        synth_root = model_root / "synthetic"
        summaries_dir = model_root / "summaries"

        for cfg in cfgs:
            for seed in seeds:
                run_dir = synth_root / f"{model}_{cfg}_seed{seed}"
                manifest = run_dir / "manifest.json"
                done_flag = summaries_dir / f"done_{model}_{cfg}_seed{seed}.txt"

                has_manifest = manifest.exists()
                has_done = done_flag.exists()

                latest_summary = None
                has_summary = False
                run_id = None
                bpc = None

                if summaries_dir.exists() and has_manifest:
                    latest_summary = _find_latest_summary_for_model(summaries_dir, manifest)
                    if latest_summary is not None and latest_summary.exists():
                        has_summary = True
                        run_id, bpc = _read_summary_fields(latest_summary)

                rows.append(
                    RunStatus(
                        model=model,
                        cfg=cfg,
                        seed=seed,
                        manifest_path=manifest,
                        done_flag=done_flag,
                        has_manifest=has_manifest,
                        has_done=has_done,
                        latest_summary=latest_summary,
                        has_summary=has_summary,
                        run_id=run_id,
                        budget_per_class=bpc,
                    )
                )

    # Totals
    total = len(rows)
    n_manifest = sum(r.has_manifest for r in rows)
    n_summary = sum(r.has_summary for r in rows)
    n_done = sum(r.has_done for r in rows)

    # Print
    print("=" * 88)
    print(f"[dashboard] artifacts={arts}")
    print(f"[dashboard] grid = {len(models)} models × {len(cfgs)} cfgs × {len(seeds)} seeds = {total} runs")
    print(f"[dashboard] manifest: {n_manifest}/{total} | summary: {n_summary}/{total} | done: {n_done}/{total}")
    print("=" * 88)

    # Header
    if args.wide:
        print(f"{'MODEL':<18} {'CFG':<3} {'SEED':<5} {'MAN':<3} {'SUM':<3} {'DONE':<4} {'BPC':<4} {'RUN_ID':<30} {'SUMMARY_FILE'}")
    else:
        print(f"{'MODEL':<18} {'CFG':<3} {'SEED':<5} {'MAN':<3} {'SUM':<3} {'DONE':<4} {'BPC':<4} {'RUN_ID'}")

    def mark(b: bool) -> str:
        return "Y" if b else "-"

    shown = 0
    for r in rows:
        incomplete = not (r.has_manifest and r.has_summary and r.has_done)
        if args.show_missing_only and not incomplete:
            continue

        bpc = str(r.budget_per_class) if r.budget_per_class is not None else "-"
        rid = r.run_id or "-"

        if args.wide:
            sf = str(r.latest_summary) if r.latest_summary else "-"
            print(f"{r.model:<18} {r.cfg:<3} {r.seed:<5} {mark(r.has_manifest):<3} {mark(r.has_summary):<3} {mark(r.has_done):<4} {bpc:<4} {rid:<30} {sf}")
        else:
            print(f"{r.model:<18} {r.cfg:<3} {r.seed:<5} {mark(r.has_manifest):<3} {mark(r.has_summary):<3} {mark(r.has_done):<4} {bpc:<4} {rid}")

        shown += 1

    if args.show_missing_only:
        print("-" * 88)
        print(f"[dashboard] showing incomplete only: {shown}/{total} rows displayed")

    # Summary by model
    print("=" * 88)
    print("[dashboard] per-model completion (done flags):")
    for model in models:
        mrows = [r for r in rows if r.model == model]
        mdone = sum(r.has_done for r in mrows)
        print(f"  - {model:<18}: {mdone:>2}/{len(mrows)} done")
    print("=" * 88)


if __name__ == "__main__":
    main()
