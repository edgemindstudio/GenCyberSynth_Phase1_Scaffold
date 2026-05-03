#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict


def _safe_get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _extract_util(summary: Dict[str, Any]):
    ro = summary.get("utility_real_only") or summary.get("real_only") or {}
    rs = summary.get("utility_real_plus_synth") or summary.get("real_plus_synth") or {}

    ro = ro if isinstance(ro, dict) else {}
    rs = rs if isinstance(rs, dict) else {}

    return ro, rs


def _extract_deltas(summary: Dict[str, Any]):
    d = summary.get("deltas_RS_minus_R") or summary.get("deltas") or {}

    if not isinstance(d, dict):
        d = {}

    # Some summaries keep deltas at top level.
    for k in ["delta_macro_f1", "delta_bal_acc", "delta_macro_auprc"]:
        if k in summary and summary.get(k) is not None:
            d[k] = summary.get(k)

    return d


def _infer_budget_from_config(config_id: str):
    if not config_id:
        return None

    if config_id == "paper4_realonly_balanced":
        return 0

    m = re.search(r"_b(\d+)", config_id)
    if m:
        return int(m.group(1))

    return None


def _infer_seed(summary: Dict[str, Any]):
    seed = summary.get("seed")

    if seed is not None:
        return seed

    seed = _safe_get(summary, "run_meta.seed")
    if seed is not None:
        return seed

    run_id = summary.get("run_id") or ""
    m = re.search(r"_s(\d+)", run_id)
    if m:
        return int(m.group(1))

    return None


def _infer_num_fake(summary: Dict[str, Any], budget_per_class, dataset_id):
    if summary.get("num_fake") is not None:
        return summary.get("num_fake")

    counts = summary.get("counts")
    if isinstance(counts, dict) and counts.get("num_fake") is not None:
        return counts.get("num_fake")

    if summary.get("counts.num_fake") is not None:
        return summary.get("counts.num_fake")

    if budget_per_class == 0:
        return 0

    # USTC-TFC2016 has 9 classes.
    if dataset_id == "ustc_tfc2016" and budget_per_class is not None:
        return int(budget_per_class) * 9

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    art_root = Path(args.artifacts)
    out_path = Path(args.out)

    summary_paths = sorted(glob.glob(str(art_root / "*" / "summaries" / "summary_*.json")))

    rows = []

    for sp in summary_paths:
        p = Path(sp)

        try:
            with open(p, "r") as f:
                j = json.load(f)
        except Exception as e:
            print(f"[warn] skip unreadable summary {p}: {e}")
            continue

        model = p.parts[-3] if len(p.parts) >= 3 else j.get("model")

        run_meta = j.get("run_meta") if isinstance(j.get("run_meta"), dict) else {}

        config_id = j.get("config_id") or run_meta.get("config_id")
        paper_id = j.get("paper_id") or run_meta.get("paper_id")
        dataset_id = j.get("dataset_id") or run_meta.get("dataset_id")

        if paper_id is None and config_id and config_id.startswith("paper4_"):
            paper_id = "paper4"

        if dataset_id is None and config_id and config_id.startswith("paper4_"):
            dataset_id = "ustc_tfc2016"

        if paper_id != "paper4":
            continue

        ro, rs = _extract_util(j)
        deltas = _extract_deltas(j)

        budget_per_class = (
            j.get("budget_per_class")
            or run_meta.get("budget_per_class")
            or _safe_get(run_meta, "regime.synth_budget_per_class")
            or _infer_budget_from_config(config_id or "")
        )

        if config_id == "paper4_realonly_balanced":
            budget_per_class = 0

        seed = _infer_seed(j)
        num_fake = _infer_num_fake(j, budget_per_class, dataset_id)

        manifest_path = j.get("manifest_path") or run_meta.get("manifest_path")

        if config_id == "paper4_realonly_balanced":
            num_fake = 0
            manifest_path = None

        kid = j.get("kid") or j.get("metrics.kid") or _safe_get(j, "metrics.kid")
        ms_ssim = j.get("ms_ssim") or j.get("metrics.ms_ssim") or _safe_get(j, "metrics.ms_ssim")

        row = {
            "summary_file": str(p),
            "model": model,
            "run_id": j.get("run_id"),
            "paper_id": paper_id,
            "config_id": config_id,
            "dataset_id": dataset_id,
            "seed": seed,
            "budget_per_class": budget_per_class,
            "num_fake": num_fake,
            "manifest_path": manifest_path,

            "macro_f1_real_only": ro.get("macro_f1"),
            "macro_f1_real_plus_synth": rs.get("macro_f1"),
            "delta_macro_f1": deltas.get("delta_macro_f1"),

            "bal_acc_real_only": ro.get("bal_acc"),
            "bal_acc_real_plus_synth": rs.get("bal_acc"),
            "delta_bal_acc": deltas.get("delta_bal_acc"),

            "macro_auprc_real_only": ro.get("macro_auprc"),
            "macro_auprc_real_plus_synth": rs.get("macro_auprc"),
            "delta_macro_auprc": deltas.get("delta_macro_auprc"),

            "kid": kid,
            "ms_ssim": ms_ssim,
        }

        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "summary_file", "model", "run_id", "paper_id", "config_id", "dataset_id", "seed",
        "budget_per_class", "num_fake", "manifest_path",
        "macro_f1_real_only", "macro_f1_real_plus_synth", "delta_macro_f1",
        "bal_acc_real_only", "bal_acc_real_plus_synth", "delta_bal_acc",
        "macro_auprc_real_only", "macro_auprc_real_plus_synth", "delta_macro_auprc",
        "kid", "ms_ssim",
    ]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[ok] wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()