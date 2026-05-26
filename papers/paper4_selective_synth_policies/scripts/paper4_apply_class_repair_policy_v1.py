#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.load(open(path))


def extract_real_only_per_class_f1(summary: Dict[str, Any]) -> List[float]:
    ro = summary.get("real_only") or summary.get("utility_real_only") or {}
    if not isinstance(ro, dict):
        raise RuntimeError("Could not find real_only/utility_real_only block in summary.")
    pc = ro.get("per_class") or {}
    if not isinstance(pc, dict):
        raise RuntimeError("Could not find per_class block in real-only metrics.")
    f1 = pc.get("f1")
    if not isinstance(f1, list) or len(f1) == 0:
        raise RuntimeError("Could not find real-only per-class f1 list.")
    return [float(x) for x in f1]


def compute_repair_allocation(
    f1: List[float],
    total_budget: int,
    base_per_class: int,
) -> Dict[int, int]:
    k = len(f1)
    base_total = base_per_class * k
    if base_total > total_budget:
        raise ValueError(f"base_per_class too large: base_total={base_total}, total_budget={total_budget}")

    remaining = total_budget - base_total
    max_f1 = max(f1)
    weakness = np.asarray([max(0.0, max_f1 - x) for x in f1], dtype="float64")

    alloc = np.asarray([base_per_class] * k, dtype="int64")

    if remaining > 0:
        if weakness.sum() <= 0:
            weights = np.ones(k, dtype="float64") / k
        else:
            weights = weakness / weakness.sum()

        raw_extra = weights * remaining
        extra_floor = np.floor(raw_extra).astype("int64")
        alloc += extra_floor

        leftover = int(remaining - extra_floor.sum())
        if leftover > 0:
            frac_order = np.argsort(-(raw_extra - extra_floor))
            for idx in frac_order[:leftover]:
                alloc[idx] += 1

    return {int(i): int(v) for i, v in enumerate(alloc.tolist())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_manifest", required=True)
    ap.add_argument("--audit_csv", required=True)
    ap.add_argument("--baseline_summary", default=None)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--policy_id", default="class_repair_topk9000")
    ap.add_argument("--total_budget", type=int, default=9000)
    ap.add_argument("--base_per_class", type=int, default=500)
    args = ap.parse_args()

    source_manifest_path = Path(args.source_manifest)
    audit_csv_path = Path(args.audit_csv)
    baseline_summary_path = Path(args.baseline_summary) if args.baseline_summary else None
    out_manifest_path = Path(args.out_manifest)

    manifest = load_json(source_manifest_path)
    entries = manifest.get("paths", [])
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"No paths found in source manifest: {source_manifest_path}")

    audit = pd.read_csv(audit_csv_path)

    if "index" not in audit.columns:

        audit.insert(0, "index", range(len(audit)))

    if len(audit) != len(entries):
        raise RuntimeError(f"audit rows ({len(audit)}) != manifest entries ({len(entries)})")

    required_cols = {"index", "requested_label", "requested_class_confidence"}
    missing = required_cols - set(audit.columns)
    if missing:
        raise RuntimeError(f"Audit CSV missing required columns: {sorted(missing)}")

    class_scores = audit.groupby(audit["requested_label"].astype(int))["requested_class_confidence"].median().sort_index().tolist()
    # Lower median requested-class confidence means harder/weaker synthetic class under the external audit signal.
    # Reuse allocation function by treating audit confidence like a class quality score.
    f1 = [float(x) for x in class_scores]
    allocation = compute_repair_allocation(
        f1=f1,
        total_budget=int(args.total_budget),
        base_per_class=int(args.base_per_class),
    )

    selected_indices: List[int] = []
    selection_rows: List[Dict[str, Any]] = []

    for cls, n_keep in allocation.items():
        g = audit[audit["requested_label"].astype(int) == int(cls)].copy()
        if len(g) < n_keep:
            raise RuntimeError(f"Class {cls} has only {len(g)} candidates, requested {n_keep}")
        g = g.sort_values("requested_class_confidence", ascending=False)
        chosen = g.head(n_keep)
        selected_indices.extend(chosen["index"].astype(int).tolist())
        selection_rows.append({
            "class": int(cls),
            "real_only_f1": float(f1[cls]),
            "allocated": int(n_keep),
            "candidate_count": int(len(g)),
            "min_selected_requested_conf": float(chosen["requested_class_confidence"].min()),
            "mean_selected_requested_conf": float(chosen["requested_class_confidence"].mean()),
        })

    selected_set = set(selected_indices)
    selected_entries = [deepcopy(e) for i, e in enumerate(entries) if i in selected_set]

    per_class_counts: Dict[str, int] = {}
    for e in selected_entries:
        y = str(int(e["label"]))
        per_class_counts[y] = per_class_counts.get(y, 0) + 1

    out_manifest = deepcopy(manifest)
    out_manifest["paths"] = selected_entries
    out_manifest["num_fake"] = int(len(selected_entries))
    out_manifest["per_class_counts"] = per_class_counts
    out_manifest["paper4_policy"] = {
        "paper_id": "paper4",
        "policy_id": args.policy_id,
        "policy_type": "weak_class_repair_confidence_ranked_topk",
        "source_manifest": str(source_manifest_path),
        "audit_csv": str(audit_csv_path),
        "baseline_summary": str(baseline_summary_path) if args.baseline_summary else None,
        "total_budget": int(args.total_budget),
        "base_per_class": int(args.base_per_class),
        "allocation": {str(k): int(v) for k, v in allocation.items()},
        "audit_median_requested_class_confidence": {str(i): float(v) for i, v in enumerate(f1)},
        "num_before": int(len(entries)),
        "num_after": int(len(selected_entries)),
        "num_rejected": int(len(entries) - len(selected_entries)),
        "acceptance_rate": float(len(selected_entries) / max(1, len(entries))),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Allocate a fixed total synthetic budget toward classes with lower median requested-class confidence, then select top requested-class-confidence samples within each requested class."
    }

    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_manifest, open(out_manifest_path, "w"), indent=2)

    alloc_csv = out_manifest_path.with_name("allocation.csv")
    with open(alloc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(selection_rows[0].keys()))
        w.writeheader()
        w.writerows(selection_rows)

    selected_audit_csv = out_manifest_path.with_name("selected_audit.csv")
    audit["selected"] = audit["index"].astype(int).isin(selected_set).astype(int)
    audit.to_csv(selected_audit_csv, index=False)

    print("[ok] wrote class-repair manifest:", out_manifest_path)
    print("[ok] wrote allocation csv:", alloc_csv)
    print("[ok] wrote selected audit csv:", selected_audit_csv)
    print("policy_id:", args.policy_id)
    print("num_before:", len(entries))
    print("num_after:", len(selected_entries))
    print("acceptance_rate:", len(selected_entries) / max(1, len(entries)))
    print("allocation:", {str(k): int(v) for k, v in allocation.items()})
    print("per_class_counts:", per_class_counts)


if __name__ == "__main__":
    main()