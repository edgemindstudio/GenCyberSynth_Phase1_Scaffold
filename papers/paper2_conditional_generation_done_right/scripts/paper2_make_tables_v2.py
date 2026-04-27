#!/usr/bin/env python3
"""
Paper 2 table builder v2.

Purpose:
    Combine GAN eval summary and conditioning audit outputs into a single
    Evidence Block 1 CSV.

Run from repo root:
    python papers/paper2_conditional_generation_done_right/scripts/paper2_make_tables_v2.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]

if Path.cwd().resolve() != REPO_ROOT:
    os.chdir(REPO_ROOT)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PAPER_DIR = REPO_ROOT / "papers" / "paper2_conditional_generation_done_right"
TABLES_DIR = PAPER_DIR / "results" / "tables"

ARTIFACTS_ROOT = Path("/home/bruno.fonkeng/gencys/artifacts_paper2")
LATEST_JSON = ARTIFACTS_ROOT / "gan" / "summaries" / "latest.json"

AUDIT_SUMMARY = TABLES_DIR / "paper2_gan_conditioning_summary_v2.json"
AUDIT_PER_CLASS = TABLES_DIR / "paper2_gan_conditioning_accuracy_v2.csv"

OUT_CSV = TABLES_DIR / "paper2_evidence_block1_consolidated.csv"
OUT_JSON = TABLES_DIR / "paper2_evidence_block1_consolidated.json"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[warn] JSON file missing: {path}")
        return {}

    with open(path, "r") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return default
        if part not in cur:
            return default
        cur = cur[part]
    return cur


def flatten_json(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}

    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            out.update(flatten_json(value, new_key))
        elif isinstance(value, list):
            out[new_key] = json.dumps(value)
        else:
            out[new_key] = value

    return out


def read_per_class_accuracy(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[warn] Per-class audit CSV missing: {path}")
        return []

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


def summarize_per_class(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "audit.per_class_macro_conditioning_accuracy": None,
            "audit.per_class_worst_conditioning_accuracy": None,
            "audit.per_class_weighted_conditioning_accuracy": None,
            "audit.worst_class_id": None,
        }

    accs = []
    counts = []
    class_ids = []

    for row in rows:
        class_id = int(row["class_id"])
        acc = float(row["cond_accuracy"])
        count = int(row["count"])

        class_ids.append(class_id)
        accs.append(acc)
        counts.append(count)

    total = sum(counts)

    macro_acc = sum(accs) / len(accs) if accs else None
    weighted_acc = (
        sum(a * c for a, c in zip(accs, counts)) / total
        if total > 0
        else None
    )

    worst_idx = min(range(len(accs)), key=lambda i: accs[i])
    worst_acc = accs[worst_idx]
    worst_class = class_ids[worst_idx]

    return {
        "audit.per_class_macro_conditioning_accuracy": macro_acc,
        "audit.per_class_worst_conditioning_accuracy": worst_acc,
        "audit.per_class_weighted_conditioning_accuracy": weighted_acc,
        "audit.worst_class_id": worst_class,
    }


def extract_eval_metrics(latest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract likely available eval metrics safely.

    This is intentionally permissive because Paper 1/Paper 2 summaries may have
    nested keys that vary slightly by model family.
    """

    flat = flatten_json(latest)

    wanted_substrings = [
        "kid",
        "fid",
        "cfid",
        "msssim",
        "ms_ssim",
        "accuracy",
        "macro_f1",
        "balanced",
        "auprc",
        "ece",
        "brier",
    ]

    extracted = {}

    for key, value in flat.items():
        low = key.lower()
        if any(token in low for token in wanted_substrings):
            extracted[f"eval.{key}"] = value

    return extracted


def main() -> None:
    print(f"[paper2-tables-v2] Reading eval summary: {LATEST_JSON}")
    latest = load_json_if_exists(LATEST_JSON)

    print(f"[paper2-tables-v2] Reading audit summary: {AUDIT_SUMMARY}")
    audit = load_json_if_exists(AUDIT_SUMMARY)

    per_class_rows = read_per_class_accuracy(AUDIT_PER_CLASS)

    row = {
        "paper_id": "paper2",
        "model_family": "gan",
        "evidence_block": "block1_conditioning_audit",
        "artifacts_root": str(ARTIFACTS_ROOT),
        "latest_json": str(LATEST_JSON),
        "audit_summary_json": str(AUDIT_SUMMARY),
    }

    row.update(extract_eval_metrics(latest))

    row.update({
        "audit.version": audit.get("audit_version"),
        "audit.seed": audit.get("seed"),
        "audit.total_synthetic_audited": audit.get("total_synthetic_audited"),
        "audit.overall_conditioning_accuracy": audit.get("overall_conditioning_accuracy"),
        "audit.overall_conditioning_failure_rate": audit.get("overall_conditioning_failure_rate"),
        "audit.leakage_rate": audit.get("leakage_rate"),
        "audit.minority_conditioning_accuracy": audit.get("minority_conditioning_accuracy"),
        "audit.minority_conditioning_failure_rate": audit.get("minority_conditioning_failure_rate"),
        "audit.minority_classes": json.dumps(
            safe_get(audit, "minority_definition.minority_classes", [])
        ),
        "audit.predicted_label_histogram": json.dumps(
            audit.get("predicted_label_histogram", {})
        ),
        "audit.top_confusions": json.dumps(
            audit.get("top_confusions", [])
        ),
        "audit.real_only_val_accuracy": safe_get(
            audit, "real_only_classifier.val_accuracy"
        ),
        "audit.real_only_test_accuracy": safe_get(
            audit, "real_only_classifier.test_accuracy"
        ),
        "audit.real_only_weights_path": safe_get(
            audit, "real_only_classifier.weights_path"
        ),
    })

    row.update(summarize_per_class(per_class_rows))

    # Write JSON
    with open(OUT_JSON, "w") as f:
        json.dump(row, f, indent=2)

    # Write single-row CSV
    fieldnames = sorted(row.keys())

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print("[paper2-tables-v2] DONE")
    print(f"[paper2-tables-v2] CSV: {OUT_CSV}")
    print(f"[paper2-tables-v2] JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()