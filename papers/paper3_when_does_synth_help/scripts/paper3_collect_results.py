#!/usr/bin/env python3
from __future__ import annotations
import json, csv
from pathlib import Path
from typing import Any, Dict, List, Optional

def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r") as f:
        return json.load(f)

def get(d: Dict[str, Any], key: str) -> Optional[Any]:
    # supports flattened keys like "metrics.downstream.macro_f1"
    if key in d:
        return d[key]
    cur: Any = d
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    arts = args.artifacts
    models = args.models
    if not models:
        # discover models by folders
        models = [p.name for p in arts.iterdir() if p.is_dir()]

    rows: List[Dict[str, Any]] = []

    for m in sorted(models):
        latest = arts / m / "summaries" / "latest.json"
        if not latest.exists():
            continue
        s = load_json(latest)

        rm = get(s, "run_meta") or {}
        if not isinstance(rm, dict):
            rm = {}

        regime = rm.get("regime", {}) if isinstance(rm.get("regime", {}), dict) else {}

        # utility
        ro_f1 = get(s, "utility_real_only.macro_f1") or get(s, "metrics.downstream.real_only.macro_f1")
        ra_f1 = get(s, "utility_real_plus_synth.macro_f1") or get(s, "metrics.downstream.macro_f1")
        ro_ba = get(s, "utility_real_only.bal_acc") or get(s, "utility_real_only.balanced_acc")
        ra_ba = get(s, "utility_real_plus_synth.bal_acc") or get(s, "utility_real_plus_synth.balanced_acc")
        ro_auprc = get(s, "utility_real_only.macro_auprc")
        ra_auprc = get(s, "utility_real_plus_synth.macro_auprc")

        def f(x): 
            try: return float(x)
            except: return None

        ro_f1, ra_f1 = f(ro_f1), f(ra_f1)
        ro_ba, ra_ba = f(ro_ba), f(ra_ba)
        ro_auprc, ra_auprc = f(ro_auprc), f(ra_auprc)

        row = {
            "model": m,
            "run_id": get(s, "run_id"),
            "paper_id": rm.get("paper_id"),
            "config_id": rm.get("config_id"),
            "dataset_id": rm.get("dataset_id"),
            "imbalance": regime.get("imbalance"),
            "synth_budget_per_class": regime.get("synth_budget_per_class"),
            "aug_method": regime.get("aug_method"),

            "macro_f1_real_only": ro_f1,
            "macro_f1_real_plus_synth": ra_f1,
            "delta_macro_f1": (ra_f1 - ro_f1) if (ro_f1 is not None and ra_f1 is not None) else None,

            "bal_acc_real_only": ro_ba,
            "bal_acc_real_plus_synth": ra_ba,
            "delta_bal_acc": (ra_ba - ro_ba) if (ro_ba is not None and ra_ba is not None) else None,

            "macro_auprc_real_only": ro_auprc,
            "macro_auprc_real_plus_synth": ra_auprc,
            "delta_macro_auprc": (ra_auprc - ro_auprc) if (ro_auprc is not None and ra_auprc is not None) else None,

            # generative diagnostics if present
            "kid": f(get(s, "generative.kid") or get(s, "metrics.kid")),
            "ms_ssim": f(get(s, "generative.ms_ssim") or get(s, "metrics.ms_ssim")),
        }
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[ok] wrote {args.out} ({len(rows)} rows)")

if __name__ == "__main__":
    main()