#!/usr/bin/env python3
from pathlib import Path
import json
import glob
import pandas as pd

ROOT = Path("papers/paper4_selective_synth_policies")
OUT_TABLE_DIR = ROOT / "results/journal/tables"
OUT_NOTE_DIR = ROOT / "results/journal/notes"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_NOTE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_GLOB = "/home/bruno.fonkeng/gencys/artifacts_paper4_cicmaldroid/gan/summaries/summary_*.json"

POLICY_LABELS = {
    "paper4_cic_policy_keep_all_b2000": "CIC Keep-all b2000",
    "paper4_cic_policy_confidence_ranked_topk1000_b2000": "CIC Top-k1000",
    "paper4_cic_policy_class_repair_topk5000_b2000": "CIC Class-repair",
}

ORDER = {
    "CIC Keep-all b2000": 0,
    "CIC Top-k1000": 1,
    "CIC Class-repair": 2,
}

def write_markdown(df: pd.DataFrame, path: Path):
    tmp = df.copy().fillna("")
    headers = list(tmp.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")

def main():
    rows = []
    for p in sorted(glob.glob(SUMMARY_GLOB)):
        with open(p) as f:
            s = json.load(f)

        rm = s.get("run_meta") or {}
        cid = s.get("config_id") or rm.get("config_id")
        seed = s.get("seed") or rm.get("seed")

        if cid not in POLICY_LABELS:
            continue

        ro = s.get("utility_real_only") or s.get("real_only") or {}
        rs = s.get("utility_real_plus_synth") or s.get("real_plus_synth") or {}
        d = s.get("deltas_RS_minus_R") or {}

        rows.append({
            "summary_path": p,
            "config_id": cid,
            "policy_label": POLICY_LABELS[cid],
            "seed": int(seed),
            "num_fake": s.get("counts.num_fake"),
            "real_only_macro_f1": ro.get("macro_f1"),
            "real_plus_synth_macro_f1": rs.get("macro_f1"),
            "delta_macro_f1": d.get("delta_macro_f1"),
            "delta_bal_acc": d.get("delta_bal_acc"),
            "delta_macro_auprc": d.get("delta_macro_auprc"),
            "delta_ece": d.get("delta_ece"),
            "delta_brier": d.get("delta_brier"),
            "kid": s.get("metrics.kid"),
            "ms_ssim": s.get("metrics.ms_ssim"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[error] no CICMalDroid policy summaries found")

    df = df.sort_values(["config_id", "seed", "summary_path"], kind="stable")
    all_csv = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_policy_all_summaries.csv"
    df.to_csv(all_csv, index=False)

    dedup = df.drop_duplicates(subset=["config_id", "seed"], keep="last").copy()
    dedup["_order"] = dedup["policy_label"].map(ORDER)
    dedup = dedup.sort_values(["_order", "seed"], kind="stable").drop(columns=["_order"])

    seed_csv = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_policy_seed_level.csv"
    seed_md = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_policy_seed_level.md"
    seed_cols = ["policy_label", "seed", "num_fake", "delta_macro_f1", "delta_bal_acc", "delta_macro_auprc", "delta_ece", "delta_brier", "kid", "ms_ssim"]
    dedup.to_csv(seed_csv, index=False)
    write_markdown(dedup[seed_cols].round(6), seed_md)

    metric_cols = ["delta_macro_f1", "delta_bal_acc", "delta_macro_auprc", "delta_ece", "delta_brier", "kid", "ms_ssim"]
    summary_rows = []

    for label, g in dedup.groupby("policy_label", sort=False):
        row = {
            "policy_label": label,
            "n_seeds": int(g["seed"].nunique()),
            "mean_num_fake": float(g["num_fake"].mean()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary["_order"] = summary["policy_label"].map(ORDER)
    summary = summary.sort_values("_order", kind="stable").drop(columns=["_order"])

    summary_csv = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_policy_mean_std.csv"
    summary_md = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_policy_mean_std.md"
    summary.to_csv(summary_csv, index=False)
    write_markdown(summary.round(6), summary_md)

    note = OUT_NOTE_DIR / "paper4_journal_cicmaldroid_policy_notes.md"
    note.write_text(
        "# Paper 4 CICMalDroid2020 Policy Validation\n\n"
        "This diagnostic compares CICMalDroid2020 Keep-all b2000, Top-k1000, and Class-repair policies.\n\n"
        "The collector deduplicates by config_id and seed, keeping the latest summary when duplicate seed runs exist.\n\n"
        "Main interpretation:\n\n"
        "- CICMalDroid2020 acts as the second-dataset validation layer for Paper 4.\n"
        "- Keep-all b2000 uses 10000 synthetic samples, while Top-k1000 and Class-repair use 5000 selected samples.\n"
        "- Reduced-budget policies do not consistently reverse negative balanced-accuracy effects on CICMalDroid2020.\n"
        "- Top-k1000 is generally more favorable than Class-repair on CICMalDroid in the current runs, especially for Macro-AUPRC.\n"
        "- This contrasts with USTC-TFC2016 and supports the journal claim that selective synthetic-data policies are dataset-sensitive.\n"
        "- The paper should avoid claiming that Class-repair is universally best.\n"
    )

    print("[ok] wrote", all_csv)
    print("[ok] wrote", seed_csv)
    print("[ok] wrote", seed_md)
    print("[ok] wrote", summary_csv)
    print("[ok] wrote", summary_md)
    print("[ok] wrote", note)

    print("\nSEED LEVEL")
    print(dedup[seed_cols].to_string(index=False))

    print("\nMEAN/STDEV")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()