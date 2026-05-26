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
TARGET_CONFIGS = {"paper4_cic_realonly_balanced", "paper4_cic_policy_keep_all_b2000"}

def metric_get(summary, name):
    if name in summary:
        return summary.get(name)
    return None

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
        if cid not in TARGET_CONFIGS:
            continue
        ro = s.get("utility_real_only") or s.get("real_only") or {}
        rs = s.get("utility_real_plus_synth") or s.get("real_plus_synth") or {}
        d = s.get("deltas_RS_minus_R") or {}
        rows.append({
            "summary_path": p,
            "config_id": cid,
            "seed": int(seed) if seed is not None else None,
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
        raise SystemExit("[error] no CICMalDroid summaries found")

    df = df.sort_values(["config_id", "seed", "summary_path"], kind="stable")
    full_csv = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_keepall_all_summaries.csv"
    df.to_csv(full_csv, index=False)

    dedup = df.drop_duplicates(subset=["config_id", "seed"], keep="last").copy()
    dedup = dedup.sort_values(["config_id", "seed"], kind="stable")
    dedup_csv = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_keepall_seed_level.csv"
    dedup_md = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_keepall_seed_level.md"
    dedup.to_csv(dedup_csv, index=False)
    display_cols = ["config_id", "seed", "num_fake", "real_only_macro_f1", "real_plus_synth_macro_f1", "delta_macro_f1", "delta_bal_acc", "delta_macro_auprc", "delta_ece", "delta_brier"]
    write_markdown(dedup[display_cols].round(6), dedup_md)

    keep = dedup[dedup["config_id"] == "paper4_cic_policy_keep_all_b2000"].copy()

    mean_std = pd.DataFrame([{

        "config_id": "paper4_cic_policy_keep_all_b2000",

        "n_seeds": int(keep["seed"].nunique()),

        "mean_num_fake": float(keep["num_fake"].mean()),

        "delta_macro_f1_mean": float(keep["delta_macro_f1"].mean()),

        "delta_macro_f1_std": float(keep["delta_macro_f1"].std()),

        "delta_bal_acc_mean": float(keep["delta_bal_acc"].mean()),

        "delta_bal_acc_std": float(keep["delta_bal_acc"].std()),

        "delta_macro_auprc_mean": float(keep["delta_macro_auprc"].mean()),

        "delta_macro_auprc_std": float(keep["delta_macro_auprc"].std()),

        "delta_ece_mean": float(keep["delta_ece"].mean()),

        "delta_ece_std": float(keep["delta_ece"].std()),

        "delta_brier_mean": float(keep["delta_brier"].mean()),

        "delta_brier_std": float(keep["delta_brier"].std()),

        "kid_mean": float(keep["kid"].mean()),

        "kid_std": float(keep["kid"].std()),

        "ms_ssim_mean": float(keep["ms_ssim"].mean()),

        "ms_ssim_std": float(keep["ms_ssim"].std()),

    }])

    mean_csv = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_keepall_mean_std.csv"
    mean_md = OUT_TABLE_DIR / "paper4_journal_cicmaldroid_keepall_mean_std.md"
    mean_std.to_csv(mean_csv, index=False)
    write_markdown(mean_std.round(6), mean_md)

    note = OUT_NOTE_DIR / "paper4_journal_cicmaldroid_keepall_notes.md"
    note.write_text(
        "# Paper 4 CICMalDroid2020 Keep-all Validation\n\n"
        "This diagnostic collects Paper 4 CICMalDroid2020 real-only and keep-all b2000 GAN evaluations.\n\n"
        "The collector deduplicates by config_id and seed, keeping the latest summary path when duplicate seed runs exist.\n\n"
        "Main interpretation:\n\n"
        "- CICMalDroid2020 provides the second-dataset validation layer for Paper 4.\n"
        "- Keep-all b2000 uses 10000 synthetic GAN samples, corresponding to 2000 samples per class across five classes.\n"
        "- Across seeds, keep-all tends to reduce Macro-F1 and balanced accuracy.\n"
        "- Macro-AUPRC and calibration metrics may improve even when Macro-F1 and balanced accuracy decline.\n"
        "- This supports a journal-safe claim that synthetic-data policy effects are dataset-sensitive and metric-sensitive.\n"
    )

    print("[ok] wrote", full_csv)
    print("[ok] wrote", dedup_csv)
    print("[ok] wrote", dedup_md)
    print("[ok] wrote", mean_csv)
    print("[ok] wrote", mean_md)
    print("[ok] wrote", note)
    print("\nDEDUP SEED LEVEL")
    print(dedup[display_cols].to_string(index=False))
    print("\nKEEPALL MEAN/STDEV")
    print(mean_std.to_string(index=False))

if __name__ == "__main__":
    main()