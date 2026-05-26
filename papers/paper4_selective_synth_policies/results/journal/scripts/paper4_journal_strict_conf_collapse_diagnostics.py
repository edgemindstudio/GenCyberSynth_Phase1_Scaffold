#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("papers/paper4_selective_synth_policies")
OUT_TABLE_DIR = ROOT / "results/journal/tables"
OUT_FIG_DIR = ROOT / "results/journal/figures"
OUT_NOTE_DIR = ROOT / "results/journal/notes"

OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_NOTE_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_CSV = Path("/home/bruno.fonkeng/gencys/artifacts_paper4/gan/synthetic/paper4_baseline_b2000/seed42/policy/confidence_accept_t080/audit.csv")

def write_markdown(df: pd.DataFrame, path: Path):
    tmp = df.copy().fillna("")
    headers = list(tmp.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")

def main():
    if not AUDIT_CSV.exists():
        raise SystemExit(f"Missing audit CSV: {AUDIT_CSV}")

    df = pd.read_csv(AUDIT_CSV)
    total_before = len(df)
    accepted = df[df["accepted"] == 1].copy()
    total_after = len(accepted)
    acceptance_rate = total_after / total_before if total_before else 0.0

    requested_counts = df["requested_label"].value_counts().sort_index()
    accepted_counts = accepted["requested_label"].value_counts().sort_index()
    predicted_counts = df["predicted_label"].value_counts().sort_index()
    accepted_predicted_counts = accepted["predicted_label"].value_counts().sort_index()

    classes = sorted(set(requested_counts.index.tolist()) | set(accepted_counts.index.tolist()) | set(predicted_counts.index.tolist()))
    rows = []
    for c in classes:
        before = int(requested_counts.get(c, 0))
        after = int(accepted_counts.get(c, 0))
        pred_all = int(predicted_counts.get(c, 0))
        pred_acc = int(accepted_predicted_counts.get(c, 0))
        rows.append({"class": int(c), "requested_before": before, "accepted_after": after, "class_acceptance_rate": after / before if before else 0.0, "predicted_all": pred_all, "predicted_accepted": pred_acc})

    by_class = pd.DataFrame(rows)
    by_class_csv = OUT_TABLE_DIR / "paper4_journal_strict_conf_collapse_by_class.csv"
    by_class_md = OUT_TABLE_DIR / "paper4_journal_strict_conf_collapse_by_class.md"
    by_class.to_csv(by_class_csv, index=False)
    display_by_class = by_class.copy()

    int_cols = ["class", "requested_before", "accepted_after", "predicted_all", "predicted_accepted"]

    display_by_class[int_cols] = display_by_class[int_cols].astype(int).astype(str)

    display_by_class["class_acceptance_rate"] = display_by_class["class_acceptance_rate"].round(6).astype(str)

    write_markdown(display_by_class, by_class_md)

    summary = pd.DataFrame([{"policy": "Strict-conf.", "source_pool_size": total_before, "accepted_size": total_after, "acceptance_rate": acceptance_rate, "requested_classes_before": int(df["requested_label"].nunique()), "requested_classes_after": int(accepted["requested_label"].nunique()), "predicted_classes_before": int(df["predicted_label"].nunique()), "predicted_classes_after": int(accepted["predicted_label"].nunique())}])
    summary_csv = OUT_TABLE_DIR / "paper4_journal_strict_conf_collapse_summary.csv"
    summary_md = OUT_TABLE_DIR / "paper4_journal_strict_conf_collapse_summary.md"
    summary.to_csv(summary_csv, index=False)
    write_markdown(summary.round(6), summary_md)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = range(len(by_class))
    ax.bar([i - 0.2 for i in x], by_class["requested_before"], width=0.4, label="Before filtering")
    ax.bar([i + 0.2 for i in x], by_class["accepted_after"], width=0.4, label="After Strict-conf.")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(c) for c in by_class["class"]])
    ax.set_xlabel("Requested class")
    ax.set_ylabel("Number of samples")
    ax.set_title("Strict-conf. class coverage collapse")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "paper4_journal_strict_conf_class_coverage_collapse.png", dpi=300)
    fig.savefig(OUT_FIG_DIR / "paper4_journal_strict_conf_class_coverage_collapse.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(by_class["class"].astype(str), by_class["class_acceptance_rate"])
    ax.set_xlabel("Requested class")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Strict-conf. acceptance rate by requested class")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "paper4_journal_strict_conf_acceptance_rate_by_class.png", dpi=300)
    fig.savefig(OUT_FIG_DIR / "paper4_journal_strict_conf_acceptance_rate_by_class.pdf")
    plt.close(fig)

    note = OUT_NOTE_DIR / "paper4_journal_strict_conf_collapse_notes.md"
    note.write_text(
        "# Paper 4 Strict-conf. Collapse Diagnostics\n\n"
        f"Strict-conf. was evaluated on the seed42 b2000 source pool with {total_before} generated samples.\n\n"
        f"It accepted {total_after} samples, for an acceptance rate of {acceptance_rate:.6f}.\n\n"
        f"The source pool contains {int(df['requested_label'].nunique())} requested classes before filtering, but the accepted set contains {int(accepted['requested_label'].nunique())} requested class after filtering.\n\n"
        "Main interpretation:\n\n"
        "- Strict confidence acceptance is diagnostic rather than practical.\n"
        "- The policy applies a plausible confidence rule but destroys class coverage.\n"
        "- This supports the claim that sample-level confidence alone is insufficient for synthetic-data policy selection.\n"
        "- Confidence filtering must be combined with class coverage, budget, diversity, or downstream utility constraints.\n"
    )

    print("[ok] wrote", by_class_csv)
    print("[ok] wrote", by_class_md)
    print("[ok] wrote", summary_csv)
    print("[ok] wrote", summary_md)
    print("[ok] wrote figures to", OUT_FIG_DIR)
    print("[ok] wrote", note)
    print(summary.to_string(index=False))
    print(by_class.to_string(index=False))

if __name__ == "__main__":
    main()