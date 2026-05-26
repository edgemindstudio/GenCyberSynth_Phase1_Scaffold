#!/usr/bin/env python3
from pathlib import Path
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("papers/paper4_selective_synth_policies")
OUT_TABLE_DIR = ROOT / "results/journal/tables"
OUT_FIG_DIR = ROOT / "results/journal/figures"
OUT_NOTE_DIR = ROOT / "results/journal/notes"

OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_NOTE_DIR.mkdir(parents=True, exist_ok=True)

ALLOC_GLOB = "/home/bruno.fonkeng/gencys/artifacts_paper4/gan/synthetic/paper4_baseline_b2000/seed*/policy/class_repair_topk9000/allocation.csv"

def extract_seed(path: str) -> int:
    m = re.search(r"/seed(\d+)/", path)
    if not m:
        raise ValueError(f"Could not extract seed from path: {path}")
    return int(m.group(1))

def write_markdown(df: pd.DataFrame, path: Path):
    tmp = df.copy().fillna("")
    headers = list(tmp.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")

def main():
    files = sorted(glob.glob(ALLOC_GLOB))
    if not files:
        raise SystemExit(f"No allocation files found with glob: {ALLOC_GLOB}")

    rows = []
    for f in files:
        seed = extract_seed(f)
        df = pd.read_csv(f)
        df = df.rename(columns={"real_only_f1": "audit_confidence_score"})
        df["seed"] = seed
        df["source_file"] = f
        rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df[["seed", "class", "audit_confidence_score", "allocated", "candidate_count", "min_selected_requested_conf", "mean_selected_requested_conf", "source_file"]]
    all_df = all_df.sort_values(["seed", "class"], kind="stable")

    by_seed_csv = OUT_TABLE_DIR / "paper4_journal_class_repair_allocation_by_seed.csv"
    by_seed_md = OUT_TABLE_DIR / "paper4_journal_class_repair_allocation_by_seed.md"
    all_df.to_csv(by_seed_csv, index=False)
    write_markdown(all_df[["seed", "class", "audit_confidence_score", "allocated", "candidate_count", "mean_selected_requested_conf"]].round(6), by_seed_md)

    mean_std = all_df.groupby("class", as_index=False).agg(
        allocation_mean=("allocated", "mean"),
        allocation_std=("allocated", "std"),
        allocation_min=("allocated", "min"),
        allocation_max=("allocated", "max"),
        audit_confidence_mean=("audit_confidence_score", "mean"),
        audit_confidence_std=("audit_confidence_score", "std"),
        selected_confidence_mean=("mean_selected_requested_conf", "mean"),
        selected_confidence_std=("mean_selected_requested_conf", "std"),
    )
    mean_std_csv = OUT_TABLE_DIR / "paper4_journal_class_repair_allocation_mean_std.csv"
    mean_std_md = OUT_TABLE_DIR / "paper4_journal_class_repair_allocation_mean_std.md"
    mean_std.to_csv(mean_std_csv, index=False)
    write_markdown(mean_std.round(6), mean_std_md)

    # Allocation heatmap-style table plot
    pivot_alloc = all_df.pivot(index="class", columns="seed", values="allocated").sort_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    im = ax.imshow(pivot_alloc.values, aspect="auto")
    ax.set_xticks(range(len(pivot_alloc.columns)))
    ax.set_xticklabels([f"seed{c}" for c in pivot_alloc.columns])
    ax.set_yticks(range(len(pivot_alloc.index)))
    ax.set_yticklabels([str(c) for c in pivot_alloc.index])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Class")
    ax.set_title("Class-repair allocation by class and seed")
    for i, cls in enumerate(pivot_alloc.index):
        for j, seed in enumerate(pivot_alloc.columns):
            ax.text(j, i, str(int(pivot_alloc.loc[cls, seed])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Allocated samples")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "paper4_journal_class_repair_allocation_heatmap.png", dpi=300)
    fig.savefig(OUT_FIG_DIR / "paper4_journal_class_repair_allocation_heatmap.pdf")
    plt.close(fig)

    # Allocation versus audit confidence scatter
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.scatter(all_df["audit_confidence_score"], all_df["allocated"])
    for _, r in all_df.iterrows():
        ax.annotate(f"c{int(r['class'])}/s{int(r['seed'])}", (r["audit_confidence_score"], r["allocated"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Audit confidence score")
    ax.set_ylabel("Allocated samples")
    ax.set_title("Class-repair allocation versus audit confidence")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "paper4_journal_class_repair_allocation_vs_audit_confidence.png", dpi=300)
    fig.savefig(OUT_FIG_DIR / "paper4_journal_class_repair_allocation_vs_audit_confidence.pdf")
    plt.close(fig)

    corr = all_df[["audit_confidence_score", "allocated"]].corr(method="spearman").iloc[0, 1]
    pearson = all_df[["audit_confidence_score", "allocated"]].corr(method="pearson").iloc[0, 1]
    corr_df = pd.DataFrame([{"relationship": "audit_confidence_score_vs_allocated", "pearson": pearson, "spearman": corr, "n": len(all_df)}])
    corr_csv = OUT_TABLE_DIR / "paper4_journal_class_repair_allocation_correlation.csv"
    corr_md = OUT_TABLE_DIR / "paper4_journal_class_repair_allocation_correlation.md"
    corr_df.to_csv(corr_csv, index=False)
    write_markdown(corr_df.round(4), corr_md)

    note = OUT_NOTE_DIR / "paper4_journal_class_repair_allocation_notes.md"
    note.write_text(
        "# Paper 4 Class-repair Allocation Diagnostics\n\n"
        "This diagnostic summarizes how Class-repair reallocates the fixed 9000-sample budget across classes and seeds.\n\n"
        "Important naming note: the source allocation CSV column `real_only_f1` is used here as `audit_confidence_score`. It reflects the audit-derived class confidence score used by the allocation rule, not true downstream real-only F1.\n\n"
        "Main interpretation:\n\n"
        "- Class-repair is not equivalent to balanced Top-k1000. Top-k1000 assigns 1000 samples per class, while Class-repair reallocates the same 9000 total samples adaptively.\n"
        "- Classes with higher audit confidence generally receive the base allocation or near-base allocation.\n"
        "- Classes with lower audit confidence receive larger allocations.\n"
        "- The allocation pattern is seed-dependent, which should be reported as a policy behavior rather than hidden.\n"
        "- This supports the journal claim that allocation structure can affect downstream utility under a fixed selected-sample budget.\n"
    )

    print("[ok] wrote", by_seed_csv)
    print("[ok] wrote", by_seed_md)
    print("[ok] wrote", mean_std_csv)
    print("[ok] wrote", mean_std_md)
    print("[ok] wrote", corr_csv)
    print("[ok] wrote", corr_md)
    print("[ok] wrote figures to", OUT_FIG_DIR)
    print("[ok] wrote", note)

if __name__ == "__main__":
    main()