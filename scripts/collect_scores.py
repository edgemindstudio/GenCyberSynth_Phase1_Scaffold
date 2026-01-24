# scripts/collect_scores.py

#!/usr/bin/env python3
import os, json, csv, pathlib as p

ROOT = p.Path("artifacts")

# Which per-model summary file to read (paper-clean default)
SUMMARY_NAME = os.environ.get("PHASE1_SUMMARY_NAME", "paper1.json").strip()

# Keep MODELS list, but you can also auto-discover if you want later
MODELS = ["gan","vae","gaussianmixture","diffusion","autoregressive","restrictedboltzmann","maskedautoflow"]

def pick(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

rows = []
for m in MODELS:
    fp = ROOT / m / "summaries" / SUMMARY_NAME
    if not fp.exists():
        rows.append({"model": m, "summary_file": None})
        continue

    d = json.loads(fp.read_text())

    rows.append({
        "model": m,
        "summary_file": str(fp),

        # counts
        "num_real":      pick(d, "counts.num_real"),
        "num_fake":      pick(d, "counts.num_fake"),

        # generative metrics
        "kid":           pick(d, "metrics.kid"),
        "cfid":          pick(d, "metrics.cfid"),
        "gen_precision": pick(d, "metrics.gen_precision"),
        "gen_recall":    pick(d, "metrics.gen_recall"),
        "ms_ssim":       pick(d, "metrics.ms_ssim"),

        # downstream metrics
        "macro_f1":      pick(d, "metrics.downstream.macro_f1"),
        "macro_auprc":   pick(d, "metrics.downstream.macro_auprc"),
        "balanced_acc":  pick(d, "metrics.downstream.balanced_acc"),
    })

# Pretty print
cols = ["kid","cfid","gen_precision","gen_recall","ms_ssim","balanced_acc","num_real","num_fake"]
w = max(len(r["model"]) for r in rows) if rows else 10
print(f"Using summaries: {SUMMARY_NAME}\n")
print(f"{'MODEL':<{w}}  " + "  ".join(f"{c:>13}" for c in cols))
for r in rows:
    vals = []
    for c in cols:
        v = r.get(c, "—")
        if isinstance(v, float):
            v = f"{v:.4g}"
        vals.append(str(v))
    print(f"{r['model']:<{w}}  " + "  ".join(f"{v:>13}" for v in vals))

# CSV
csv_path = ROOT / "phase1_scores.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {csv_path}")
