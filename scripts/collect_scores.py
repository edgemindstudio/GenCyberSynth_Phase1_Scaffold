# scripts/collect_scores.py

#!/usr/bin/env python3
import json, csv, pathlib as p

ROOT = p.Path("artifacts")
MODELS = ["gan","vae","gaussianmixture","diffusion","autoregressive","restrictedboltzmann","maskedautoflow"]

def latest_summary(model: str):
    sdir = ROOT/model/"summaries"
    files = sorted(sdir.glob("summary_*.json"))
    return files[-1] if files else None

def pick(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

rows = []
for m in MODELS:
    fp = latest_summary(m)
    if not fp:
        rows.append({"model": m, "summary_file": None})
        continue
    d = json.loads(fp.read_text())
    rows.append({
        "model": m,
        "summary_file": str(fp),
        "num_real":      pick(d, "counts.num_real"),
        "num_fake":      pick(d, "counts.num_fake"),
        "kid":           pick(d, "metrics.kid"),
        "cfid":          pick(d, "metrics.cfid"),
        "gen_precision": pick(d, "metrics.gen_precision"),
        "gen_recall":    pick(d, "metrics.gen_recall"),
        "ms_ssim":       pick(d, "metrics.ms_ssim"),
        "macro_f1":      pick(d, "metrics.downstream.macro_f1"),
        "macro_auprc":   pick(d, "metrics.downstream.macro_auprc"),
        "balanced_acc":  pick(d, "metrics.downstream.balanced_acc"),
    })

# Pretty print
cols = ["kid","cfid","gen_precision","gen_recall","ms_ssim","balanced_acc","num_real","num_fake"]
w = max(len(r["model"]) for r in rows) if rows else 10
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
csv_path = ROOT/"phase1_scores.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader(); writer.writerows(rows)
print(f"\nWrote {csv_path}")
