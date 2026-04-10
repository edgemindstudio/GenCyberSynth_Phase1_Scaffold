# scripts/metrics/fid_best.py

#!/usr/bin/env python3
import sys, csv, pathlib

combined = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("artifacts/summaries/fid_grid_combined.csv")
outdir   = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("artifacts/summaries")

rows = list(csv.DictReader(combined.open()))
if not rows:
    raise SystemExit(f"No rows in {combined}")

# Best row per model only
best = {}
for r in rows:
    model = r["model"]
    fid = float(r["fid"])
    if model not in best or fid < float(best[model]["fid"]):
        best[model] = r

outdir.mkdir(parents=True, exist_ok=True)

# Write CSV
csv_path = outdir / "fid_grid_best_per_model.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(best.values())

# Write MD
md_path = outdir / "fid_grid_best_per_model.md"
cols = ["model","backbone","img_size","per_class_cap","total_cap","batch","fid","job_id","task_id","log"]
with md_path.open("w") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
    for model in sorted(best.keys()):
        r = best[model]
        f.write("| " + " | ".join(r.get(c, "") for c in cols) + " |\n")

print("Wrote", csv_path, "and", md_path)