#!/usr/bin/env python3
import sys, csv, pathlib

combined = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("artifacts/summaries/fid_grid_combined.csv")
outdir   = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("artifacts/summaries")
rows = list(csv.DictReader(combined.open()))
if not rows:
    raise SystemExit(f"No rows in {combined}")

best = {}
for r in rows:
    key = (r.get("model",""), r.get("backbone",""), r.get("img_size",""))
    fid = float(r["fid"])
    if key not in best or fid < float(best[key]["fid"]):
        best[key] = r

outdir.mkdir(parents=True, exist_ok=True)

csv_path = outdir/"fid_grid_best_per_model_backbone_img.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(best.values())

md_path = outdir/"fid_grid_best_per_model_backbone_img.md"
cols = ["model","backbone","img_size","per_class_cap","total_cap","batch","fid","job_id","task_id","log"]
with md_path.open("w") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
    for k in sorted(best):
        r = best[k]
        f.write("| " + " | ".join(r.get(c,"") for c in cols) + " |\n")

print("Wrote", csv_path, "and", md_path)
