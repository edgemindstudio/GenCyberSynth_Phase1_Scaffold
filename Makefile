    key = (r.get("model",""), r.get("backbone",""), r.get("img_size",""))
    fid = float(r["fid"])
    if key not in best or fid < float(best[key]["fid"]):
        best[key] = r
outd = pathlib.Path("$(SUMMARIES_DIR)"); outd.mkdir(parents=True, exist_ok=True)
csv_path = outd/"fid_grid_best_per_model_backbone_img.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(best.values())
md_path = outd/"fid_grid_best_per_model_backbone_img.md"
cols = ["model","backbone","img_size","per_class_cap","total_cap","batch","fid","job_id","task_id","log"]
with md_path.open("w") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
    for key in sorted(best):
        r = best[key]
        f.write("| " + " | ".join(r.get(c,"") for c in cols) + " |\n")
print("Wrote", csv_path, "and", md_path)
PY

fid-plot-3d: fid-best-3d
	MPLBACKEND=Agg $(PY) scripts/metrics/fid_plot.py "$(SUMMARIES_DIR)/fid_grid_best_per_model_backbone_img.csv" "$(SUMMARIES_DIR)/fid_best_bar_backbone_img.png"
.PHONY: fid-best-3d fid-plot-3d

fid-best-3d: fid-parse
\t$(PY) - <<'PY'
import csv, pathlib
src = pathlib.Path("$(FID_COMBINED_CSV)")
rows = list(csv.DictReader(src.open()))
if not rows:
    raise SystemExit(f"No rows in {src}")
best = {}
for r in rows:
    key = (r.get("model",""), r.get("backbone",""), r.get("img_size",""))
    fid = float(r["fid"])
    if key not in best or fid < float(best[key]["fid"]):
        best[key] = r
outd = pathlib.Path("$(SUMMARIES_DIR)"); outd.mkdir(parents=True, exist_ok=True)
csv_path = outd/"fid_grid_best_per_model_backbone_img.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(best.values())
md_path = outd/"fid_grid_best_per_model_backbone_img.md"
cols = ["model","backbone","img_size","per_class_cap","total_cap","batch","fid","job_id","task_id","log"]
with md_path.open("w") as f:
    f.write("| " + " | ".join(cols) + " |\\n")
    f.write("|" + "|".join(["---"]*len(cols)) + "|\\n")
    for key in sorted(best):
        r = best[key]
        f.write("| " + " | ".join(r.get(c,"") for c in cols) + " |\\n")
print("Wrote", csv_path, "and", md_path)
PY

fid-plot-3d: fid-best-3d
\tMPLBACKEND=Agg $(PY) scripts/metrics/fid_plot.py "$(SUMMARIES_DIR)/fid_grid_best_per_model_backbone_img.csv" "$(SUMMARIES_DIR)/fid_best_bar_backbone_img.png"

fid-show: fid-parse
\t@(head -n1 "$(FID_COMBINED_CSV)"; tail -n +2 "$(FID_COMBINED_CSV)" | sort -t, -k9,9g) | column -t -s,
