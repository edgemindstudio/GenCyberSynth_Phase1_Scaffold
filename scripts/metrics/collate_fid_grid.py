#!/usr/bin/env python3
import csv, glob, os, pathlib

root = pathlib.Path("artifacts/summaries")
root.mkdir(parents=True, exist_ok=True)

# collect all grid CSVs (in case you run multiple arrays)
rows = []
for p in sorted(glob.glob(str(root / "fid_grid_runs.csv"))):
    with open(p, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            # normalize types
            try: r["fid"] = float(r["fid"]) if r["fid"] not in ("", "NA", None) else None
            except: r["fid"] = None
            for k in ("img_size","per_class_cap","total_cap","batch"):
                try: r[k] = int(r[k])
                except: pass
            rows.append(r)

# write combined CSV
out_csv = root / "fid_grid_combined.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "job_id","task_id","model","backbone","img_size","per_class_cap","total_cap","batch","fid"
    ])
    w.writeheader()
    w.writerows(rows)

# write markdown sorted by (backbone, per_class_cap, fid)
def key_func(r):
    fid = r["fid"]
    return (r["backbone"], int(r["per_class_cap"]), float("inf") if fid is None else fid)

rows_sorted = sorted(rows, key=key_func)

out_md = root / "fid_grid_combined.md"
with open(out_md, "w") as w:
    w.write("| backbone | per_class | model | cFID | img | batch |\n")
    w.write("|---|---:|---|---:|---:|---:|\n")
    def fmt(x): 
        if isinstance(x,(float,int)): 
            try: return f"{float(x):.6f}"
            except: return str(x)
        return "-" if x in ("", None, "NA") else str(x)
    for r in rows_sorted:
        w.write(f"| {r['backbone']} | {r['per_class_cap']} | {r['model']} | {fmt(r['fid'])} | {r['img_size']} | {r['batch']} |\n")

print(f"Wrote {out_csv}")
print(f"Wrote {out_md}")
