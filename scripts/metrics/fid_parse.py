import re, csv, glob, pathlib, sys

out_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/summaries/fid_grid_combined.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted(glob.glob("slurm/logs/gcs-fidkid_*_*.out")):
    m = re.search(r'gcs-fidkid_(\d+)_(\d+)\.out$', f)
    if not m:
        continue
    job_id, task_id = m.group(1), m.group(2)
    text = pathlib.Path(f).read_text(errors="ignore")

    mm = re.search(
        r'\[fidkid[^\]]*\]\s+model=([a-z0-9]+).*?\bbackbone=([A-Za-z0-9_-]+)?\b.*?\bimg=(\d+)?\b.*?\bper_class_cap=(\d+).*?\btotal_cap=(\d+).*?\bbatch=(\d+)',
        text
    )
    if not mm:
        continue
    model, backbone, img_size, per_cap, tot_cap, batch = mm.groups()
    backbone = backbone or "mobilenet"
    img_size = img_size or "160"

    fid_matches = list(re.finditer(r'\[fidkid\].*?FID=([0-9.]+)', text))
    if not fid_matches:
        continue
    fid = fid_matches[-1].group(1)

    rows.append([job_id, task_id, model, backbone, img_size, per_cap, tot_cap, batch, fid, f])

rows.sort(key=lambda r: (int(r[0]), int(r[1])))

with out_path.open("w", newline="") as w:
    writer = csv.writer(w)
    writer.writerow(["job_id","task_id","model","backbone","img_size","per_class_cap","total_cap","batch","fid","log"])
    writer.writerows(rows)

print(f"Wrote {out_path} with {len(rows)} rows.")
