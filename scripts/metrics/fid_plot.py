import csv, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

best_csv = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("artifacts/summaries/fid_grid_best_per_model.csv")
out_png  = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("artifacts/summaries/fid_best_bar.png")

rows = list(csv.DictReader(best_csv.open()))
if not rows:
    raise SystemExit(f"No rows in {best_csv}")

rows.sort(key=lambda r: float(r["fid"]))
labels = [r["model"] for r in rows]
vals   = [float(r["fid"]) for r in rows]

plt.figure(figsize=(8,4))
plt.bar(labels, vals)
plt.title("Best cFID per model")
plt.ylabel("cFID (lower is better)")
plt.xticks(rotation=30, ha="right")
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout(); plt.savefig(out_png, dpi=150)
print("Wrote", out_png)
