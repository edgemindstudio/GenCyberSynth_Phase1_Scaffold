# tools/build_phase1_scores.py
import os, json
from pathlib import Path
import pandas as pd

def g(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# Choose which summary file to read:
#   latest.json  -> whatever is newest (can be 25/class after smoke)
#   paper1.json  -> frozen snapshot (e.g., 2000/class)
SUMMARY_NAME = os.environ.get("PHASE1_SUMMARY_NAME", "latest.json").strip()

rows = []
for p in Path("artifacts").glob(f"*/summaries/{SUMMARY_NAME}"):
    d = json.loads(p.read_text())
    model = d.get("model") or p.parts[-3]
    rm = d.get("run_meta") if isinstance(d.get("run_meta"), dict) else {}

    bpc = d.get("budget_per_class") or rm.get("budget_per_class")
    num_fake = (
        d.get("counts.num_fake")
        or g(d, "counts", "num_fake")
        or g(d, "counts", "synthetic")
        or g(d, "counts", "synthetic_count")
    )

    kid = d.get("metrics.kid") or g(d, "generative", "kid")
    cfid = d.get("metrics.cfid") or d.get("metrics.cfid_macro") or g(d, "generative", "cfid_macro")
    ms_ssim = d.get("metrics.ms_ssim") or g(d, "generative", "ms_ssim")
    fid = d.get("metrics.fid") or g(d, "generative", "fid")

    rows.append({
        "model": model,
        "seed": d.get("seed"),
        "budget_per_class": int(bpc) if bpc is not None else None,
        "num_fake": int(num_fake) if num_fake is not None else None,
        "config_path": d.get("config_path") or rm.get("config_path"),
        "config_sha1": d.get("config_sha1") or rm.get("config_sha1"),
        "git_commit": d.get("git_commit") or rm.get("git_commit"),
        "kid": kid,
        "cfid": cfid,
        "ms_ssim": ms_ssim,
        "fid": fid,
        "summary_path": str(p),
    })

df = pd.DataFrame(rows).sort_values(["model"])
out = Path("artifacts/phase1_scores.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print("wrote", out, "rows=", len(df))
print("budgets:", sorted(df["budget_per_class"].dropna().unique().tolist()))
print("num_fake:", sorted(df["num_fake"].dropna().unique().tolist()))
print("using summaries:", SUMMARY_NAME)

