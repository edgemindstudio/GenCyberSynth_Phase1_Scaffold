# tools/check_phase1_integrity.py
import json, os, sys
from pathlib import Path

NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "9"))

ALLOWED = os.environ.get("PHASE1_ALLOWED_BUDGETS", "500,1000,2000").strip()
ALLOWED_BUDGETS = {int(x) for x in ALLOWED.split(",") if x.strip()}

# Which summaries are we validating?
#   paper1.json -> frozen “paper” snapshot
#   latest.json -> newest run (dev/smoke)
SUMMARY_NAME = os.environ.get("PHASE1_SUMMARY_NAME", "paper1.json").strip()

def get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

bad = []
missing = []

for summ in Path("artifacts").glob(f"*/summaries/{SUMMARY_NAME}"):
    d = json.loads(summ.read_text())
    rm = d.get("run_meta") if isinstance(d.get("run_meta"), dict) else {}

    model = d.get("model") or summ.parts[-3]
    bpc = d.get("budget_per_class") or rm.get("budget_per_class")
    nf  = d.get("counts.num_fake") or get(d, "counts", "num_fake") or get(d, "counts", "synthetic")

    if bpc is None or nf is None:
        bad.append((model, "missing budget_per_class or num_fake", bpc, nf, str(summ)))
        continue

    bpc = int(bpc)
    nf = int(nf)

    if bpc not in ALLOWED_BUDGETS:
        bad.append((model, f"budget_per_class={bpc} not in allowed={sorted(ALLOWED_BUDGETS)}", bpc, nf, str(summ)))

    exp = bpc * NUM_CLASSES
    if nf != exp:
        bad.append((model, f"num_fake={nf} expected={exp}", bpc, nf, str(summ)))

# Optional: ensure every model has the snapshot file
# (Only enforce when checking paper snapshots)
if SUMMARY_NAME == "paper1.json":
    for model_dir in Path("artifacts").iterdir():
        if model_dir.is_dir() and (model_dir / "summaries").exists():
            if not (model_dir / "summaries" / "paper1.json").exists():
                missing.append(model_dir.name)

print("bad:", len(bad))
for r in bad:
    print(" -", r)

if missing:
    print("missing paper1 snapshots:", missing)

sys.exit(1 if bad or missing else 0)


