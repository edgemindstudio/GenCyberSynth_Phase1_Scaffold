# scripts/phase1_report.py

from pathlib import Path
import json, math

MODELS = ["gan","vae","gaussianmixture","diffusion","autoregressive","restrictedboltzmann","maskedautoflow"]
ART = Path("artifacts")
out_md = ART/"phase1_report.md"

def latest_summary(m):
    sdir = ART/m/"summaries"
    files = sorted(sdir.glob("summary_*.json"))
    return files[-1] if files else None

rows=[]
for m in MODELS:
    f = latest_summary(m)
    if not f:
        rows.append((m, None, None))
        continue
    d = json.loads(f.read_text())
    ms = d.get("metrics",{}).get("ms_ssim")
    nf = d.get("counts",{}).get("num_fake")
    rows.append((m, ms, nf))

# sort by ms_ssim (None last)
rows = sorted(rows, key=lambda r: (math.inf if r[1] is None else r[1]))

with open(out_md, "w") as w:
    w.write("# Phase 1 – Synth Evaluation (Snapshot)\n\n")
    w.write("| Model | MS-SSIM ↓ | #Fake |\n|---|---:|---:|\n")
    for m,ms,nf in rows:
        w.write(f"| {m} | {ms if ms is not None else '—'} | {nf if nf is not None else '—'} |\n")
    w.write("\n**Notes**\n")
    w.write("- Lower MS-SSIM indicates higher intra-class diversity.\n")
    w.write("- Extremely low values can correspond to noisy samples; grids were visually checked.\n")
    w.write("- Metrics requiring `gcs_core` backends (KID/CFID) are pending environment setup.\n")

print(f"Wrote {out_md}")
