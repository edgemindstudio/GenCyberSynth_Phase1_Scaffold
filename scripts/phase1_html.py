# scripts/phase1_html.py
from pathlib import Path
import json, math, html

MODELS = ["gan","vae","gaussianmixture","diffusion","autoregressive","restrictedboltzmann","maskedautoflow"]
ART = Path("artifacts")
OUT = ART/"phase1_report.html"

def latest_summary(m):
    files = sorted((ART/m/"summaries").glob("summary_*.json"))
    return files[-1] if files else None

def row(m):
    s = latest_summary(m)
    ms = nf = None
    if s:
        d = json.loads(s.read_text())
        ms = d.get("metrics",{}).get("ms_ssim")
        nf = d.get("counts",{}).get("num_fake")
    return (m, ms, nf)

rows = [row(m) for m in MODELS]
rows_sorted = sorted(rows, key=lambda r: (math.inf if r[1] is None else r[1]))

def td(x):
    return "—" if x is None else (f"{x:.6g}" if isinstance(x, float) else html.escape(str(x)))

html_rows = "\n".join(
    f"<tr><td>{html.escape(m)}</td><td style='text-align:right'>{td(ms)}</td><td style='text-align:right'>{td(nf)}</td></tr>"
    for m,ms,nf in rows_sorted
)

# embed grids if present
grid_imgs = []
for m in MODELS:
    p = ART/"preview_grids"/f"{m}_grid.png"
    if p.exists():
        grid_imgs.append(f"<h3>{html.escape(m)}</h3><img src='{p.as_posix()}' style='max-width:100%;image-rendering:pixelated;border:1px solid #ddd'/>")

doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Phase 1 – Synth Evaluation</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{border:1px solid #ddd;padding:8px}}
th{{background:#fafafa;text-align:left}}
.note{{color:#555}}
</style>
<h1>Phase 1 – Synth Evaluation (Snapshot)</h1>
<p class="note">Lower MS-SSIM ⇒ higher intra-class diversity. Extremely low values can indicate noise; verify with the grids below.</p>
<table>
  <thead><tr><th>Model</th><th style="text-align:right">MS-SSIM ↓</th><th style="text-align:right">#Fake</th></tr></thead>
  <tbody>
  {html_rows}
  </tbody>
</table>
<h2>Per-model grids</h2>
{''.join(grid_imgs)}
"""
OUT.write_text(doc, encoding="utf-8")
print(f"Wrote {OUT}")
