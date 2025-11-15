import glob, json, os, sys
mods = ["gan","diffusion","autoregressive","vae","gaussianmixture","restrictedboltzmann","maskedautoflow"]
rows=[]
for m in mods:
    pat = os.path.expanduser(f"~/gencys/artifacts/{m}/summaries/summary_*.json")
    fs  = sorted(glob.glob(pat))
    j   = json.load(open(fs[-1])) if fs else {}
    g   = j.get("generative") or {}
    rows.append((m, g.get("cfid_macro"), g.get("kid"), g.get("ms_ssim")))
rows.sort(key=lambda r: (float('inf') if r[1] is None else r[1]))
print(f"{'model':20s} {'cFID':>12} {'KID':>12} {'MS-SSIM':>12}")
fmt = lambda x: "-" if x is None else f"{x:.6f}"
for m,cfid,kid,mss in rows:
    print(f"{m:20s} {fmt(cfid):>12} {fmt(kid):>12} {fmt(mss):>12}")
