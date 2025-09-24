# scripts/backfill_ms_ssim_per_class.py

import json, numpy as np, tensorflow as tf
from pathlib import Path
from PIL import Image

def safe_resize(x):
    # NHWC float32
    H,W = x.shape[1], x.shape[2]
    if min(H,W) < 11:
        x = tf.image.resize(tf.convert_to_tensor(x), [max(11,H), max(11,W)], method="nearest").numpy()
    return x

def ms_ssim(a,b,fs):
    try:
        v = tf.image.ssim_multiscale(a,b,max_val=1.0,filter_size=fs)
    except Exception:
        v = tf.image.ssim(a,b,max_val=1.0,filter_size=fs)
    return float(tf.reduce_mean(v).numpy())

def load_manifest_items(model):
    man = Path("artifacts")/model/"synthetic"/"manifest.json"
    d = json.loads(man.read_text())
    return d.get("paths") or d.get("samples") or []

def load_x_y(items):
    xs, ys = [], []
    for it in items:
        im = Image.open(it["path"]).convert("RGB")
        xs.append(np.asarray(im, dtype=np.float32)/255.0)
        ys.append(int(it["label"]))
    x = np.stack(xs); y = np.array(ys)
    if x.ndim==3: x = x[...,None]
    if x.shape[-1]==1: x = np.repeat(x,3,axis=-1)
    x = safe_resize(x)
    H,W = x.shape[1], x.shape[2]
    fs = min(11,H,W); fs = fs-1 if fs%2==0 else fs; fs = max(fs,3)
    return x,y,fs

def per_class_ms_ssim(x,y,fs,cap_pairs=50):
    out = {}
    rng = np.random.default_rng(0)
    for cls in sorted(set(y)):
        idx = np.where(y==cls)[0]
        if len(idx) < 2:
            out[str(cls)] = None
            continue
        n_pairs = min(cap_pairs, len(idx)*(len(idx)-1)//2)
        vals=[]
        for _ in range(n_pairs):
            i,j = rng.choice(idx, size=2, replace=False)
            vals.append(ms_ssim(x[i:i+1], x[j:j+1], fs))
        out[str(cls)] = float(np.mean(vals))
    return out

def latest_summary(model):
    sdir = Path("artifacts")/model/"summaries"
    files = sorted(sdir.glob("summary_*.json"))
    return files[-1] if files else None

if __name__ == "__main__":
    models = ["gan","vae","gaussianmixture","diffusion","autoregressive","restrictedboltzmann","maskedautoflow"]
    for m in models:
        man_items = load_manifest_items(m)
        if not man_items:
            print(f"[skip] {m}: no items")
            continue
        x,y,fs = load_x_y(man_items)
        pcs = per_class_ms_ssim(x,y,fs)
        summ = latest_summary(m)
        if not summ:
            print(f"[skip] {m}: no summary")
            continue
        d = json.loads(summ.read_text())
        d.setdefault("metrics",{})["ms_ssim_per_class"] = pcs
        d["metrics"].setdefault("_warnings", []).append("Per-class MS-SSIM backfilled locally.")
        summ.write_text(json.dumps(d, indent=2))
        print(f"[ok] {m}: wrote per-class MS-SSIM -> {summ}")
