# scripts/make_grids.py

from pathlib import Path
import json, numpy as np
from PIL import Image, ImageDraw, ImageFont

MODELS = ["gan","vae","gaussianmixture","diffusion","autoregressive","restrictedboltzmann","maskedautoflow"]
ART = Path("artifacts")
OUT = ART / "preview_grids"
OUT.mkdir(parents=True, exist_ok=True)

def pad_to(img, size):
    # img: HxWxC float[0,1]; size: (H,W)
    H,W = size
    h,w = img.shape[:2]
    if (h,w)==(H,W): return img
    pil = Image.fromarray((img*255).astype("uint8"))
    pil = pil.resize((W,H), Image.NEAREST)
    return np.asarray(pil).astype("float32")/255.0

def load_items(man_path):
    man = json.loads(Path(man_path).read_text())
    items = man.get("paths") or man.get("samples") or []
    return items

def load_image_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32)/255.0

def grid_for_model(model, per_class=25, classes=9, tile=5, cell=40):
    man = ART/model/"synthetic"/"manifest.json"
    if not man.exists():
        print(f"[skip] {model}: no manifest")
        return
    items = load_items(man)
    by_c = {c:[] for c in range(classes)}
    for it in items:
        c = int(it["label"])
        if len(by_c[c]) < per_class:
            by_c[c].append(it["path"])
    if not any(by_c.values()):
        print(f"[skip] {model}: empty")
        return

    # build a tall grid: one row per class, tile x tile samples per row
    rows_img = []
    for c in range(classes):
        paths = by_c.get(c, [])[:tile*tile]
        if not paths: continue
        imgs=[]
        for p in paths:
            x = load_image_rgb(p)
            x = pad_to(x, (cell, cell))
            imgs.append(x)
        # fill if not enough
        while len(imgs) < tile*tile:
            imgs.append(np.ones((cell,cell,3),dtype=np.float32))  # white filler
        row = []
        for r in range(tile):
            row.append(np.concatenate(imgs[r*tile:(r+1)*tile], axis=1))
        rows_img.append(np.concatenate(row, axis=0))
    if not rows_img:
        print(f"[skip] {model}: no rows")
        return
    full = np.concatenate(rows_img, axis=0)
    out = (full*255).astype("uint8")
    Image.fromarray(out).save(OUT/f"{model}_grid.png")
    print(f"[ok] wrote {OUT/f'{model}_grid.png'}")

if __name__ == "__main__":
    for m in MODELS:
        grid_for_model(m)
