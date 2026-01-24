# tools/freeze_phase1_snapshots.py
import json, os, shutil
from pathlib import Path

def parse_allowed(s: str):
    # supports "2000" or "500 1000 2000" or "500,1000,2000"
    toks = [t for t in s.replace(",", " ").split() if t.strip()]
    return {int(t) for t in toks}

ALLOWED = parse_allowed(os.getenv("PHASE1_ALLOWED_BUDGETS", "2000"))
SUMMARY_NAME  = os.getenv("PHASE1_SUMMARY_NAME", "paper1.json").strip()

# Manifest lock
MANIFEST_NAME = os.getenv("PHASE1_MANIFEST_NAME", "paper1_manifest.json").strip()
NUM_CLASSES   = int(os.getenv("NUM_CLASSES", "9"))
# Seed folder under artifacts/<model>/synthetic/<class>/<seed>/*.png
MANIFEST_SEED = os.getenv("PHASE1_MANIFEST_SEED", "42").strip()  # set to "all" to include all seeds

def pick_best_summary(model_dir: Path):
    cands = sorted((model_dir / "summaries").glob("summary_*.json"), reverse=True)
    for p in cands:
        try:
            d = json.loads(p.read_text())
            rm = d.get("run_meta") or {}
            b = d.get("budget_per_class") or rm.get("budget_per_class")
            if b is not None and int(b) in ALLOWED:
                return p
        except Exception:
            pass
    return None

def _collect_pngs_for_class(model_dir: Path, cls: int):
    syn = model_dir / "synthetic" / str(cls)
    if not syn.exists():
        return []

    if MANIFEST_SEED.lower() == "all":
        # gather from all seed subfolders
        pngs = sorted(syn.rglob("*.png"))
        return pngs

    # gather from a single seed subfolder
    seed_dir = syn / MANIFEST_SEED
    if not seed_dir.exists():
        return []
    return sorted(seed_dir.glob("*.png"))

def build_manifest_from_pngs(model_dir: Path):
    """
    Build a paper-locked manifest from PNGs on disk:
      artifacts/<model>/synthetic/<class>/<seed>/*.png
    Writes:
      artifacts/<model>/synthetic/<MANIFEST_NAME>
    """
    per_class = {}
    for c in range(NUM_CLASSES):
        pngs = _collect_pngs_for_class(model_dir, c)
        per_class[c] = pngs

    counts = {c: len(v) for c, v in per_class.items()}
    if any(counts[c] == 0 for c in range(NUM_CLASSES)):
        missing = [c for c in range(NUM_CLASSES) if counts[c] == 0]
        raise RuntimeError(
            f"{model_dir.name}: missing PNGs for classes {missing} "
            f"(seed={MANIFEST_SEED}). counts={counts}"
        )

    budget_per_class = min(counts.values())  # safest: use the common minimum across classes
    if budget_per_class not in ALLOWED:
        raise RuntimeError(
            f"{model_dir.name}: derived budget_per_class={budget_per_class} not in allowed={sorted(ALLOWED)} "
            f"(seed={MANIFEST_SEED}). counts={counts}"
        )

    items = []
    for c in range(NUM_CLASSES):
        # deterministic: sorted paths, take first budget_per_class
        for p in per_class[c][:budget_per_class]:
            items.append({"label": int(c), "path": str(p)})

    out = {
        "budget_per_class": int(budget_per_class),
        "num_fake": int(len(items)),
        "paths": items,
        "counts_per_class": {str(k): int(v) for k, v in counts.items()},
        "seed_lock": MANIFEST_SEED,
        "source": "freeze_phase1_snapshots.py (rebuild from PNGs)",
    }

    out_path = model_dir / "synthetic" / MANIFEST_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out_path, budget_per_class, len(items), counts

root = Path("artifacts")
picked, missing = [], []
manifest_written = []

# 1) Freeze summaries
for model_dir in sorted(root.iterdir()):
    if not model_dir.is_dir():
        continue
    if not (model_dir / "summaries").exists():
        continue

    best = pick_best_summary(model_dir)
    if best is None:
        missing.append(model_dir.name)
        continue

    out = model_dir / "summaries" / SUMMARY_NAME
    shutil.copy2(best, out)
    picked.append((model_dir.name, best.name))

print(f"wrote {len(picked)} snapshot(s) -> {SUMMARY_NAME} (allowed={sorted(ALLOWED)})")
for m, src in sorted(picked):
    print(" -", m, "<-", src)
if missing:
    print("missing models (no allowed budget found):", missing)

# 2) Rebuild manifests from PNGs (paper lock)
for model_dir in sorted(root.iterdir()):
    if not model_dir.is_dir():
        continue
    syn_dir = model_dir / "synthetic"
    if not syn_dir.exists():
        continue

    try:
        out_path, bpc, nf, counts = build_manifest_from_pngs(model_dir)
        manifest_written.append((model_dir.name, out_path.name, bpc, nf))
    except Exception as e:
        raise SystemExit(f"[FAIL] manifest build: {e}")

print(f"wrote {len(manifest_written)} manifest(s) -> {MANIFEST_NAME}")
for m, name, bpc, nf in manifest_written:
    print(f" - {m} <- {name} (bpc={bpc} num_fake={nf})")
