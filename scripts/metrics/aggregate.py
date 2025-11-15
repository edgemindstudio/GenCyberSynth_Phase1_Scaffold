#scripts/metrics/aggregate.py

#!/usr/bin/env python3
import argparse, glob, json, math, os, csv
from typing import Any, Dict, Iterable, Optional

# Pretty-name mapping (adapter -> Phase-1 table names)
PRETTY = {
    "gan": "ConditionalDCGAN",
    "vae": "ConditionalVAE",
    "autoregressive": "ConditionalAutoregressive",
    "diffusion": "ConditionalDiffusion",
    "gaussianmixture": "GaussianMixture",
    "restrictedboltzmann": "RestrictedBoltzmann",
    "maskedautoflow": "MaskedAutoflow",
}
# Also accept already-pretty names
for v in list(PRETTY.values()):
    PRETTY[v] = v

def normalize_name(m: Optional[str], path_hint: Optional[str] = None) -> str:
    """Map adapter names (or infer from path) to pretty Phase-1 names."""
    if not m and path_hint:
        parts = path_hint.split("/")
        for k in PRETTY:
            if k in parts:
                m = k
                break
    if m in PRETTY:
        return PRETTY[m]
    low = (m or "").lower()
    if low in PRETTY:
        return PRETTY[low]
    return m or "unknown"

def dict_get(d: Dict[str, Any], path: Iterable[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def first_num(*vals):
    for v in vals:
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return float(v)
    return None

def mean_or_none(x):
    if isinstance(x, (list, tuple)) and x:
        xs = [float(v) for v in x if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
        return sum(xs) / len(xs) if xs else None
    return None

def pull_metrics(j: Dict[str, Any]):
    """
    Extract FID, NN-dist-mean, and RS−R deltas from many possible layouts.
    - FID: prefer fid; fallback to fid_macro
    - NN: prefer explicit mean; fallback to mean(nn_dists) if present
    """
    # FID
    fid = first_num(
        j.get("fid"),
        dict_get(j, ("generative", "fid")),
        dict_get(j, ("metrics", "fid")),
        dict_get(j, ("generative", "fid_macro")),
        dict_get(j, ("metrics", "fid_macro")),
    )

    # NN distance mean
    mem = j.get("memorization") or {}
    met = j.get("metrics") or {}
    nn = first_num(
        mem.get("nn_dist_mean"),
        met.get("nn_dist_mean"),
        mem.get("nn_mean"),
        met.get("nn_mean"),
        j.get("nn_dist_mean"),
    )
    if nn is None:
        nn = mean_or_none(mem.get("nn_dists")) or mean_or_none(met.get("nn_dists"))

    # RS−R deltas
    deltas = j.get("deltas_RS_minus_R") or {}
    da = deltas.get("accuracy")
    df = deltas.get("macro_f1")
    return fid, nn, da, df

def collect_files(arts_root: str, phase1_jsonl: Optional[str]):
    files = []
    for m in ["autoregressive","vae","gaussianmixture","restrictedboltzmann","maskedautoflow","diffusion","gan"]:
        files += glob.glob(f"{arts_root}/{m}/summaries/summary_*.json")
    if phase1_jsonl and os.path.exists(phase1_jsonl):
        files.append(phase1_jsonl)
    return files

def parse_file(path: str, best: Dict[str, Dict[str, Any]]) -> None:
    def upd(name, fid, nn, da, df):
        if not name:
            return
        d = best.setdefault(name, {"fid": math.inf, "nn": math.inf, "da": None, "df": None})
        if isinstance(fid, (int, float)) and fid < d["fid"]:
            d["fid"] = fid
        if isinstance(nn, (int, float)) and nn < d["nn"]:
            d["nn"] = nn
        if isinstance(da, (int, float)):
            d["da"] = da
        if isinstance(df, (int, float)):
            d["df"] = df

    if path.endswith(".jsonl"):
        with open(path, "r", errors="ignore") as f:
            for ln in f:
                try:
                    j = json.loads(ln)
                except Exception:
                    continue
                raw = j.get("model") or j.get("model_name") or j.get("adapter") or j.get("repo")
                name = normalize_name(raw, path_hint=path)
                fid, nn, da, df = pull_metrics(j)
                upd(name, fid, nn, da, df)
        return

    try:
        with open(path, "r") as f:
            j = json.load(f)
    except Exception:
        return
    raw = j.get("model") or j.get("adapter") or j.get("repo")
    name = normalize_name(raw, path_hint=path)
    fid, nn, da, df = pull_metrics(j)
    upd(name, fid, nn, da, df)

def main():
    ap = argparse.ArgumentParser(description="Aggregate best metrics across models.")
    ap.add_argument("--artifacts", required=True, help="Artifacts root (per-model subdirs).")
    ap.add_argument("--phase1", default=None, help="Optional Phase-1 summaries .jsonl")
    ap.add_argument("--out_csv", default="artifacts_summary.csv")
    args = ap.parse_args()

    files = collect_files(args.artifacts, args.phase1)
    if not files:
        print("No summary files found. Check logs for 'Saved evaluation summary'.")
        return

    best: Dict[str, Dict[str, Any]] = {}
    for p in files:
        parse_file(p, best)

    print("Model                      best_FID   best_NNdist   ΔAcc(RS−R)   ΔF1(RS−R)")
    rows = []
    for m in sorted(best):
        b = best[m]
        f  = "-" if math.isinf(b["fid"]) else f"{b['fid']:.3f}"
        nn = "-" if math.isinf(b["nn"]) else f"{b['nn']:.4f}"
        da = "-" if b["da"] is None else f"{b['da']:+.6f}"
        df = "-" if b["df"] is None else f"{b['df']:+.6f}"
        print(f"{m:26s} {f:>9}   {nn:>10}   {da:>10}   {df:>10}")
        rows.append([
            m,
            "" if math.isinf(b["fid"]) else f"{b['fid']:.6f}",
            "" if math.isinf(b["nn"]) else f"{b['nn']:.6f}",
            "" if b["da"] is None else f"{b['da']:.6f}",
            "" if b["df"] is None else f"{b['df']:.6f}",
        ])

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model","best_FID","best_NNdist","DeltaAcc_RSminusR","DeltaF1_RSminusR"])
        w.writerows(rows)
    print(f"\nWrote CSV → {os.path.abspath(args.out_csv)}")

if __name__ == "__main__":
    main()
