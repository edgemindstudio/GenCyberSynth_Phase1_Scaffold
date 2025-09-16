# tools/backfill_rs_metrics.py
"""
Backfill REAL+SYNTH ('utility_real_plus_synth') metrics into Phase-1 eval summaries
without TensorFlow: trains a tiny NumPy softmax probe on (Real) and (Real+Synth).

Why this script?
- Some earlier eval summaries were written without the RS block.
- This script finds each repo’s latest *_eval_summary_seed*.json,
  resolves DATA_DIR correctly (even if it's relative), loads synthetic dumps,
  trains a NumPy softmax probe, and writes:
    - "utility_real_plus_synth"
    - "images.synthetic"
    - "deltas_RS_minus_R"
  back into the JSON.

Run:
  python tools/backfill_rs_metrics.py --epochs 8 --cap-per-class 4000
Then:
  python tools/aggregate_phase1.py --diagnose
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time
from typing import Dict, Tuple, Optional, List

import numpy as np
import yaml


# -----------------------------------------------------------------------------
# Repo discovery / defaults
# -----------------------------------------------------------------------------
DEFAULT_BASE = Path.home() / "PycharmProjects"
REPOS = [
    "GAN",
    "VAEs",
    "AUTOREGRESSIVE",
    "MASKEDAUTOFLOW",
    "RESTRICTEDBOLTZMANN",
    "GAUSSIANMIXTURE",
    "DIFFUSION",
]
SUMMARY_GLOBS = [
    "artifacts/**/summaries/*_eval_summary_seed*.json",
    "artifacts/*/summaries/*_eval_summary_seed*.json",  # fallback
]
CONFIG_CANDIDATES = ["config.yaml", "configs/default.yaml"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def _pick(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _load_yaml(p: Path) -> Dict:
    with open(p, "r") as f:
        return yaml.safe_load(f)


def _newest_json(repo_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for g in SUMMARY_GLOBS:
        candidates.extend(repo_dir.glob(g))
    if not candidates:
        return None
    return max(candidates, key=lambda pp: pp.stat().st_mtime)


def _find_config(repo_dir: Path) -> Optional[Path]:
    for rel in CONFIG_CANDIDATES:
        p = repo_dir / rel
        if p.exists():
            return p
    return None


def _one_hot_from_any(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Accept integers or one-hot; return one-hot [N, K] float32."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32")
    y = y.astype(int).ravel()
    return np.eye(num_classes, dtype="float32")[y]


def _to_01_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    H, W, C = img_shape
    x = x.astype("float32")
    if x.size == 0:
        return x.reshape((-1, H, W, C))
    if x.max() > 1.5:  # bytes -> [0,1]
        x = x / 255.0
    x = x.reshape((-1, H, W, C))
    return np.clip(x, 0.0, 1.0)


def _resolve_data_dir(
    repo_dir: Path,
    cfg: Dict,
    data_root: Optional[Path],
) -> Path:
    """
    Resolve DATA_DIR robustly:
      1) absolute path => use as-is
      2) repo_dir / DATA_DIR
      3) repo_dir.parent / DATA_DIR         (e.g., ~/PycharmProjects/<dataset>)
      4) data_root / DATA_DIR               (if --data-root is provided)
      5) Path.home() / DATA_DIR
    Must contain train_data.npy/test_data.npy.
    """
    dd = Path(str(cfg["DATA_DIR"]))
    candidates: List[Path] = []
    if dd.is_absolute():
        candidates.append(dd)
    else:
        candidates.extend([
            repo_dir / dd,
            repo_dir.parent / dd,
        ])
        if data_root:
            candidates.append(Path(data_root) / dd)
        candidates.append(Path.home() / dd)

    checked = []
    for cand in candidates:
        td = cand / "train_data.npy"
        te = cand / "test_data.npy"
        checked.append(str(cand))
        if td.exists() and te.exists():
            return cand

    raise FileNotFoundError(
        "DATA_DIR could not be resolved.\n"
        f"  cfg.DATA_DIR = {cfg['DATA_DIR']}\n"
        "  Tried:\n    - " + "\n    - ".join(checked) + "\n"
        "  Fix: place the dataset in one of the above, or pass --data-root <folder>."
    )


def _load_real_from_cfg(
    repo_dir: Path,
    cfg: Dict,
    data_root: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = _resolve_data_dir(repo_dir, cfg, data_root)
    img_shape = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])
    val_fraction = float(cfg.get("VAL_FRACTION", 0.5))

    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test = np.load(data_dir / "test_data.npy")
    y_test = np.load(data_dir / "test_labels.npy")

    x_train01 = _to_01_hwc(x_train, img_shape)
    x_test01  = _to_01_hwc(x_test,  img_shape)

    y_train1h = _one_hot_from_any(y_train, num_classes)
    y_test1h  = _one_hot_from_any(y_test,  num_classes)

    n_val = int(len(x_test01) * val_fraction)
    x_val01, y_val1h   = x_test01[:n_val], y_test1h[:n_val]
    x_test01, y_test1h = x_test01[n_val:], y_test1h[n_val:]

    return x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h


def _load_synth(repo_dir: Path, cfg: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Look for combined x_synth/y_synth, else per-class NPYS."""
    img_shape = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])
    synth_dir = Path(_pick(cfg, "ARTIFACTS", "synthetic", default="artifacts/synthetic"))
    if not synth_dir.is_absolute():
        synth_dir = (repo_dir / synth_dir).resolve()

    # Combined dumps
    x_all = synth_dir / "x_synth.npy"
    y_all = synth_dir / "y_synth.npy"
    if x_all.exists() and y_all.exists():
        try:
            x = np.load(x_all)
            y = np.load(y_all)
            H, W, C = img_shape
            x = x.reshape((-1, H, W, C)).astype("float32")
            y = _one_hot_from_any(y, num_classes)
            return x, y
        except Exception:
            pass

    # Per-class dumps
    xs, ys = [], []
    for k in range(num_classes):
        xp = synth_dir / f"gen_class_{k}.npy"
        yp = synth_dir / f"labels_class_{k}.npy"
        if xp.exists() and yp.exists():
            xs.append(np.load(xp))
            ys.append(_one_hot_from_any(np.load(yp), num_classes))
    if xs:
        H, W, C = img_shape
        x = np.concatenate(xs, 0).reshape((-1, H, W, C)).astype("float32")
        y = np.concatenate(ys, 0).astype("float32")
        return x, y

    return None, None


def _cap_per_class(x: np.ndarray, y1h: np.ndarray, cap: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if not cap:
        return x, y1h
    labels = y1h.argmax(-1)
    keep_idx = []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        if len(idx) > cap:
            idx = np.random.choice(idx, cap, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx, 0)
    np.random.shuffle(keep_idx)
    return x[keep_idx], y1h[keep_idx]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _accuracy(y_true, y_prob):
    return float((y_prob.argmax(-1) == y_true.argmax(-1)).mean())


def _macro_f1(y_true, y_prob, eps=1e-8):
    pred = y_prob.argmax(-1)
    true = y_true.argmax(-1)
    K = y_true.shape[1]
    f1s = []
    for k in range(K):
        tp = np.sum((pred == k) & (true == k))
        fp = np.sum((pred == k) & (true != k))
        fn = np.sum((pred != k) & (true == k))
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1s.append(2 * prec * rec / (prec + rec + eps))
    return float(np.mean(f1s))


def _balanced_accuracy(y_true, y_prob, eps=1e-8):
    pred = y_prob.argmax(-1)
    true = y_true.argmax(-1)
    K = y_true.shape[1]
    recalls = []
    for k in range(K):
        idx = np.where(true == k)[0]
        if idx.size == 0:
            continue
        recalls.append(np.mean(pred[idx] == k))
    return float(np.mean(recalls)) if recalls else float("nan")


def _ece(y_true, y_prob, bins=10):
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    true = y_true.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        m = mask.sum()
        if m == 0:
            continue
        acc = np.mean(pred[mask] == true[mask])
        avg_conf = np.mean(conf[mask])
        ece += (m / n) * abs(acc - avg_conf)
    return float(ece)


def _brier(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def _metrics(y_true, y_prob) -> Dict[str, Optional[float]]:
    return {
        "accuracy": _accuracy(y_true, y_prob),
        "macro_f1": _macro_f1(y_true, y_prob),
        "balanced_accuracy": _balanced_accuracy(y_true, y_prob),
        "ece": _ece(y_true, y_prob),
        "brier": _brier(y_true, y_prob),
        # Placeholders retained for schema compatibility; aggregator tolerates None.
        "macro_auprc": None,
        "recall_at_1pct_fpr": None,
    }


# -----------------------------------------------------------------------------
# NumPy softmax (multinomial logistic) probe
# -----------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # stability
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def _train_softmax(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    lr: float,
    l2: float,
    epochs: int,
    batch_size: int,
    patience: int = 3,
) -> np.ndarray:
    """
    Train linear softmax: minimize CE + 0.5*l2*||W||^2 via mini-batch GD.
    Returns weight matrix W of shape [D+1, K] (includes bias).
    """
    n, d = x_tr.shape
    k = y_tr.shape[1]
    W = np.zeros((d + 1, k), dtype=np.float32)  # last row is bias

    best_W = W.copy()
    best_val = float("inf")
    bad = 0

    def forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        return Xb @ W

    def loss_and_grad(X: np.ndarray, Y: np.ndarray, W: np.ndarray):
        Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)  # [m, d+1]
        logits = Xb @ W  # [m, k]
        P = _softmax(logits)
        # CE loss
        ce = -np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, None)), axis=1))
        # L2 (include bias for simplicity)
        reg = 0.5 * l2 * np.sum(W * W)
        loss = ce + reg
        # Grad
        G = (Xb.T @ (P - Y)) / Xb.shape[0] + l2 * W
        return loss, G

    for _ in range(max(1, epochs)):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            _, G = loss_and_grad(x_tr[idx], y_tr[idx], W)
            W -= lr * G

        # Early stopping by val loss
        val_logits = forward(x_va, W)
        val_prob = _softmax(val_logits)
        val_ce = -np.mean(np.sum(y_va * np.log(np.clip(val_prob, 1e-12, None)), axis=1))
        val_reg = 0.5 * l2 * np.sum(W * W)
        val_loss = float(val_ce + val_reg)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_W = W.copy()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_W


def _fit_probe_and_eval(
    x_train01: np.ndarray, y_train1h: np.ndarray,
    x_val01: np.ndarray,   y_val1h: np.ndarray,
    x_test01: np.ndarray,  y_test1h: np.ndarray,
    epochs: int, batch_size: int, lr: float, l2: float
) -> Dict[str, Optional[float]]:
    # Flatten to [N, D]
    d = int(np.prod(x_train01.shape[1:]))
    Xtr = x_train01.reshape((x_train01.shape[0], d)).astype("float32")
    Xva = x_val01.reshape((x_val01.shape[0], d)).astype("float32")
    Xte = x_test01.reshape((x_test01.shape[0], d)).astype("float32")

    W = _train_softmax(Xtr, y_train1h, Xva, y_val1h, lr=lr, l2=l2, epochs=epochs, batch_size=batch_size)
    # Evaluate
    Xte_b = np.concatenate([Xte, np.ones((Xte.shape[0], 1), dtype=Xte.dtype)], axis=1)
    Pte = _softmax(Xte_b @ W)
    return _metrics(y_test1h, Pte)


# -----------------------------------------------------------------------------
# Backfill per repo
# -----------------------------------------------------------------------------
def backfill_repo(repo_dir: Path, args) -> None:
    repo = repo_dir.name
    summ_path = _newest_json(repo_dir)
    if summ_path is None:
        print(f"[{repo:<18s}] no summary found – skipping")
        return

    cfg_path = _find_config(repo_dir)
    if cfg_path is None:
        print(f"[{repo:<18s}] no config.yaml – skipping")
        return

    data = json.loads(summ_path.read_text())
    rs_block = _pick(data, "utility_real_plus_synth")
    if rs_block and isinstance(rs_block, dict) and "accuracy" in rs_block:
        print(f"[{repo:<18s}] RS already present – skipping")
        return

    cfg = _load_yaml(cfg_path)

    # Load real + synth
    try:
        x_tr, y_tr, x_va, y_va, x_te, y_te = _load_real_from_cfg(
            repo_dir, cfg, Path(args.data_root).expanduser() if args.data_root else None
        )
    except FileNotFoundError as e:
        print(f"[{repo:<18s}] cannot resolve DATA_DIR\n{e}")
        return

    x_s, y_s = _load_synth(repo_dir, cfg)
    if x_s is None or y_s is None or x_s.size == 0:
        print(f"[{repo:<18s}] no synthetic npy found – cannot backfill")
        return

    # Optional capping for speed
    if args.cap_per_class:
        x_tr, y_tr = _cap_per_class(x_tr, y_tr, args.cap_per_class)
        x_s,  y_s  = _cap_per_class(x_s,  y_s,  args.cap_per_class)

    # Train and evaluate probes
    print(
        f"[{repo:<18s}] training probes: epochs={args.epochs}, cap={args.cap_per_class or '∞'}, "
        f"lr={args.lr}, l2={args.l2}"
    )
    util_R  = _fit_probe_and_eval(x_tr, y_tr, x_va, y_va, x_te, y_te,
                                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, l2=args.l2)
    util_RS = _fit_probe_and_eval(
        np.concatenate([x_tr, x_s], 0), np.concatenate([y_tr, y_s], 0),
        x_va, y_va, x_te, y_te,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, l2=args.l2
    )

    # Update JSON
    data.setdefault("images", {})
    data["images"]["synthetic"] = int(x_s.shape[0])

    if not _pick(data, "utility_real_only"):
        data["utility_real_only"] = util_R
    data["utility_real_plus_synth"] = util_RS

    deltas = {}
    for k in ["accuracy", "macro_f1", "balanced_accuracy", "macro_auprc", "recall_at_1pct_fpr", "ece", "brier"]:
        r = _pick(data, "utility_real_only", k)
        s = _pick(data, "utility_real_plus_synth", k)
        deltas[k] = (None if (r is None or s is None) else float(s - r))
    data["deltas_RS_minus_R"] = deltas

    if args.dry_run:
        print(f"[{repo:<18s}] (dry-run) would write RS + deltas -> {summ_path}")
        return

    summ_path.write_text(json.dumps(data, separators=(",", ":"), indent=2))
    print(f"[{repo:<18s}] wrote RS + deltas -> {summ_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Backfill REAL+SYNTH utility metrics into Phase-1 eval summaries (no TF needed)."
    )
    p.add_argument("--base", default=str(DEFAULT_BASE),
                   help="Folder that contains the 7 repos (default: ~/PycharmProjects)")
    p.add_argument("--repos", nargs="*", default=REPOS,
                   help="Subset of repos to process (default: all 7)")
    p.add_argument("--data-root", default=None,
                   help="Optional absolute folder that contains the dataset; "
                        "joined with cfg.DATA_DIR when it is relative")
    p.add_argument("--epochs", type=int, default=6, help="Probe training epochs")
    p.add_argument("--batch-size", type=int, default=2048, help="Probe mini-batch size")
    p.add_argument("--lr", type=float, default=0.5, help="Learning rate for GD")
    p.add_argument("--l2", type=float, default=1e-3, help="L2 regularization strength")
    p.add_argument("--cap-per-class", type=int, default=None,
                   help="Optional cap per class (train + synth) for speed")
    p.add_argument("--dry-run", action="store_true", help="Do not write JSON; preview only")
    return p.parse_args()


def main():
    set_seed(42)
    args = parse_args()
    base = Path(args.base).expanduser()
    print(f"[backfill] base={base}  repos={', '.join(args.repos)}")
    t0 = time.time()
    for name in args.repos:
        backfill_repo(base / name, args)
    print(f"[done] elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
