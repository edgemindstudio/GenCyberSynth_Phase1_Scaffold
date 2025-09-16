
from __future__ import annotations
import argparse
from pathlib import Path
from gcs_core.eval_common import evaluate_model_suite

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    def add_common(sp):
        sp.add_argument("--config", type=str, default="configs/default.yaml")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--simulate", action="store_true", help="Run in simulate mode (Phase 1)")
        sp.add_argument("--model-name", type=str, default="REPLACE_ME")
    sp_train = sub.add_parser("train"); add_common(sp_train)
    sp_synth = sub.add_parser("synth"); add_common(sp_synth)
    sp_synth.add_argument("--per-class", type=int, default=1000)
    sp_synth.add_argument("--out", type=str, default="artifacts/synthetic")
    sp_eval = sub.add_parser("eval"); add_common(sp_eval)
    sp_eval.add_argument("--fid-cap", type=int, default=200)
    sp_eval.add_argument("--json", type=str, default="runs/summary.jsonl")
    sp_eval.add_argument("--console", type=str, default="runs/console.txt")
    return p.parse_args()

def main():
    args = parse_args()
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    if args.cmd == "train":
        if args.simulate:
            print(f"[SIM] Training {args.model_name}... seed={args.seed}")
        else:
            raise NotImplementedError("Implement real training here.")
    elif args.cmd == "synth":
        Path(args.out).mkdir(parents=True, exist_ok=True)
        if args.simulate:
            print(f"[SIM] Generating {args.per_class}/class into {args.out}")
        else:
            raise NotImplementedError("Implement real synthesis here.")
    elif args.cmd == "eval":
        real_dirs = {"train": "/path/to/train", "val": "/path/to/val", "test": "/path/to/test"}
        evaluate_model_suite(
            model_name=args.model_name,
            seed=args.seed,
            real_dirs=real_dirs,
            synth_dir="artifacts/synthetic",
            fid_cap_per_class=args.fid_cap,
            evaluator="small_cnn_v1",
            output_json=args.json,
            output_console=args.console,
            metrics=None,
            notes="phase1-simulate"
        )
        print(f"Wrote {args.console} and {args.json}")
    else:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
