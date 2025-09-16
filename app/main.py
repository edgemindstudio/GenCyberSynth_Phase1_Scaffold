import argparse, os, json
from datetime import datetime

try:
    import yaml
except Exception:
    yaml = None

def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        raise SystemExit("PyYAML not installed. Run: pip install pyyaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def cmd_synth(args):
    _ = load_config(args.config)
    print(f"[synth] Using config: {args.config}")
    out = ensure_dir(os.path.join("artifacts", "scaffold", "synthetic"))
    print(f"[synth] (stub) Would write per-class images to: {out}")

def cmd_eval(args):
    _ = load_config(args.config)
    print(f"[eval] Using config: {args.config}")
    if args.no_synth:
        print("[eval] --no-synth: skipping generation; running metrics-only (stub).")
    out = ensure_dir(os.path.join("artifacts", "scaffold", "summaries"))
    path = os.path.join(out, f"scaffold_eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "no_synth": bool(args.no_synth),
            "note": "Stub summary; wire to real evaluator next."
        }, f, indent=2)
    print(f"[eval] Saved evaluation summary to {path}")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p_s = sub.add_parser("synth"); p_s.add_argument("--config", default="configs/config.yaml"); p_s.set_defaults(func=cmd_synth)
    p_e = sub.add_parser("eval");  p_e.add_argument("--config", default="configs/config.yaml"); p_e.add_argument("--no-synth", action="store_true"); p_e.set_defaults(func=cmd_eval)
    args = p.parse_args(); args.func(args)

if __name__ == "__main__":
    main()
