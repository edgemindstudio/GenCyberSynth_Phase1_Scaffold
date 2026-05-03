#!/usr/bin/env python3
"""
Paper 4 policy manifest writer.

Initial version:
- Reads a baseline synthetic manifest.
- Writes an identity-policy manifest to a policy-specific path.
- Does not overwrite the baseline manifest.
- Adds lightweight Paper 4 policy provenance.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--policy_id", default="keep_all")
    ap.add_argument("--policy_type", default="identity")
    args = ap.parse_args()

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)

    if not in_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {in_path}")

    if out_path.resolve() == in_path.resolve():
        raise ValueError("Refusing to overwrite the source manifest.")

    with open(in_path, "r") as f:
        manifest = json.load(f)

    manifest["paper4_policy"] = {
        "paper_id": "paper4",
        "policy_id": args.policy_id,
        "policy_type": args.policy_type,
        "source_manifest": str(in_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Identity policy placeholder: keeps all synthetic samples from the source manifest."
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("[ok] wrote policy manifest")
    print(" input :", in_path)
    print(" output:", out_path)
    print(" policy:", args.policy_id)


if __name__ == "__main__":
    main()