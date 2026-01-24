# tools/build_paper1_jsonl.py

import json, os
from pathlib import Path

NAME = os.getenv("PHASE1_SUMMARY_NAME", "paper1.json")
OUT_JSONL = Path(os.getenv("OUT_JSONL", "artifacts/summaries/phase1_summaries.jsonl"))

OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

rows = []
for p in Path("artifacts").glob(f"*/summaries/{NAME}"):
    try:
        d = json.loads(p.read_text())
        d.setdefault("source_path", str(p))
        rows.append(d)
    except Exception:
        pass

rows.sort(key=lambda x: (x.get("model", ""), x.get("seed", 0)))

with OUT_JSONL.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"wrote {OUT_JSONL} rows={len(rows)} from {NAME}")
