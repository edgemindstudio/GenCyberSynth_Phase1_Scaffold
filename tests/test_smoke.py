# tests/test_smoke.py
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

MODEL      = os.getenv("SMOKE_MODEL", "gan")
CFG        = os.getenv("SMOKE_CFG", "configs/config.yaml")
REAL_ROOT  = os.getenv("SMOKE_REAL_ROOT", "USTC-TFC2016_malware/real")
SEED       = int(os.getenv("SMOKE_SEED", "0"))
SYNTH_BASE = os.getenv("SMOKE_SYNTH_BASE", "artifacts")

def run(cmd: list[str]):
    print("==>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def test_synth_and_eval_smoke():
    # isolate this test’s outputs
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    synth_root = f"{SYNTH_BASE}/{MODEL}/synthetic_smoke_seed{SEED}_{stamp}"
    Path(synth_root).mkdir(parents=True, exist_ok=True)

    # tiny synth
    run([
        "python","-m","app.main","synth",
        "--model", MODEL, "--config", CFG,
        "--seed", str(SEED),
        "--synth-per-class","8",
        "--real-root", REAL_ROOT,
        "--synth-root", synth_root
    ])

    manifest = Path(synth_root) / "manifest.json"
    assert manifest.exists(), "Manifest not written"

    # eval writes summary_*.json under artifacts/<model>/summaries/
    run([
        "python","-m","app.main","eval",
        "--model", MODEL, "--config", CFG,
        "--seed", str(SEED),
        "--real-root", REAL_ROOT,
        "--synth-root", synth_root
    ])

    summaries_dir = Path(f"{SYNTH_BASE}/{MODEL}/summaries")
    summaries = sorted(summaries_dir.glob("summary_*.json"))
    assert summaries, f"No evaluation summaries found in {summaries_dir}"

    # JSON is parseable; spot-check model field (some adapters may fill 'unknown')
    with summaries[-1].open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert data.get("model") in (MODEL, "unknown")
