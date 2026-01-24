# tools/phase1_gate.sh

#!/usr/bin/env bash
set -euo pipefail

python tools/build_phase1_scores.py
python tools/check_phase1_integrity.py

echo "[phase1_gate] OK"
