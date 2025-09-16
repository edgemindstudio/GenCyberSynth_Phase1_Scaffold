#!/usr/bin/env bash
set -euo pipefail
git submodule update --init --recursive
# Optional: install core in editable mode
# pip install -e ./gcs-core
[ -f requirements.txt ] && pip install -r requirements.txt || true
echo " Submodules ready. Env initialized."
