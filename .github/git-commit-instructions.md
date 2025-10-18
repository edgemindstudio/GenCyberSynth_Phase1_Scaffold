# Copilot Repo Guidance — GenCyberSynth

## What we're building

- CLI: `python -m app.main {train|synth|eval} --model <name> --config configs/config.yaml`
- Outputs: each eval MUST append one JSON line to `artifacts/<model>/summaries/summary.jsonl` matching `eval_summary.schema.json`.
- Metrics: include MS-SSIM (global & per-class). Prefer vectorized NumPy/TensorFlow.

## Style

- Python 3.11, type hints, flake8/black-friendly.
- Avoid heavy deps; only use packages in `requirements.txt`.
- Small pure functions; no global state. Deterministic RNG from seed in config.

## Tests

- `tests/test_smoke.py`: CPU-only, ≤60s, uses tiny configs. No network calls.

## CI

- Don’t add git+ssh deps. CI uses `requirements.ci.txt`. Keep `make smoke` green.

## File contracts Copilot should follow

- `common/jsonl.py`: `write_jsonl_line(path: str, obj: dict) -> None`
- `schemas/eval_summary.schema.json`: strict JSON schema for eval summary line
- `scripts/aggregate_results.py`: merge many `summary.jsonl` to CSV with stable column order

## Conventions

- Artifacts live under `artifacts/<model>/...`
- Use `PATHS = dataclasses` when helpful, not globals.
