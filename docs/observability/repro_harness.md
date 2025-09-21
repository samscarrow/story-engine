# Repro Harness

Re-run a demo scenario from saved artifacts to reproduce issues and compare results.

## Usage
- Generate artifacts:
  - `python src/story_engine/scripts/run_demo.py [--flags]`
  - Outputs under `dist/run-YYYYMMDD-HHMMSS/` including:
    - `config.snapshot.yaml` (or JSON fallback)
    - `args.snapshot.json` (CLI args)
    - `trace.ndjson`, `metrics.json`, `story.json`

- Re-run with the same settings:
  - `python scripts/repro_harness.py --from-dir dist/run-YYYYMMDD-HHMMSS`
  - Optional: `--dry-run` to print the command only
  - Optional: `--trace-id TRACE123` to tag logs for correlation

The harness invokes `run_demo.py` with the same CLI flags and passes the snapshot config as `--config`.

> Note: This reproduces the flow deterministically but not provider responses. For full determinism, use mock providers or freeze seeds where supported.

## Compare Runs (regression checks)
- Compare a new run directory against a baseline:
  - `python scripts/repro_harness.py --from-dir NEW_RUN --compare-to OLD_RUN --compare-only`
  - Add `--fail-on-regression` to exit with code 3 if regressions are detected.
  - Add `--report-json path/to/report.json` to emit a machine-readable summary.
- Checks:
  - `schema_valid` and `continuity_ok` should not flip from true to false
  - `degraded_runs` and `continuity_violations` should not increase
  - `continuity_report.violations` count should not increase

## Deterministic Seeds
- Set one of these env vars before running the demo to fix global seeds:
  - `DEMO_SEED=1234` (preferred), or `STORY_ENGINE_SEED`, or `SEED`
- Effects:
  - Sets `PYTHONHASHSEED`, Python `random`, and NumPy (if installed)
