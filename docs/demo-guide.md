# Story Engine Demo Guide

Run a cohesive demo that produces evaluatable artifacts.

## Quick Start

```
python -m story_engine.scripts.run_demo --runs 3 --emphasis doubt
```

Artifacts in `dist/run-<timestamp>/`:
- `story.json` – character runs, scene plan, narrative graph (if available)
- `continuity_report.json` – issues and suggested fixes
- `metrics.json` – minimal quality signals (schema, continuity, repetition)
- `console.md` – readable sample outputs
- `config.snapshot.yaml`, `env.capture` – reproducibility

## Options

- `--live` – use unified orchestrator if configured in `config.yaml`/env
- `--use-poml` – enable POML adapter prompting
- `--strict-persona` – enable persona guardrails with threshold
- `--persona-threshold <int>` – override threshold (default from config)
- `--profile <id>` – character preset (`pilate` default)
- `--situation <text>` – override default situation
- `--runs <n>` – number of simulations

## Reproducibility

The demo saves a snapshot of relevant env vars and the resolved config into the output directory. For more deterministic behavior, set seeds at the process level if you add modules that use global randomness.

## Live LLMs

Set provider credentials and model preferences in `config.yaml` or env. With `--live`, the script attempts to initialize the `UnifiedLLMOrchestrator`; it falls back to a mock if unavailable.

