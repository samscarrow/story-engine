# Story Engine CLI

A lightweight command-line interface to run simulations/generations and persist results to the configured database.

## Installation
The CLI is installed with the package. In editable mode:

```
uv pip install -e .
```

This exposes the `story` command.

## Quickstart
- Ensure an LM endpoint is reachable (defaults to LM Studio at `http://localhost:1234`).
- DB defaults to SQLite unless Postgres/Oracle is configured via env.

Run a prompt and persist to DB:
```
story run --prompt "Write one vivid sentence about sunrise in Jerusalem." --runs 3 --parallel 2
```

Show effective config:
```
story config-show
```

Check DB health:
```
story db-health
```

Export results for a workflow:
```
story db-export --workflow cli_run --output outputs.ndjson
```

## Options
- `--provider`: currently supports `lmstudio` (default)
- `--endpoint`: defaults to `LM_ENDPOINT` or `http://localhost:1234`
- `--model`: model hint for the provider (defaults to `LM_MODEL` or `auto`)
- `--input`: file with prompts (txt, json, jsonl)
- `--runs`: repeat each input N times
- `--parallel`: concurrent requests
- `--tag key=value`: attach arbitrary tags to stored payloads (repeatable)
- `--dry-run`: skip DB persistence
- `--require-healthy`: fail if LM or DB is unreachable

## DB Configuration
- Uses `get_db_settings()` which falls back to SQLite when Oracle/Postgres are not available.
- To require Oracle explicitly, set `DB_REQUIRE_ORACLE=1`.
- See `docs/oracle/test_gating.md` for Oracle-specific notes.

## Environment-Aware Workflows
For staging and production orchestration, prefer the `storyctl` CLI which wraps
environment selection, preflight checks, and command execution. See
`docs/storyctl.md` for details.
