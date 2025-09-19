# storyctl — Environment-Oriented CLI

`storyctl` wraps the Story Engine tooling in a single command surface optimised for
staging and production operations. It centralises environment selection,
configuration loading, and safety checks so teams can promote workloads with
confidence.

## Installation

Install the package in editable mode to expose the CLI entry point:

```bash
uv pip install -e .
```

This makes the `storyctl` command available.

## Environment Selection

Environment definitions live under `story_engine/tooling/environments/` and can
be extended via `--env-dir` or the `STORYCTL_ENV_DIR` environment variable.
Each YAML file provides:

- metadata: name, description, optional tags
- `env`: declarative environment variable specs (defaults, `from_env`, and
  sensitivity)
- `checks`: preflight validation steps (`db`, `http`, or `command`)

`local`, `staging`, and `production` environments are bundled by default. `local`
serves as the default context when no override is provided.

## Key Commands

- `storyctl env list` — enumerate available environments and the active default
- `storyctl env show [--env ENV]` — inspect resolved environment variables;
  add `--reveal` to show sensitive values (when present)
- `storyctl env export --format shell|json` — emit exports for scripts or CI
- `storyctl check [--env ENV] [--check NAME]` — run preflight checks across
  database, LLM endpoints, and optional smoke tests
- `storyctl run <command…>` — execute a command with environment variables
  applied (use `--dry-run` to print exports without executing)

## Custom Environments

To introduce a bespoke environment, create a YAML file under a separate
directory and load it with:

```bash
storyctl --env-dir deploy/environments --env staging check
```

Definitions support inheritance through the `extends` field. Secrets should be
referenced via `from_env` to pull values from the host environment or secret
manager at runtime.

## Production Safety

The bundled staging and production definitions include:

- required database credentials retrieved from `STAGING_DB_PASSWORD` and
  `PRODUCTION_DB_PASSWORD`
- LLM endpoint health probes (`/health`)
- optional smoke tests for `scripts/db_smoke_test.py` and
  `scripts/aggregate_healthcheck.py`

`storyctl check` exits non-zero when a required value is missing or a mandatory
check fails, making it suitable for CI gating or manual promotion workflows.
