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

## Cluster Execution (LM Studio Fleet)

Use the `cluster` environment to fan requests across multiple LM Studio hosts
through the `ai-lb/` load balancer. The default configuration assumes:

- `ai-lb` running on `pc-sam` (`http://localhost:8000`)
- LM Studio nodes at `localhost:1234`, `sams-macbook-pro:1234`, and
  `macbook:1234`
- a PostgreSQL database whose connection details are supplied at runtime via
  `CLUSTER_DB_HOST`, `CLUSTER_DB_NAME`, `CLUSTER_DB_USER`, and
  `CLUSTER_DB_PASSWORD`

### 1. Launch the load balancer

```bash
pushd ai-lb
docker compose --profile full up --build
popd
```

Override node endpoints by exporting `LM_NODE_LOCAL`, `LM_NODE_SAMS_MAC`, or
`LM_NODE_MACBOOK` before invoking the CLI.

### 2. Validate fleet health

```bash
storyctl check --env cluster --fail-fast
```

The cluster definition probes the load balancer as well as each individual
node and optionally runs `scripts/aggregate_healthcheck.py`.

### 3. Run and persist stories

Use the helper script to execute full cluster runs and collect artifacts:

```bash
scripts/run_cluster_story.sh --prompt "Write an intense senate debate" --runs 5 --parallel 3
```

The script performs the following steps:

1. `storyctl check --env cluster` to guard the run
2. `storyctl run --env cluster --workflow <timestamp>`
3. `storyctl env export --env cluster --format json` to snapshot effective config
4. `story db-export --workflow <id>` saving results to
   `artifacts/cluster/<workflow>/outputs.ndjson`

Artifacts include an `env.json` capture and a `run.info` metadata file so you
can replay or tune future iterations.

### 4. Sync configuration with lmstudio-omnibus

To keep the MCP omnibus server aligned with the cluster definition, generate
its `.env` file from the same storyctl export:

```bash
scripts/sync_lmstudio_omnibus_env.py
```

By default the script writes to `../src/lmstudio-omnibus/.env`, populating
`AILB_URL`, `LMSTUDIO_API_URL`, and all `LM_NODE_*` entries. Rerun it whenever
the cluster environment changes (for example, if the ai-lb host or node ports
move) so both toolchains share the same routing configuration.
