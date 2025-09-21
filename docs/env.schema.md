# Environment Schema

This document enumerates environment variables used across the story‑engine repository, grouped by subsystem with defaults and notes. Values are case‑sensitive; booleans accept 1|true|yes|on.

## Core Logging & Observability
- `LOG_FORMAT` (json|text, default: json) — output format.
- `LOG_LEVEL` (DEBUG|INFO|WARNING|ERROR, default: INFO) — log level.
- `LOG_DEST` (stdout|stderr|file, default: stdout) — destination.
- `LOG_FILE_PATH` (path, default: story_engine.log) — when `LOG_DEST=file`.
- `LOG_SERVICE_NAME` (string, default: story-engine) — service tag.
- `TRACE_ID` (string, optional) — preset correlation/trace id.
- `LOG_SAMPLING_RATE` (0.0–1.0, default: 1.0) — sample info logs.

## Database (Selector)
- `DB_TYPE` (oracle|postgresql|sqlite, default: postgresql) — active DB adapter.

### PostgreSQL
- `DB_HOST` (default: localhost)
- `DB_PORT` (default: 5432)
- `DB_NAME` (default: story_db)
- `DB_USER` (default: story)
- `DB_PASSWORD` (required for write access)
- `DB_SSLMODE` (disable|require|verify-ca|verify-full, optional)
- `DB_SSLROOTCERT` (path, optional)
- `DB_SSLCERT` (path, optional)
- `DB_SSLKEY` (path, optional)

### Oracle (Autonomous DB or XE)
- `DB_USER` (required)
- `DB_PASSWORD` (required)
- `DB_DSN` (service name, TNS alias, or full connect string; default: localhost/XEPDB1)
- `DB_CONNECT_STRING` (alias for `DB_DSN`)
- `DB_WALLET_LOCATION` (dir path to wallet; or set `TNS_ADMIN`)
- `DB_WALLET_PASSWORD` (optional; if wallet encrypted)
- `TNS_ADMIN` (dir path to wallet; alternative to `DB_WALLET_LOCATION`)
- Pooling & stability knobs (optional; sensible defaults):
  - `ORACLE_USE_POOL` (bool, default: true)
  - `ORACLE_POOL_MIN` (int, default: 1)
  - `ORACLE_POOL_MAX` (int, default: 4)
  - `ORACLE_POOL_INC` (int, default: 1)
  - `ORACLE_POOL_TIMEOUT` (seconds, default: 60)
  - `ORACLE_WAIT_TIMEOUT` (seconds, optional) — wait for pool acquire.
  - `ORACLE_RETRY_ATTEMPTS` (int, default: 3) — transient connect retries.
  - `ORACLE_RETRY_BACKOFF` (seconds, default: 1.0) — exponential backoff base.
  - `ORACLE_PING_ON_CONNECT` (bool, default: true) — validate connection with `SELECT 1 FROM DUAL`.

### SQLite (local fallback)
- `SQLITE_DB` (filename, default: workflow_outputs.db)

### Result Store Feature Flag
- `STORE_ALL` (bool, default: false) — opportunistically store outputs when DB creds present.

## LLM Orchestrator & Clients
- Routing / defaults:
  - `LM_ENDPOINT` (URL, default: http://localhost:1234 for LM Studio; http://localhost:5001 for KoboldCpp in some CLIs)
  - `LM_MODEL` (string, special value `auto` to delegate to AI LB)
  - `LMSTUDIO_MODEL` (string, optional; if set, used as model hint)
  - `LLM_TIMEOUT_SECS` (int, default: 60) — request timeout budget.
  - `LM_PREFER_SMALL` (bool, default: false) — prefer smaller models when auto.
  - `LM_CLIENT_MODEL_AUTOPICK` (bool, default: false) — client sets `LM_MODEL` once per run.
- Resilience / retries:
  - `LM_RETRY_ATTEMPTS` (int, default: 1) — LMStudio path.
  - `LM_RETRY_BASE_DELAY` (seconds, default: 0.2) — LMStudio path.
  - `KOBOLD_RETRY_ATTEMPTS` (int, default: 1)
  - `KOBOLD_RETRY_BASE_DELAY` (seconds, default: 0.2)
  - `LM_REQUEST_BUDGET_MS` (int, optional) — total budget across retries.
- Circuit breaker (orchestrator-level):
  - `LLM_CB_THRESHOLD` (int, default: 3) — failures to open.
  - `LLM_CB_WINDOW_SEC` (seconds, default: 15) — open duration.
- Provider-specific endpoints (when not using AI LB):
  - `KOBOLD_ENDPOINT` (URL, default: http://localhost:5001)

## AI Load Balancer (external component)
Used when `LM_ENDPOINT` points at the LB. Common LB envs (see `ai-lb/`):
- `REQUEST_TIMEOUT_SECS` (default: 60)
- `RETRY_AFTER_SECS` (default: 2) — returned on 429.

## Services & Workers
Entry points initialize logging via `init_logging_from_env()` and rely on the variables above. See `src/services/*/__main__.py`.

## Docker / Compose
- Root `docker-compose.yml` exposes Postgres with `DB_*` variables.
- `docker-compose.oracle.yml` runs Oracle XE; relies on `.env.oracle`.

## Examples
See `.env.example` for a safe starting point and `.env.oracle-example` for Oracle wallet setups.

