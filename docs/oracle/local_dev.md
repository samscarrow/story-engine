# Local Oracle XE for Development

This guide describes a lightweight, self-contained Oracle XE setup for local development and repeatable smoke tests.

## Prerequisites
- Docker Desktop or Docker Engine 20.10+
- No Oracle wallet required for XE

## Start Oracle XE
```bash
# From repo root
docker compose -f docker-compose.oracle.yml up -d

# Check container status
docker ps --filter name=story_engine_oracle_xe
```

The setup creates an application user by default using env defaults:
- DB user: `STORY_DB`
- DB password: `story_pwd`
- PDB/service: `XEPDB1`

## Environment Variables
Add the following to `.env.oracle` (or export in your shell):
```
DB_TYPE=oracle
DB_USER=STORY_DB
DB_PASSWORD=story_pwd
DB_DSN=localhost/XEPDB1

# Optional connection stability
ORACLE_USE_POOL=1
ORACLE_POOL_MIN=1
ORACLE_POOL_MAX=4
ORACLE_POOL_INC=1
ORACLE_POOL_TIMEOUT=60
ORACLE_RETRY_ATTEMPTS=3
ORACLE_RETRY_BACKOFF=1.0
```

Wallet variables (e.g., `DB_WALLET_LOCATION`) are NOT required for XE.

## Verify Connectivity
Option A: Python diagnostic script
```bash
python diagnose_oracle_connection.py
```

Option B: Pytest smoke test
```bash
pytest -q tests/oracle/test_oracle_xe_integration.py
# Or run all Oracle-marked tests
pytest -q -m oracle
```

Both expect `DB_USER`, `DB_PASSWORD`, `DB_DSN` to be set as above.

## Using From Story Engine
The codebase supports Oracle via `OracleConnection` in `src/story_engine/core/core/storage/database.py`.
Set `DB_TYPE=oracle` and ensure the above env vars are present. Pooling and retry parameters are controlled via the `ORACLE_*` envs listed above.

For centralizing configuration, a minimal settings helper is provided in `core.common.settings` (optional to use initially).

## Stop and Clean Up
```bash
docker compose -f docker-compose.oracle.yml down

# Remove persistent data (DANGER: drops all XE data)
docker volume rm story-engine_oracle_xe_data || true
```

## Troubleshooting
- Port 1521 already in use: stop other Oracle containers/services or change the mapping in the compose file.
- Authentication errors: confirm `APP_USER`/`APP_USER_PASSWORD` in compose match `DB_USER`/`DB_PASSWORD` in your environment.
- DSN issues: prefer `localhost/XEPDB1` or `127.0.0.1/XEPDB1`.
