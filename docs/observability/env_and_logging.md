# Environment and Logging Quickstart

## Local Environment
- Create `.env.oracle` in repo root (never commit real secrets):
```
DB_USER=story_db
DB_PASSWORD=...
DB_DSN=mainbase_high
DB_WALLET_LOCATION=./oracle_wallet
# Optional: ORACLE_* mirrors and TNS_ADMIN if preferred
```

### With direnv (recommended)
- Install direnv, hook your shell, run `direnv allow` in repo root.
- `.envrc` will `set -a; source .env.oracle; set +a`, exporting variables.
- Verify: `env | rg 'DB_|ORACLE_|TNS_ADMIN'`.

### Without direnv
- Scripts use `python-dotenv` to load `.env.oracle` directly, so they still work.

## Logging
- Default JSON logs; configure via env:
  - `LOG_FORMAT=json|text` (default json)
  - `LOG_LEVEL=INFO|DEBUG|...` (default INFO)
  - `LOG_DEST=stdout|stderr|file` (default stdout)
  - `LOG_FILE_PATH=story_engine.log` if `LOG_DEST=file`
  - `LOG_SAMPLING_RATE=0.0..1.0`
- Standard fields: `level,name,message,time,service,trace_id,correlation_id`.
- Extras included automatically: `elapsed_ms,attempt,ok,event,error_code,...` plus any `extra={}` keys passed to logger.
  - Sensitive keys are redacted: `api_key, authorization, password, db_password, oracle_password`.

## Healthcheck
- Run locally: `python scripts/oracle_healthcheck.py --pool`
- In CI: configure secrets `DB_DSN, DB_USER, DB_PASSWORD, DB_WALLET_LOCATION`.
- CI uploads `oracle_health.ndjson` and posts a summary with connect success rate and P50/P95 latency.

