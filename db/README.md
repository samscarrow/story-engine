Centralized Story Engine database schema (Oracle-first)

This directory contains an Oracle-compatible schema to centralize providers, models, prompts, runs, scenes, evaluations, and world state. It is designed to run locally first and then migrate to ADB with minimal changes.

Contents
- schema.sql — Core DDL (tables, indexes, constraints)
- grants.sql — Minimal roles and privileges (optional, for non-SYS users)
- sync_providers_models.py — Populates providers/models from orchestrator `/v1/models` responses

Connection
- Uses environment variables compatible with `oracledb`:
  - `ORACLE_DSN` (e.g., `localhost/XEPDB1` or Easy Connect Plus for ADB:
    `tcps://adb.<region>.oraclecloud.com:1522/<full_service_name>?wallet_location=/path/to/wallet&ssl_server_dn_match=true`)
  - `ORACLE_USER`
  - `ORACLE_PASSWORD`

Quick start
1) Create schema
   sqlplus ${ORACLE_USER}/${ORACLE_PASSWORD}@${ORACLE_DSN} @db/schema.sql

2) Seed providers/models from your running orchestrator config
   uv run python db/sync_providers_models.py --config config.yaml

Notes
- JSON columns are CLOB with `IS JSON` for portability. In ADB, you can switch to native JSON.
- Switch to Liquibase/Flyway for versioned migrations when moving to ADB.
- The sync script is idempotent for providers/models based on unique constraints.
