# core/storage — Summary

Responsibility: database/storage adapters

Boundaries
- Depends on: 
- Used by: scripts, core/common, tests, core/story_engine

Public Surface (inferred)
- Symbols: DatabaseConnection, SQLiteConnection, PostgreSQLConnection, OracleConnection, get_database_connection
- CLIs: scripts/* map to orchestration/story engines (see scripts module)

Data Models / Schemas
- core/domain provides primary models; others consume them.

Notable Files
- database.py, __init__.py

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database.py (psycopg2/oracledb) in storage, cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- TNS_ADMIN

Complexity Signals
- Files: 2; LOC: 348; TODO/FIXME: 0

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM provider implementations behind clear interfaces
- Harden storage config handling and secrets via vault/CI masks
- Add type hints and docstrings in high-centrality modules
