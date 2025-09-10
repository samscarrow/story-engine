# scripts — Summary

Responsibility: CLI runners for eval/simulations/setup

Boundaries
- Depends on: core/story_engine, core/orchestration, poml, core/storage, core/common, core/character_engine, core/domain
- Used by: 

Public Surface (inferred)
- Symbols: main, _read_inputs, load_character, handle_model_change, update_last_model, load_persona_yaml, build_db, extract_wallet, create_env_config, find_proxy_binary, build_command
- CLIs: scripts/* map to orchestration/story engines (see scripts module)

Data Models / Schemas
- core/domain provides primary models; others consume them.

Notable Files
- run_meta_pipeline.py, evaluate_poml_storytelling.py, evaluate_poml.py

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database.py (psycopg2/oracledb) in storage, cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- CLOUDSQL_PROXY_BIN, CLOUDSQL_ADDRESS, DB_PASSWORD, STORE_ALL, DB_SSLKEY, LM_ENDPOINT, DB_NAME, CLOUDSQL_PORT, LMSTUDIO_MODEL, DB_SSLCERT, DB_HOST, DB_PORT, DB_USER, DB_SSLMODE, CLOUDSQL_INSTANCE, DB_SSLROOTCERT

Complexity Signals
- Files: 14; LOC: 1582; TODO/FIXME: 0

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM provider implementations behind clear interfaces
- Harden storage config handling and secrets via vault/CI masks
- Add type hints and docstrings in high-centrality modules
