# core/common — Summary

Responsibility: config, env loading, logging, CLI, result store

Boundaries
- Depends on: core/storage
- Used by: scripts, core/orchestration, core/character_engine, tests, core/story_engine

Public Surface (inferred)
- Symbols: add_model_client_args, get_model_and_client_config, detect_loaded_model, detect_lmstudio_model, detect_koboldcpp_model, validate_model_connection, validate_lmstudio_connection, validate_koboldcpp_connection, print_connection_status, setup_standard_args, _deep_merge, load_config, load_dotenv_keys, configure_logging, store_workflow_output
- CLIs: scripts/* map to orchestration/story engines (see scripts module)

Data Models / Schemas
- core/domain provides primary models; others consume them.

Notable Files
- cli_utils.py, config.py, result_store.py

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database.py (psycopg2/oracledb) in storage, cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- LMSTUDIO_ENDPOINT, DB_PASSWORD, DB_SSLKEY, LM_ENDPOINT, DB_NAME, KOBOLD_ENDPOINT, LMSTUDIO_MODEL, DB_SSLCERT, DB_HOST, DB_PORT, DB_USER, DB_SSLMODE, DB_SSLROOTCERT

Complexity Signals
- Files: 5; LOC: 484; TODO/FIXME: 0

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM provider implementations behind clear interfaces
- Harden storage config handling and secrets via vault/CI masks
- Add type hints and docstrings in high-centrality modules
