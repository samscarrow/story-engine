# examples — Summary

Responsibility: usage examples/demonstrations

Boundaries
- Depends on: core/orchestration
- Used by: tests

Public Surface (inferred)
- Symbols: StoryProjectManager, print_usage_instructions, AutonomousPersonaDemonstration, MigratedCharacterSimulationEngine, MigratedNarrativePipeline, LegacyEngineAdapter, print_migration_benefits
- CLIs: scripts/* map to orchestration/story engines (see scripts module)

Data Models / Schemas
- core/domain provides primary models; others consume them.

Notable Files
- autonomous_agent_usage_example.py, autonomous_persona_demonstration.py, standardized_llm_migration_example.py

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database.py (psycopg2/oracledb) in storage, cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- 

Complexity Signals
- Files: 3; LOC: 1980; TODO/FIXME: 0

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM provider implementations behind clear interfaces
- Harden storage config handling and secrets via vault/CI masks
- Add type hints and docstrings in high-centrality modules
