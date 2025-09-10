# poml — Summary

Responsibility: POML components/templates and integration

Boundaries
- Depends on: 
- Used by: scripts, core/orchestration, core/character_engine, tests, core/story_engine

Public Surface (inferred)
- Symbols: POMLCharacterSimulationEngine, create_poml_engine, SimulationEngineMigrator, PromptTemplate, POMLOrchestrator, OrchestratorMigrationHelper, POMLConfig, POMLCache, POMLEngine, create_engine, render_template, StoryEnginePOMLAdapter
- CLIs: scripts/* map to orchestration/story engines (see scripts module)

Data Models / Schemas
- core/domain provides primary models; others consume them.

Notable Files
- poml_integration.py, llm_orchestrator_poml.py, character_simulation_poml.py

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database.py (psycopg2/oracledb) in storage, cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- 

Complexity Signals
- Files: 3; LOC: 2500; TODO/FIXME: 0

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM provider implementations behind clear interfaces
- Harden storage config handling and secrets via vault/CI masks
- Add type hints and docstrings in high-centrality modules
