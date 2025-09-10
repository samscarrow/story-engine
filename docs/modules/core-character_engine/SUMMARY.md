# core/character_engine — Summary

Responsibility: character simulation engines and group dynamics

Boundaries
- Depends on: core/story_engine, poml, core/common, core/orchestration
- Used by: scripts

Public Surface (inferred)
- Symbols: LLMResponse, LLMInterface, MockLLM, OpenAILLM, LMStudioLLM, SimulationError, LLMError, RetryHandler, EmotionalState, CharacterMemory, CharacterState, SimulationEngine, Faction, ActionType, GroupCharacter
- CLIs: scripts/* map to orchestration/story engines (see scripts module)

Data Models / Schemas
- core/domain provides primary models; others consume them.

Notable Files
- character_simulation_engine_v2.py, complex_group_dynamics.py, multi_character_simulation.py

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database.py (psycopg2/oracledb) in storage, cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- 

Complexity Signals
- Files: 4; LOC: 2266; TODO/FIXME: 0

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM provider implementations behind clear interfaces
- Harden storage config handling and secrets via vault/CI masks
- Add type hints and docstrings in high-centrality modules
