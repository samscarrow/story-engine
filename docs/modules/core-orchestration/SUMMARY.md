# core/orchestration — Summary

Responsibility: LLM orchestration, agent controllers, standardized interfaces

Boundaries
- Depends on: poml, core/common
- Used by: scripts, core/character_engine, tests, examples, core/story_engine

Public Surface (inferred)
- Symbols: SimulationContext, TemplateMetadata, TemplateRepository, AutonomousPersonaAgent, CharacterSimulatorAgent, SceneArchitectAgent, PersonaAgentFactory, ModelProvider, GenerationError, ProviderFailure, LLMConfig, LLMResponse, LLMProvider, LMStudioProvider, KoboldCppProvider
- CLIs: scripts/* map to orchestration/story engines (see scripts module)

Data Models / Schemas
- core/domain provides primary models; others consume them.

Notable Files
- llm_orchestrator.py, autonomous_persona_agents.py, recursive_simulation_engine.py

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database.py (psycopg2/oracledb) in storage, cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- 

Complexity Signals
- Files: 6; LOC: 2965; TODO/FIXME: 0

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM provider implementations behind clear interfaces
- Harden storage config handling and secrets via vault/CI masks
- Add type hints and docstrings in high-centrality modules
