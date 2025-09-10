# core/story_engine — Summary

Responsibility: narrative graph, pipelines, world state, story arcs

Boundaries
- Depends on: core/common, poml, core/domain, core/cache, core/orchestration, core/storage
- Used by: scripts, core/character_engine, tests

Public Surface (inferred)
- Symbols: FeedbackType, RevisionStrategy, StoryMetrics, Feedback, StoryVersion, BranchPoint, IterativeStoryEngine, AdaptiveStorySystem, GraphNode, GraphEdge, NarrativeGraph, NarrativePipeline, _strip_md, _slugify, SceneEntry
- CLIs: scripts/* map to orchestration/story engines (see scripts module)

Data Models / Schemas
- core/domain provides primary models; others consume them.

Notable Files
- iterative_story_system.py, story_arc_engine.py, story_engine_orchestrated.py

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database.py (psycopg2/oracledb) in storage, cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- DB_DSN, DB_WALLET_LOCATION, DB_PASSWORD, DB_TYPE, DB_SSLKEY, DB_NAME, DB_WALLET_PASSWORD, DB_SSLCERT, DB_HOST, DB_PORT, DB_USER, DB_SSLMODE, DB_SSLROOTCERT

Complexity Signals
- Files: 10; LOC: 3902; TODO/FIXME: 0

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM provider implementations behind clear interfaces
- Harden storage config handling and secrets via vault/CI masks
- Add type hints and docstrings in high-centrality modules
