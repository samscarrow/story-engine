# Repository Map (L4 — Max)

Assumptions: Offline static analysis; Python-only codebase; manifests absent; internal imports may rely on sys.path. Secrets redacted as [REDACTED].

**Purpose & Domain**
- Narrative/story generation and simulation engine with orchestrated LLM pipelines, persona agents, POML-driven prompting, and scene-bank assets.

**Tech Stack**
- Language: Python
- Libraries (observed in imports): aiohttp, requests, PyYAML, numpy, jsonschema, oracledb, psycopg2
- Testing: pytest

**Entry Points / CLIs**
- scripts/*.py (e.g., run_meta_pipeline.py, simulate_from_scene_bank.py, evaluate_*); multiple scripts have if __name__ == "__main__".

**Services / Packages**
- core/common — config, env loading, logging, CLI, result store
- core/domain — domain models (characters, scenes, narratives)
- core/orchestration — orchestrators, agent controllers, standardized LLM interfaces
- core/story_engine — narrative graph + pipelines; world state and story arc
- core/character_engine — character simulation engines and group dynamics
- core/cache — response/result caching
- core/storage — DB/storage adapters
- poml — POML components, templates, integration
- scripts — runners for evaluation/simulation/setup; operational utilities
- examples/tests/scene_bank — demos, tests, assets

**Runtime & Infra (inferred)**
- Docker Compose: Postgres service external-kb with volume and healthcheck.
- Databases: Postgres (compose) and optional Oracle (wallet files present); Cloud SQL proxy scripts.
- External: LLM endpoints (LM_ENDPOINT, LMSTUDIO_ENDPOINT, KOBOLD_ENDPOINT).

**Configuration & Env Vars**
- Sources: .env* files, config.yaml, docker-compose, scripts; dotenv loader in core/common.
- Detected vars: CLOUDSQL_ADDRESS, CLOUDSQL_INSTANCE, CLOUDSQL_PORT, CLOUDSQL_PROXY_BIN, DB_DSN, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_SSLCERT, DB_SSLKEY, DB_SSLMODE, DB_SSLROOTCERT, DB_TYPE, DB_USER, DB_WALLET_LOCATION, DB_WALLET_PASSWORD, KOBOLD_ENDPOINT, LM_ENDPOINT, LMSTUDIO_ENDPOINT, LMSTUDIO_MODEL, STORE_ALL, TNS_ADMIN
- Producers/Consumers: core/common/config (produce config), core/storage/database (consume DB_*), scripts/tests (consume various), docker-compose (produce DB_* defaults).

**CI/CD & Release**
- No .github/workflows found; no CI config detected.

**Testing Strategy**
- pytest covering orchestration, pipelines, database, cache, POML adapters, live/integration scenarios; pytest.ini present.

**Module Dependency Graph**
`mermaid
flowchart LR
  scripts[scripts]
  core_character_engine[core/character_engine]
  poml[poml]
  core_orchestration[core/orchestration]
  core_story_engine[core/story_engine]
  core_cache[core/cache]
  core_common[core/common]
  tests[tests]
  core_domain[core/domain]
  core_storage[core/storage]
  examples[examples]
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
   --> 
`

**Service Interaction (best-effort)**
`mermaid
flowchart LR
  Scripts --> Orchestr
  Orchestr --> Story
  Orchestr --> LLM[(External LLMs)]
  Orchestr --> Storage[(DB)]
  Story --> Storage
  Storage --> Postgres[(Compose: external-kb)]
  Scripts --> CloudSQL[(Cloud SQL Proxy)]
`

**Risks & Hotspots**
- Secrets in repo: TLS keys and Oracle wallet artifacts [REDACTED path] — enforce secret scanning and encrypted storage.
- Central coupling: core/story_engine depends on orchestration/common/storage; review for modular boundaries and potential cycles.
- Import fragility: top-level imports (e.g., character_engine.*) imply sys.path hacks — package modules properly.
- External variability: Multiple LLM providers; ensure timeouts, retries, observability.
- Database config: Multiple drivers (psycopg2/oracledb); clarify DB_TYPE, SSL modes, wallet paths; validate on startup.
- TODO density low (good), but LOC high in orchestration/story modules; consider splitting responsibilities.

**How to Regenerate**
- Run: python scripts/repo_map.py --suffix L4
- Outputs: DOCS/REPO_MAP_L4.md, DOCS/MODULES_L4.md, DOCS/repo-map.L4.json, and DOCS/modules/*/SUMMARY.md.

**Assumptions & Questions**
- Packaging: Should this become a proper Python package (pyproject/requirements) to stabilize imports?
- Storage: Is Postgres the default KB with Oracle optional? How is selection handled at runtime?
- LLM SDKs: Which providers are officially supported and how are they configured per env?
