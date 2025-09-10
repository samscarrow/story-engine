# Repository Map (Deep)

Assumptions: Python-centric monorepo without explicit package manifest; imports inferred statically; some internal imports use top-level module names (sys.path manipulations likely). External deps approximated; secrets redacted.

**Business Purpose**
- Narrative/story generation and simulation engine with orchestrated LLM pipelines, persona agents, and POML-driven prompting; supports scene banks and external knowledge base.

**Tech Stack**
- Language: Python
- Libraries (observed): aiohttp, requests, PyYAML, numpy, jsonschema, oracledb, psycopg2
- Test: pytest

**Services/Packages**
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

**Runtime / Infra (inferred)**
- Docker Compose: Postgres external-kb service with healthcheck and volume.
- Databases: Postgres (compose); Oracle wallet artifacts present (potential Oracle DB connectivity); Cloud SQL Proxy scripts.
- External: LLM API endpoints via env (e.g., LM_ENDPOINT, LMSTUDIO_ENDPOINT, KOBOLD_ENDPOINT).

**Configuration & Env Vars**
- Sources: .env*, config.yaml, docker-compose, scripts; dotenv loader in core/common.
- Key vars (detected): CLOUDSQL_ADDRESS, CLOUDSQL_INSTANCE, CLOUDSQL_PORT, CLOUDSQL_PROXY_BIN, DB_DSN, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_SSLCERT, DB_SSLKEY, DB_SSLMODE, DB_SSLROOTCERT, DB_TYPE, DB_USER, DB_WALLET_LOCATION, DB_WALLET_PASSWORD, KOBOLD_ENDPOINT, LM_ENDPOINT, LMSTUDIO_ENDPOINT, LMSTUDIO_MODEL, STORE_ALL, TNS_ADMIN
- Consumers: core/common/config, core/storage/database, scripts/*, tests/* (via os.getenv/environ).

**CI/CD**
- No workflows detected under .github/workflows in this scan.

**Testing Overview**
- pytest suite covers orchestration, pipelines, DB, POML adapters, cache, settings propagation, and live/integration paths.

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

**Service Interaction Overview (best-effort)**
`mermaid
flowchart LR
  Scripts --> Orchestr
  Orchestr --> Story
  Orchestr --> LLM
  Orchestr --> Storage
  Story --> Storage
  Storage --> Postgres
  Scripts --> CloudSQL
`

**Risks & Hotspots**
- Secrets/credentials present: Oracle wallet files and TLS keys in repo (PEM/JKS/P12) — ensure encrypted storage; restrict distribution.
- DB credentials via env; ensure .env not committed in production and CI masks secrets.
- Tight coupling: core/story_engine depends on core/orchestration, storage, and common; high centrality suggests hotspot.
- Direct top-level imports (e.g., character_engine.*) imply path hacks; risk of import fragility in packaging/deployment.
- External service variability (multiple LLM endpoints) — add circuit breakers/retries and timeouts.
- Potential circulars inside core/story_engine (self-import indications) — review for cohesion.

**Assumptions & Questions**
- Packaging: Should modules be a proper package (pyproject/requirements) to avoid path manipulation?
- Storage: Is Postgres the default KB with optional Oracle? Clarify runtime DB selection (DB_TYPE).
- LLM providers: Which SDKs are officially supported; how to configure per environment?
