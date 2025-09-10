# tests — Summary

Responsibility: pytest suites and integration tests

Boundaries
- Depends on: core/orchestration, examples, core/storage, core/story_engine, core/domain, poml, core/common
- Used by: 

Public Surface (inferred)
- Symbols: find_mock_data_issues, get_line_context, analyze_specific_mock_implementations, main, BreakingPointTester, MockLLMOrchestrator, SimplifiedBreakingPointTester, TestDatabase, PilateStubResponse, PilateStubOrchestrator, test_full_pilate_flow_with_poml, _parse_eval, test_live_poml_pilate_flow_minimal, StubResp, CountingOrchestrator
- CLIs: scripts/* map to orchestration/story engines (see scripts module)

Data Models / Schemas
- core/domain provides primary models; others consume them.

Notable Files
- test_autonomous_agent_breaking_points.py, test_output_authenticity_audit.py, test_breaking_points_simplified.py

Side Effects
- Network: aiohttp/requests in orchestration/story modules
- Storage: database.py (psycopg2/oracledb) in storage, cache writes
- Filesystem: scene_bank JSON, config.yaml, POML templates

Feature Flags / Env Vars
- 

Complexity Signals
- Files: 19; LOC: 3932; TODO/FIXME: 0

Suggested Refactors
- Stabilize imports via packaging; avoid sys.path hacks
- Encapsulate LLM provider implementations behind clear interfaces
- Harden storage config handling and secrets via vault/CI masks
- Add type hints and docstrings in high-centrality modules
