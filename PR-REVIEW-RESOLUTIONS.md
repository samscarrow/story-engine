PR #3 Review Resolutions

Status: working branch feature/storyctl-cluster-runner (head at this commit)

Summary
- CI stability: fixed by installing the observability submodule and project with extras; fast/unit matrices green locally.
- Optional DB driver: added Postgres extra so psycopg2-binary is available in CI without burdening all users.
- Logging tests: adjusted to capture the actual stream (stderr vs stdout) so JSON logging assertions are reliable.
- SSE streaming: added line-buffered SSE parsing with carryover between chunks; reasoning and text streaming validated by unit tests.
- Workers timing: restored timing context via compatibility shim (core.common.observability.timing).
- Local path dep: removed; observability is now a submodule with explicit install in CI and dev-setup target.

Key Commits
- CI deps + postgres extra: 3289216
- SSE streaming buffer carryover: present in src/story_engine/core/core/orchestration/llm_orchestrator.py
- Logging test stream capture: tests/unit/test_logging_and_db_settings.py updated
- Workers timing import restored: src/services/*_worker/__main__.py
- hpq logger + lint fixes: src/story_engine/core/core/story_engine/hpq_pipeline.py

Open Review Threads
Note: GitHub’s review threads API requires auth for full context; without tokens we could not pull live thread status. The following entries capture reviewer themes observed during local work and earlier discussion. Please link threads to these rows and mark resolved.

1) Buffer SSE streaming chunks — src/story_engine/core/core/orchestration/llm_orchestrator.py
   - Resolution: Implemented buffered SSE parsing with CRLF/blank-line detection; added fallback to reasoning text when text is empty and reasoning present.
   - Status: Resolved (unit tests pass)

2) Capture stderr in logging test — tests/unit/test_logging_and_db_settings.py
   - Resolution: Test now reads from stdout or stderr; ensures JSON parse uses the last line.
   - Status: Resolved

3) Restore timing import in workers — src/services/*_worker/__main__.py
   - Resolution: Reintroduced timing via shim import from core.common.observability; init_logging_from_env retained.
   - Status: Resolved

4) Remove local-only dependency path for observability
   - Resolution: Removed file:// path; vendored as git submodule under external/llm-observability-suite; CI/dev install in editable mode.
   - Status: Resolved

5) hpq: undefined logger and lint issues — src/story_engine/core/core/story_engine/hpq_pipeline.py
   - Resolution: Added logging.getLogger(__name__); wrapped long lines; renamed single-letter variables.
   - Status: Resolved

6) DB settings/Oracle fallback behavior
   - Resolution: get_db_settings() honors SQLite fallback when Oracle env unhealthy/unset; live DB smoke opt-in preserved.
   - Status: Resolved (tests pass)

7) CI scope and markers
   - Resolution: tests-fast excludes slow/oracle/acceptance; unit job runs fast unit suite; e2e/live behind workflow_dispatch.
   - Status: Resolved

8) Release workflow and PYPI token
   - Resolution: release.yml builds sdist/wheel; publishes if PYPI_API_TOKEN present. No-op otherwise.
   - Status: Resolved

Next Actions
- Reviewer to confirm thread-by-thread: mark as Resolved where applicable.
- If any remaining thread points to now-moved lines, please reference the updated files above; happy to adjust further in a follow-up commit.

