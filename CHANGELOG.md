# Changelog

All notable changes to this project will be documented in this file.

## 0.0.2 - 2025-09-13

- LMStudio client resilience
  - Added jittered exponential backoff for network/timeouts/5xx and response_format fallback.
  - Honored `LM_RETRY_ATTEMPTS`, `LM_RETRY_BASE_DELAY`, and `LM_REQUEST_BUDGET_MS` envs.
  - Ensured circuit breaker opens after consecutive `GenerationError`s; logs `circuit.open/reset`.
  - Emitted metrics: `llm.lmstudio.gen_ms`, `llm.lmstudio.attempts`, `llm.lmstudio.retries`.
- Documentation
  - Added “Client Resilience Knobs (engine)” to `docs/ai-lb-integration.md`.
  - Cross-linked ai-lb client guidance from `SYSTEM_ARCHITECTURE.md`.
  - Expanded README with standard venv usage without direnv.
- Developer experience
  - Standardized virtualenv location to `~/.venvs/<slug>-py<ver>`; deprecated local `.venv`.
  - Makefile targets: `venv`, `test-lmstudio`.
- CI
  - Added `Orchestrator Suite` workflow to run LMStudio and Kobold provider tests on PRs and pushes.
