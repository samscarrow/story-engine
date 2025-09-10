Here’s the remaining backlog after PR‑2 (packaging) — scoped, sequenced, and acceptance‑oriented. Each PR is independent but ordered to reduce risk and align with DOCS/
  ARCHITECTURE_IMPROVEMENT_PLAN.md.

  Overview

  - Goal: complete migration to stateless services on a broker, harden security, externalize state, add observability, and stabilize packaging/imports.
  - Status: PR‑1 (Security hygiene) and PR‑2a/b/c (Packaging + src layout) completed; in‑memory bus + 2 services scaffolded; DB outbox/idempotency helpers added.

  PR‑3 — Contracts & Topics

  - Scope: Add versioned message contracts and topic/ DLQ map.
  - Tasks:
      - Add core/contracts/{scene.py, dialogue.py, evaluation.py} (dataclasses + validate()).
      - Define topic names, DLQs, retry policy doc (docs/messaging.md).
      - Update services to use contracts (validate on consume/publish).
  - Deliverables: schemas, docs, and enforcement in handlers.
  - Acceptance: invalid payloads rejected with clear errors; topic/DLQ catalog documented.

  PR‑4 — Broker Adapter (RabbitMQ)

  - Scope: Implement RabbitMQ adapter; config‑driven switch (MQ_TYPE=rabbitmq).
  - Tasks:
      - core/messaging/rabbitmq.py (pika or aio‑pika): connect, declare exchanges/queues/DLQs, prefetch, ack/NACK, reconnection.
      - Update services to use adapter via config; keep in‑memory as default for tests.
      - Add deploy/compose/docker-compose.full.yml: postgres + rabbitmq + minimal services.
  - Acceptance: end‑to‑end “plot.request→plot.done” via RabbitMQ locally; DLQ on max retries.

  PR‑5 — State Externalization & Migrations

  - Scope: Persist pipeline state and artifacts; outbox publisher.
  - Tasks:
      - SQL migrations (db/migrations/*.sql): jobs, job_events, artifacts, message_outbox, processed_messages.
      - Implement outbox publisher (scripts/outbox_publisher.py) and wire insert_outbox/mark_processed in services.
      - Persist/retrieve outline/scene/dialogue artifacts (Postgres JSONB) keyed by job_id.
  - Acceptance: crash/resume safe; dedup via processed_messages; artifacts retrievable via job_id.

  PR‑6 — Observability (Metrics & Tracing)

  - Scope: Add metrics and optional tracing.
  - Tasks:
      - Prometheus counters/histograms (prometheus-client): mq_messages_consumed_total, mq_messages_failed_total, message_processing_seconds, llm_calls_total, llm_call_seconds.
      - Optional OTEL spans (opentelemetry-sdk): publish/consume/LLM call spans; propagate correlation_id/trace_id headers.
      - Expose /-/metrics in services (simple HTTP server or pushgateway job).
  - Acceptance: metrics exposed and scrapeable; traces visible when OTEL enabled.

  PR‑7 — Async I/O & Resilience

  - Scope: Remove sync bottlenecks; add timeouts/backoff/circuit breakers.
  - Tasks:
      - LLM calls via aiohttp with per‑request timeouts, retries (exponential jitter), and circuit breaker.
      - DB: consider psycopg (async) or keep sync with thread executor; ensure timeouts and retries.
      - Standardize retryable vs permanent errors and NACK policy.
  - Acceptance: LLM/network failures don’t block workers; latency regression tests pass.

  PR‑8 — CI/CD & Secret Scanning

  - Scope: Add GH Actions pipeline and security scans.
  - Tasks:
      - Workflows: lint (ruff/flake8), type‑check (mypy optional), unit + integration (in‑memory default), optional compose e2e (rabbitmq+postgres).
      - Secret scanning: detect-secrets (uses .secrets.baseline), Trivy for container scans (optional).
      - Cache pip; publish coverage summary; status badges.
  - Acceptance: PRs blocked on lint/tests/secret scan; compose e2e job green.

  PR‑9 — Secrets Remediation & History Rewrite

  - Scope: Eliminate committed secrets and formalize secret sourcing.
  - Tasks:
      - git‑filter‑repo plan and scripts to purge PEM/JKS/P12/wallets from history (with team coordination).
      - Secret manager integration stub (config points to env/secret store; repo never reads from files).
      - Update SECURITY.md with rotation playbooks; ensure logs redact sensitive keys.
  - Acceptance: scanner reports no secrets; history rewritten; deployment docs use secret manager.

  PR‑10 — IoC & Boundaries

  - Scope: Invert control in core/story_engine and orchestration.
  - Tasks:
      - Constructors accept orchestrator, storage, cache, and bus clients instead of constructing internally.
      - Extract builders/factories (core/common/cli_utils.py or new factories.py) for wiring.
      - Remove residual cross‑component calls; use message passing or injected dependencies.
  - Acceptance: no import cycles; modules unit‑testable via injected fakes; tests updated.

  PR‑11 — Serviceization: Scene/Dialogue/Evaluation Workers

  - Scope: Extract remaining workers and coordinator if needed.
  - Tasks:
      - services/{scene_worker, dialogue_worker, evaluation_worker}/main.py with proper handlers.
      - Optional event aggregator/coordinator to drive cross‑step orchestration.
  - Acceptance: full story path: plot.request → plot.done → scene.request → scene.done → dialogue.request → dialogue.done → evaluation.done.

  PR‑12 — Backpressure & Scaling

  - Scope: Tune prefetch/concurrency and idempotency.
  - Tasks:
      - Prefetch and concurrency per worker; document recommended scaling (e.g., scene N=5).
      - Idempotency keys derived from input; enforce processed_messages check.
      - Load tests and backpressure behavior doc.
  - Acceptance: workers sustain load without message loss; DLQ rate within threshold.

  PR‑13 — Config & Feature Flags

  - Scope: Centralize configuration and flags.
  - Tasks:
      - Document MQ/DB/LLM settings; add flags for strict POML, fallback providers, validation toggles.
      - Config schema validation (lightweight) and example env files (without secrets).
  - Acceptance: config mis‑configs fail fast with clear errors; config matrix documented.

  PR‑14 — Developer Experience

  - Scope: Linting, typing, formatting, hooks.
  - Tasks:
      - Add ruff/flake8, black, mypy; integrate in pre‑commit and CI.
      - TYPE hints across high‑churn modules (orchestration/story_engine), minimal pragmas.
  - Acceptance: repo passes lint/format/type checks; hooks enabled.

  PR‑15 — Docs & Runbooks

  - Scope: Operational and architecture docs.
  - Tasks:
      - Architecture: updated sequence/interaction diagrams; messaging catalog; DLQ handling.
      - Runbooks: service health, scaling, troubleshooting, outbox lag remediation.
      - On‑call: SLOs, alerts, dashboards (Prometheus/Grafana), error budget policy.
  - Acceptance: onboarding‑ready docs; links from README; diagrams current.

  PR‑16 — Release & Versioning

  - Scope: Versioned releases and changelogs.
  - Tasks:
      - SemVer policy; changelog (Keep a Changelog); release GitHub workflow.
      - Optional: package publish or internal registry images; SBOM generation.
  - Acceptance: tagged releases produce artifacts/images; changelog entries automated.

  PR‑17 — Packaging Finalization

  - Scope: Polish package distribution.
  - Tasks:
      - Extras for adapters (rmq/redis/obs) finalized; constraints or lockfile strategy.
      - Validate pip install story-engine and console scripts on fresh envs.
  - Acceptance: clean install; CLIs available; optional extras resolve.

  PR‑18 — Security Scanning & Container Hardening

  - Scope: Harden images and scan regularly.
  - Tasks:
      - Container: non‑root, read‑only FS, minimal base images; secrets mounted at runtime.
      - CI: Trivy/Grype; SAST option if desired; SBOM attestation (cosign build optional).
  - Acceptance: images pass scans; guidance documented.

  PR‑19 — E2E Test Harness

  - Scope: Deterministic E2E in compose with stubs.
  - Tasks:
      - docker‑compose.e2e.yml: Postgres + RabbitMQ + all workers + stub LLM service.
      - Script runs a full story job and asserts evaluation.done with artifacts persisted.
  - Acceptance: one‑command E2E passes locally and in CI job.

  PR‑20 — Data Retention & Compliance (optional)

  - Scope: TTLs and data classification.
  - Tasks:
      - TTL/purge for cache/artifacts; configurable retention windows.
      - Basic classification (PII flags) and redaction in logs/outputs.
  - Acceptance: retention policies enforceable; docs reflect operational needs.

  Suggested Order & Milestones

  - Milestone 1 (Enable end‑to‑end via broker): PR‑3, PR‑4, PR‑5, PR‑11, PR‑12
  - Milestone 2 (Reliability/Obs): PR‑6, PR‑7, PR‑8
  - Milestone 3 (Security/Packaging polish): PR‑9, PR‑17, PR‑18
  - Milestone 4 (DX/Docs/Release): PR‑13, PR‑14, PR‑15, PR‑16
  - Optional/Continuous: PR‑19, PR‑20
