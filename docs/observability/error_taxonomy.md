# Error Taxonomy

Stable, grep‑friendly error codes emitted via `core.common.observability.log_exception()` and structured logs. Use these codes for dashboards and alerts.

## Codes
- `AI_LB_UNAVAILABLE` — upstream AI load balancer or provider unavailable, circuit open, or 429 saturation.
- `GEN_TIMEOUT` — request exceeded timeout budget.
- `GEN_PARSE_ERROR` — provider returned invalid/empty payload; normalization failed.
- `DB_CONN_FAIL` — database connection or pool acquire failed.
- `CONFIG_INVALID` — required configuration missing or invalid.

## Components (examples)
- `orchestrator` — core LLM orchestration layer
- `lmstudio` / `kobold` — provider adapters
- `pipeline` — narrative pipeline
- `db.oracle` / `db.postgres` — storage adapters
- `persona_agents`, `recursive_sim`, `standardized_llm` — higher‑level modules

## Shape
```
{ "event": "error", "code": "GEN_TIMEOUT", "component": "orchestrator", "message": "...", "details": { "attempt": 2, "endpoint": "..." } }
```

## Guidance
- Always include: `component`, minimal `message`, and contextual `details` (endpoint/model/attempt/workflow).
- Use `init_logging_from_env()` early so logs carry `service`, `trace_id`, `correlation_id`.
- Prefer taxonomy codes over ad‑hoc strings for alerting.

## Mapping → Remediation
- `AI_LB_UNAVAILABLE` → check upstream health `/health`, LB `Retry-After`, reduce concurrency or increase capacity.
- `GEN_TIMEOUT` → raise `LLM_TIMEOUT_SECS` or reduce prompt size; consider `LM_RETRY_ATTEMPTS` with budget.
- `GEN_PARSE_ERROR` → inspect provider payload; enforce non‑empty `choices[0].message.content`.
- `DB_CONN_FAIL` → verify wallet, DSN, network; tune pool and retry backoff.
- `CONFIG_INVALID` → validate env via startup checks; add CI config lint.

