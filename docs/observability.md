# Observability

- Init via `init_logging_from_env()` as early as possible in entrypoints and services.
- Context is automatic: `service`, `trace_id`, and `correlation_id` are injected into all log records once `init_logging_from_env()` runs.
- Use `get_logger(name, **context)` to attach additional structured fields like `job_id`, `step`, `model`, etc.
- Emit structured events (JSON) with stable keys.

## Env Vars
- `LOG_FORMAT`: `json` (default) or `text`
- `LOG_LEVEL`: `DEBUG|INFO|WARNING|ERROR` (default: `INFO`)
- `LOG_DEST`: `stdout|stderr|file` (default: `stdout`)
- `LOG_FILE_PATH`: file path when `LOG_DEST=file`
- `LOG_SERVICE_NAME`: logical service name (default: `story-engine`)
- `TRACE_ID`: preset trace id (optional; auto-generated if absent)

Notes
- When using the in-memory or RabbitMQ message bus, the engine now propagates `correlation_id` into the logging context for each consumed message. This makes it easy to trace events across producers and consumers without modifying handlers.

## Error Taxonomy
Codes in `core/common/observability.py`:
- `GEN_TIMEOUT`
- `GEN_PARSE_ERROR`
- `DB_CONN_FAIL`
- `AI_LB_UNAVAILABLE`
- `CONFIG_INVALID`

Use `log_exception(logger, code, component, exc, **context)` to record errors.

## Instrumentation
- Orchestrator: emits `llm.request` and JSON errors via taxonomy.
- Pipeline: emits `pipeline.llm.request|response` and errors.
- Workers: use env-driven logging; DLQ logs unchanged.

### Baseline Metrics
See `docs/observability/baseline.md` for the recommended timers/counters and budgets. For event shapes and helpers, see `docs/observability/metrics_events.md`.

### Orchestrator metrics
- Log-based metrics helpers (stdlib-only) are available via `story_engine.core.core.common.observability`:
  - `metric_event(name, value=None, **tags)`
  - `inc_metric(name, n=1, **tags)`
  - `observe_metric(name, value_ms, **tags)`

Emitted metrics for LMStudio provider:
- `llm.lmstudio.gen_ms` — generation latency in ms (tags: `endpoint`, `model`)
- `llm.lmstudio.attempts` — attempts used for a request (float, tags: `endpoint`, `model`)
- `llm.lmstudio.retries` — retry count if any (counter, tags: `endpoint`, `model`)

Circuit breaker events are logged with context:
- `circuit.open` — includes `provider` and `window_sec`
- `circuit.reset` — includes `provider`

Error taxonomy codes commonly used:
- `AI_LB_UNAVAILABLE`, `GEN_TIMEOUT` (see `core/common/observability.py`)

## Testing
- `tests/test_logging_config.py` validates JSON log shape.
- Extend with scenario tests to assert error event structure.
