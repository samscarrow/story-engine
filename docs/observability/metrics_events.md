# Metrics Events (Log-Based)

This project emits lightweight metric events as structured JSON logs to avoid external dependencies while providing actionable telemetry.

## Event Shapes
- Timer (ms):
  - `{ "event": "metric", "metric": "db.oracle.connect_ms", "value": 42.3, "unit": "ms", "type": "timer", "dsn": "localhost/XEPDB1", "pooled": false }`
- Counter:
  - `{ "event": "metric", "metric": "worker.plot.messages", "value": 1, "type": "counter", "component": "plot_worker" }`

## Current Metrics
- `db.oracle.connect_ms` — elapsed time to establish Oracle connection
- `llm.lmstudio.gen_ms` — LM Studio generation duration
- `llm.koboldcpp.gen_ms` — KoboldCpp generation duration
- `worker.plot.handle_ms` — Plot worker handler execution time

## Emission Points
- Emitted via `core.common.observability.observe_metric()` and `timing()` context manager.
- Logs are JSON when `LOG_FORMAT=json` (default).

## Consumption
- Local: pipe logs to `jq`/`grep` to extract metrics for quick checks.
- CI: upload logs as artifacts or parse in a later step.
- Prod: forward logs to your aggregator (e.g., Loki/Elastic) and build dashboards around `event=metric`.

## Sampling
- Use `LOG_SAMPLING_RATE` for info-level logs when volume is high.
- Metrics events are intentionally lightweight; sample at the log-router if needed.

## Configuration
- `LOG_FORMAT=json|text`, `LOG_LEVEL`, `LOG_DEST`, `LOG_SERVICE_NAME`
- For LLM retries/circuit breakers, tune via env:
  - `LLM_CB_THRESHOLD`, `LLM_CB_WINDOW_SEC`, `LM_RETRY_ATTEMPTS`, `KOBOLD_RETRY_ATTEMPTS`, `KOBOLD_RETRY_BASE_DELAY`

