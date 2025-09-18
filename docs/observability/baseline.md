# Observability Baseline

This baseline lists the core metrics to emit across services, their types, tags, and suggested budgets for dev/CI. Metrics are emitted as structured log events via `metric_event`/`observe_metric` from `core.common.observability`.

## Timers (ms)
- `db.oracle.connect_ms`
  - When: Oracle `connect()` completes (with or without pool)
  - Tags: `dsn`, `pooled`
  - Budgets: dev p95 < 500ms, CI p95 < 300ms
- `llm.lmstudio.gen_ms`
  - When: LMStudio chat completion attempt finishes
  - Tags: `endpoint`, `model`, `attempts`
  - Budgets: small prompt p95 < 2000ms (local LB), CI < 5000ms
- `llm.koboldcpp.gen_ms`
  - When: KoboldCpp generation completes
  - Tags: `endpoint`, `model`, `attempts`
- `worker.plot.handle_ms`
  - When: Plot worker handles a message end-to-end
  - Tags: `workflow`, `status`

## Counters
- `llm.lmstudio.retries`
  - Increment per retry beyond first attempt
  - Tags: `endpoint`, `model`
- `worker.plot.messages`
  - Increment per handled message
  - Tags: `status`
- `errors.total`
  - Optional global error counter; emit alongside taxonomy errors
  - Tags: `code`, `component`

## Gauges (optional)
- `lb.nodes.inflight`
  - From AI-LB monitor/metrics endpoint
  - Tags: `node`, `model`

## Conventions
- All metrics are emitted as logs (`{"event":"metric", "metric": "..."}`) to avoid external deps.
- Prefer low-cardinality tags; mask secrets; never include PII.
- Sampling: keep metrics unsampled; sample info logs via `LOG_SAMPLING_RATE` as needed.

## Alert Ideas (log-derived)
- Spike in `AI_LB_UNAVAILABLE` errors and sustained `llm.lmstudio.retries`.
- `db.oracle.connect_ms` p95 regressions beyond budgets.
- `llm.lmstudio.gen_ms` p95 exceeds budget + empty response detections in acceptance runs.

## CI Usage
- Parse logs from test runs to compute simple stats and regressions.
- For acceptance pipeline, export logs as artifact; dashboards grep `event=metric`.

