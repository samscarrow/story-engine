# Database SLIs/SLOs and Minimal Dashboard

## SLIs
- Connect Success Rate: fraction of successful `OracleConnection.connect()` calls.
- Connection Latency (ms): P50/P95 time from attempt start to ready connection.
- Retry Rate: fraction of connects requiring ≥1 retry.
- Pool Acquire Latency (ms): P50/P95 when pooling enabled.
- Query Health: success rate for `SELECT 1 FROM DUAL` health probes.

## SLOs (initial targets)
- ≥ 99.5% connect success over 7-day rolling window.
- P95 connect latency ≤ 1500 ms under normal load.
- Retry rate ≤ 2% under normal load.
- Health probes success ≥ 99.9% during business hours.

## Instrumentation Mapping
- Logs (JSON):
  - event: `oracle pool created` with `elapsed_ms` (pool init time).
  - event: `oracle connect ok` with `attempt`, `elapsed_ms`.
  - event: `error` with `code=DB_CONN_FAIL`, `attempt`, `retry_in_s`.
- Metrics (phase 2): derive counters/histograms from logs via log-based metrics.

## Minimal Dashboard (log-based)
Panels:
- Connect Success Rate: ratio of `oracle connect ok` / (`oracle connect ok` + `error: DB_CONN_FAIL` where terminal=true).
- Connect Latency: histogram/quantiles of `elapsed_ms` from `oracle connect ok`.
- Retry Heatmap: count of `attempt>1` grouped by hour.
- Healthcheck Status: last 24h pass/fail counts from CI runs.

## Alerts
- Critical: Connect Success Rate < 98% over 30 min.
- Warning: P95 connect latency > 2000 ms over 30 min.
- Flaky: Retry rate > 5% over 6 hours.

## Notes
- Enable `LOG_FORMAT=json` and export logs to your collector (e.g., GHA artifacts, Cloud logs).
- Ensure secrets provide DB connectivity in CI for the healthcheck job to run.
