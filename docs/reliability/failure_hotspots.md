# Failure Hotspots (Initial Scan)

Summary based on `generation_error_*.json` at repository root.

## Observed Error Types
- ValueError (Empty response text): 22 occurrences
- TimeoutError (LMStudio HTTP): 11 occurrences
- Exception (misc): 2 occurrences
- HealthCheckFailed: 1 occurrence

## Likely Causes
- Upstream returned 200/OK with empty or invalid payload.
- Upstream node slow/saturated; request timed out at client.
- Occasional health probe failures leading to circuit‑open.

## Mitigations (Env‑Tunable)
- Increase resilience on LMStudio path:
  - `LM_RETRY_ATTEMPTS=2` (or 3) and `LM_RETRY_BASE_DELAY=0.3`.
  - Set `LLM_TIMEOUT_SECS=90` for heavier prompts.
  - Use `LM_REQUEST_BUDGET_MS` to cap total time across retries.
- Prefer AI Load Balancer with sticky sessions when available (`LM_ENDPOINT` → LB, `LM_MODEL=auto`).
- Ensure upstream nodes stream or return non‑empty `choices[0].message.content`.
- For periodic instability, enable circuit breaker tuning:
  - `LLM_CB_THRESHOLD=3`, `LLM_CB_WINDOW_SEC=15` (adjust per SLOs).

## Next Checks
- Verify LM Studio/AI LB logs around timestamps for 429/5xx.
- Add a smoke test that asserts non‑empty content and captures model ID.
- Track `llm.lmstudio.gen_ms` and retry counts in logs; alert if p95 > budget.

