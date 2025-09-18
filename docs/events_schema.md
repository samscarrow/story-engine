# Event Schema (v1)

Stable keys for structured logs emitted by Story Engine components.

- llm.request
  - provider, endpoint, model
  - job_id, beat, character_id (optional)
- llm.response
  - provider, endpoint, model
  - elapsed_ms, len, ok
  - prompt_tokens, completion_tokens, total_tokens (optional)
  - job_id, beat, character_id (optional)
- pipeline.llm.request | pipeline.llm.response
  - temperature, ctx_len, prompt_len (request)
  - elapsed_ms, ok, len (response)
  - job_id, beat, character_id (optional)
- pipeline.step
  - step: start | craft_scene | simulate_character | simulate_character_done
  - job_id, beat, character_id (optional)
  - elapsed_ms (for *_done)
- error (via log_exception)
  - event: "error"
  - code: GEN_TIMEOUT | GEN_PARSE_ERROR | DB_CONN_FAIL | AI_LB_UNAVAILABLE | CONFIG_INVALID
  - component: e.g., orchestrator, pipeline, poml_integration, koboldcpp, lmstudio
  - message: exception text
  - details: freeform (endpoint, job_id, etc.)

Notes
- Redaction: api_key, authorization, password are redacted by the formatter.
- Sampling: set LOG_SAMPLING_RATE to sample INFO-level events if needed.
