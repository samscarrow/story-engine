# ai-lb Integration (Concise Guide)

## Endpoint
- Use `http://localhost:8000/v1/chat/completions` (configured in `llm_config.json`).

## Payload
- `model`: `"auto"` (or set via env `LM_MODEL`)
- `stream`: `false`
- `messages`, `temperature`, `max_tokens`: unchanged (driven by component profiles)

## Optional Ops Tuning
- Prefer models: set `PREFERRED_MODELS=mistralai/mistral-small-3.2,qwen/qwen3-8b`
- Routing strategy: `ROUTING_STRATEGY=LEAST_LOADED`
- Capacity caps: monitor `DEFAULT_MAXCONN` or `MAXCONN_MAP`

## Observability
- Response headers: `x-selected-model`, `x-routed-node`
- Health/metrics: `GET /v1/nodes`, `GET /metrics`

## Verification Checklist
- Non-stream JSON: with `stream: false`, responses are JSON (not event-stream)
- Auto model: with `model: "auto"`, LB chooses a real model from `/v1/models`
- Routing metadata: confirm `x-selected-model` and `x-routed-node` present
- Failover: stop an upstream; LB reroutes; after `MAX_RETRIES`, returns `502`

## Inputs

- Chat: `POST /v1/chat/completions`
  - `model`: concrete ID or sentinel ("auto"/"default" or env `LM_MODEL`)
  - `messages`: OpenAI-style array (optional system, required user)
  - Optional passthrough: `temperature`, `max_tokens`, etc.
  - `stream`: `false` (returns JSON) or `true` (Server-Sent Events)
  - Optional `?node=<host:port>` to force a specific upstream
- Embeddings: `POST /v1/embeddings` with `model`, `input`
- Models: `GET /v1/models` (aggregated, de-duplicated list)

## Behavior

- Model auto-selection: when `model` is a sentinel, LB picks a real model
  - Prefers `PREFERRED_MODELS` order, else first available
  - Forwards request with resolved model ID
- Routing: `ROUND_ROBIN`, `RANDOM`, or `LEAST_LOADED` (capacity-aware)
- Failover: retries up to `MAX_RETRIES` on network/5xx/capacity errors
- Non-stream chat: returns JSON; adds headers `x-selected-model`, `x-routed-node`

## Configuration (env)

- Core:
  - `LOAD_BALANCER_PORT` (default 8000)
  - `ROUTING_STRATEGY` (`ROUND_ROBIN`|`RANDOM`|`LEAST_LOADED`)
  - `REQUEST_TIMEOUT_SECS`, `MAX_RETRIES`
  - `CIRCUIT_BREAKER_THRESHOLD`, `CIRCUIT_BREAKER_TTL_SECS`
- Model selection:
  - `MODEL_SENTINELS` (default `auto,default`)
  - `LM_MODEL` (optional sentinel override; default `auto`)
  - `PREFERRED_MODELS` (comma/semicolon list priority)
  - `AUTO_MODEL_STRATEGY` (reserved; default `any_first`)
- Redis:
  - `REDIS_HOST` (default `localhost`), `REDIS_PORT` (default `6379`)
- Discovery/health (if using monitor service):
  - `SCAN_HOSTS`, `SCAN_PORTS`, `SCAN_INTERVAL`
  - `DEFAULT_MAXCONN`, `MAXCONN_MAP` for per-node concurrency caps

## Useful Endpoints

- `GET /v1/models` → available models
- `GET /v1/nodes` → healthy nodes with inflight, failures, maxconn
- `GET /v1/eligible_nodes?model=ID` → nodes serving a specific model
- `GET /health` → LB health summary
- `GET /metrics` → Prometheus text (requests, up, inflight, failures)

That’s it: send OpenAI-style payloads; use `model: "auto"` if you don’t want to pin; tune behavior via the env vars above.
