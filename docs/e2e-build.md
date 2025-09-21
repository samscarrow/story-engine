# Story Engine – Shippable E2E Build

This guide covers packaging, container, CI, and how to run the end‑to‑end live pipeline against an existing ai‑lb/LM Studio compatible endpoint.

## Artifacts
- Python package: sdist + wheel under `dist/`
- Docker image: `story-engine:local` (or `ghcr.io/<owner>/story-engine:<sha>` via CI)
- CLI tools:
  - `story-engine-poml` – render POML templates
  - `story-engine-demo` – run the simulation demo (mock or live orchestrator)

## Environment Contract
- `LM_ENDPOINT` (required for live): URL to the OpenAI/LM Studio compatible endpoint (e.g., `http://127.0.0.1:8000`).
- `LMSTUDIO_MODEL` (optional): model id string if your server requires it.
- `STORY_ENGINE_LIVE` (bool): set to `1` to enable live flows in tests/demos.
- `LLM_TIMEOUT_SECS` (int): per‑request timeout override (default 60).
- `LM_PREFER_SMALL` (bool): prefer smaller models when selecting from `/v1/models`.

## Local Dev
- Install and test:
  - `make install`
  - `make test`
- Live PoC (uses an existing endpoint; does not start a server):
  - `LM_ENDPOINT=http://127.0.0.1:8000 make live-poc`

## Build Package
- `make wheel`
- Outputs in `dist/story_engine-*.whl` and `*.tar.gz`

## Docker
- Build and run once:
  - `make docker-build`
  - `make docker-run` (defaults to `LM_ENDPOINT=http://host.docker.internal:8000`)
- Compose run (writes artifacts to `./dist`):
  - `cd deploy && LM_ENDPOINT=http://host.docker.internal:8000 docker compose up --build story-engine-demo`
- If your Docker does not support `host-gateway`:
  - Use host networking: `DOCKER_RUN_EXTRA="--network host" LM_ENDPOINT=http://127.0.0.1:8000 make docker-run-nobuild`
  - Or expose your ai-lb on a LAN/Tailscale IP and set `LM_ENDPOINT` to that address.

Image defaults:
- CMD `story-engine-demo --use-poml --live --runs 1`
- HEALTHCHECK: queries `${LM_ENDPOINT}/v1/models`

## CI (GitHub Actions)
- Workflow: `.github/workflows/e2e_build.yml`
  - Builds sdist/wheel and uploads artifacts
  - Builds Docker image, pushes to GHCR on push to main/tags (uses `GITHUB_TOKEN`)

## Production Notes
- Configure your ai‑lb/LM Studio endpoint with suitable model(s) and capacity.
- For Tailscale environments, set `LM_ENDPOINT` to the `*.ts.net` address and ensure the runner/container can resolve it.
- Token and privacy: this repo assumes policies are in code and PII is limited to internal fixtures.

## Troubleshooting
- 403 from LM endpoint: ensure server allows your IP/token (if applicable).
- `No healthy providers`: the endpoint must serve `/v1/models` with at least one model.
- Timeouts: raise `LLM_TIMEOUT_SECS` or choose a faster/smaller model via `LMSTUDIO_MODEL`.

