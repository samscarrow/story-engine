# Contributing

Thank you for your interest in contributing to Story Engine! This guide focuses on local environment setup, testing, and conventions to keep contributions smooth and consistent.

## Environment

- Preferred: direnv + standard venv under `~/.venvs`
  - Install direnv (https://direnv.net/) and run `direnv allow` in the repo.
  - `.envrc` creates/activates a virtualenv at `~/.venvs/<slug>-py<major.minor>` via `scripts/venv-create.sh`.
  - `VENV_PATH` is exported and used for Python tooling.
- Without direnv:
  - `make venv` or `bash scripts/venv-create.sh`
  - `source "$VENV_PATH/bin/activate"`
  - `pip install -e . pytest pytest-asyncio`
- Deprecated: a project-local `.venv` directory — please remove it if present.

## Tests

- Run fast tests: `pytest -q -m "not slow"`
- Orchestrator-only: `pytest -q -k "lmstudio or kobold"`
- Golden/silver suites are separate workflows; avoid running them locally unless needed.

## Style & Lint

- Ruff and Black checks run in CI.
- Prefer small, surgical changes with clear PR descriptions.

## Secrets & Live Tests

- Live tests rely on environment configuration and are opt-in (marked `slow`). Do not commit secrets; use GitHub Actions secrets for CI.

## Docs

- Update relevant docs when adding features (e.g., `docs/ai-lb-integration.md`, `docs/observability.md`).
- Record noteworthy changes in `CHANGELOG.md`.

## Oracle/DB

- Use `docs/oracle/local_dev.md` for local XE guidance.
- Quick health check: `python scripts/db_health.py --verbose`.

Thanks again — happy hacking!
