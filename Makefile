.PHONY: install test live-poc wheel docker-build docker-run docker-sh docs
 .PHONY: oracle-up oracle-down oracle-test

# Defaults for docker-run (can be overridden: `LM_ENDPOINT=... make docker-run`)
LM_ENDPOINT ?= http://host.docker.internal:8000
LMSTUDIO_MODEL ?=
LLM_TIMEOUT_SECS ?= 300
DOCKER_RUN_EXTRA ?=

install:
	python -m pip install -U pip
	pip install -e .
	pip install pytest

test:
	pytest -q

live-poc:
	bash scripts/run-live-poc.sh

wheel:
	python -m pip install -U build
	python -m build -n

docker-build:
	docker build -t story-engine:local .

docker-run: docker-build
	docker run --rm \
	  --add-host=host.docker.internal:host-gateway \
	  -e LM_ENDPOINT=$(LM_ENDPOINT) \
	  -e LMSTUDIO_MODEL=$(LMSTUDIO_MODEL) \
	  -e LLM_TIMEOUT_SECS=$(LLM_TIMEOUT_SECS) \
	  -v $(CURDIR)/dist:/app/dist \
	  $(DOCKER_RUN_EXTRA) \
	  story-engine:local

docker-run-nobuild:
	docker run --rm \
	  --add-host=host.docker.internal:host-gateway \
	  -e LM_ENDPOINT=$(LM_ENDPOINT) \
	  -e LMSTUDIO_MODEL=$(LMSTUDIO_MODEL) \
	  -e LLM_TIMEOUT_SECS=$(LLM_TIMEOUT_SECS) \
	  -v $(CURDIR)/dist:/app/dist \
	  $(DOCKER_RUN_EXTRA) \
	  story-engine:local

docker-sh:
	docker run --rm -it --entrypoint bash story-engine:local

docs:
	@echo "See docs/e2e-build.md for end-to-end guide"

oracle-up:
	docker compose -f docker-compose.oracle.yml up -d

oracle-down:
	docker compose -f docker-compose.oracle.yml down

# Run only Oracle-marked tests (expects DB_USER/DB_PASSWORD/DB_DSN env set)
oracle-test:
	pytest -q -m oracle
.PHONY: venv
venv:
	@bash scripts/venv-create.sh >/dev/null
	@echo "Run: source \"$$VENV_PATH/bin/activate\" to activate the venv"

.PHONY: test-lmstudio
test-lmstudio:
	@. "$$VENV_PATH/bin/activate" 2>/dev/null || true; \
	pytest tests/test_lmstudio_circuit.py -q

.PHONY: test-orchestrator
test-orchestrator:
	@. "$$VENV_PATH/bin/activate" 2>/dev/null || true; \
	pytest -q -k "lmstudio or kobold" -m "not slow"

.PHONY: e2e
e2e:
	@. "$$VENV_PATH/bin/activate" 2>/dev/null || true; \
	pytest -q -k "e2e" --maxfail=1 --disable-warnings
