# Story Engine runtime image
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (curl for simple healthchecks/logs)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml ./
COPY src ./src
COPY external/llm-observability-suite ./external/llm-observability-suite

# Install package and external observability submodule
RUN pip install --upgrade pip \
    && pip install -e external/llm-observability-suite \
    && pip install .

# Default environment expected by the orchestrator
ENV LM_ENDPOINT="http://127.0.0.1:8000" \
    STORY_ENGINE_LIVE=1 \
    LLM_TIMEOUT_SECS=300

# Simple healthcheck: verify upstream LM endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request; u=os.environ.get('LM_ENDPOINT','http://127.0.0.1:8000') + '/v1/models'; urllib.request.urlopen(u, timeout=3).read(); print('ok')"]

# Default command runs a short live demo and writes artifacts to /app/dist
CMD ["story-engine-demo", "--use-poml", "--live", "--runs", "1"]
