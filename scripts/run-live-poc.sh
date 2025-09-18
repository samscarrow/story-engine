#!/usr/bin/env bash
set -euo pipefail

# Live pipeline PoC runner (uses an existing ai-lb/LM Studio-compatible endpoint)
# - Does NOT start any server; it only validates the endpoint and runs the live test.

ENDPOINT="${LM_ENDPOINT:-${ENDPOINT:-http://127.0.0.1:8000}}"
MODEL_ID="${LMSTUDIO_MODEL:-${MODEL_ID:-}}"

echo "Checking endpoint health: ${ENDPOINT}/v1/models ..."
curl -fsS --connect-timeout 2 --max-time 5 "${ENDPOINT}/v1/models" >/dev/null

echo "Running live POML pipeline test ..."
export STORY_ENGINE_LIVE=1
export LLM_TEST_TIMEOUT=${LLM_TEST_TIMEOUT:-20}
# Raise orchestrator provider timeout if not set
export LLM_TIMEOUT_SECS=${LLM_TIMEOUT_SECS:-300}
export LM_ENDPOINT="${ENDPOINT}"
# Optional: set a model id if your server requires it; otherwise the orchestrator uses 'auto'.
if [[ -n "${MODEL_ID}" ]]; then
  export LMSTUDIO_MODEL="${MODEL_ID}"
fi

# Run with -m slow selection to ensure the live test is included
pytest -q -m slow tests/test_live_pilate_poml.py -k test_live_poml_pilate_flow_minimal -rs

echo "\nPoC complete."
