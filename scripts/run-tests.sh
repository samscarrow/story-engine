#!/usr/bin/env bash
set -euo pipefail

LIVE=false
ENDPOINT="${LM_ENDPOINT:-}"
MODEL="${LMSTUDIO_MODEL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--live) LIVE=true; shift ;;
    -e|--endpoint) ENDPOINT="$2"; shift 2 ;;
    -m|--model) MODEL="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--live] [--endpoint URL] [--model ID]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

header() { echo -e "\n=== $1 ==="; }
run_pytest() {
  pytest "$@"
}

header "Granular • Orchestrator Loader"
run_pytest -q tests/test_orchestrator_loader.py::test_loader_registers_providers

header "Granular • POML Adapter"
run_pytest -q tests/test_poml_adapter.py

header "Granular • POML Extended + Config Flag"
run_pytest -q tests/test_poml_adapter_extended.py tests/test_poml_config_flag.py

header "Granular • Pipeline + Orchestrated Integration"
run_pytest -q tests/test_pipeline_smoke.py tests/test_orchestrated_poml_integration.py::test_orchestrated_scene_and_dialogue_with_poml

if $LIVE; then
  if [[ -z "$ENDPOINT" || -z "$MODEL" ]]; then
    echo "--live requires --endpoint and --model (or set LM_ENDPOINT and LMSTUDIO_MODEL)" >&2
    exit 2
  fi
  export LM_ENDPOINT="$ENDPOINT" LMSTUDIO_MODEL="$MODEL" STORY_ENGINE_LIVE=1
  header "Live • Single Flow (Pilate POML)"
  run_pytest -q tests/test_live_pilate_poml.py::test_live_poml_pilate_flow_minimal
  header "Live • Full Suite"
  run_pytest -q
else
  echo "(Live tests skipped — pass --live and --endpoint/--model or set env vars)"
fi

echo -e "\nAll requested tests completed successfully."

