#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${STORYCTL_ENV:-cluster}
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
WORKFLOW=${WORKFLOW:-cluster_run_${TIMESTAMP}}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/cluster}
RUN_DIR="${OUTPUT_ROOT}/${WORKFLOW}"

mkdir -p "${RUN_DIR}"

cat <<INFO >"${RUN_DIR}/run.info"
timestamp=${TIMESTAMP}
workflow=${WORKFLOW}
command=${0} ${*}
INFO

echo "[storyctl] Validating '${ENV_NAME}' environment" >&2
storyctl check --env "${ENV_NAME}" --fail-fast

echo "[storyctl] Running workflow '${WORKFLOW}'" >&2
storyctl run --env "${ENV_NAME}" --workflow "${WORKFLOW}" "$@"

echo "[storyctl] Exporting environment snapshot" >&2
storyctl env export --env "${ENV_NAME}" --format json >"${RUN_DIR}/env.json"

echo "[storyctl] Exporting database rows" >&2
story db-export --workflow "${WORKFLOW}" --output "${RUN_DIR}/outputs.ndjson"

echo "[storyctl] Run artifacts stored in ${RUN_DIR}" >&2
