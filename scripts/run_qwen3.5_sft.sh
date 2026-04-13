#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/recipes/Qwen3.5-27B-Instruct/sft/config_local.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT_DIR}/recipes/accelerate_configs/zero3.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Accelerate config not found: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi

exec accelerate launch --config_file "${ACCELERATE_CONFIG}" "${ROOT_DIR}/src/open_r1/sft.py" --config "${CONFIG_FILE}" "$@"

