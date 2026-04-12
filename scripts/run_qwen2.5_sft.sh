#!/usr/bin/env bash
set -euo pipefail

# Run Qwen2.5-7B-Instruct SFT on Linux using the local config.
# Usage:
#   bash scripts/run_qwen2.5_sft.sh
#   bash scripts/run_qwen2.5_sft.sh -- --num_train_epochs 1 --per_device_train_batch_size 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_FILE="${CONFIG_FILE:-recipes/Qwen2.5-7B-Instruct/sft/config_local.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/accelerate_configs/zero3.yaml}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 2
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=${LANG:-en_US.UTF-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if command -v accelerate >/dev/null 2>&1; then
  echo "Running with accelerate using $CONFIG_FILE"
  accelerate launch --config_file "$ACCELERATE_CONFIG" src/open_r1/sft.py --config_file "$CONFIG_FILE" "$@"
else
  echo "accelerate not found; running with python using $CONFIG_FILE"
  python -X utf8 src/open_r1/sft.py --config_file "$CONFIG_FILE" "$@"
fi

