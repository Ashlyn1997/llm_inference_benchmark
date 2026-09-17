#!/usr/bin/env bash
set -euo pipefail

# Long prompt / KV Cache pressure benchmark. Extra CLI arguments override defaults.
# Example:
#   VLLM_BASE_URL=http://10.0.0.2:8000 ./scripts/run_long_context_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

"${PYTHON_BIN:-python3}" -m llm_benchmark.cli \
  --base-url "${VLLM_BASE_URL:-http://127.0.0.1:8000}" \
  --model "${VLLM_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
  --prompts-file "${PROMPTS_FILE:-prompts/long_2k_pool.jsonl}" \
  --concurrency "${CONCURRENCIES:-8,16,32,64}" \
  --requests "${REQUESTS:-64}" \
  --max-tokens "${MAX_TOKENS:-256}" \
  --warmup-requests "${WARMUP_REQUESTS:-0}" \
  --gpu-index "${GPU_INDEX:-0}" \
  --gpu-monitor-interval-ms "${GPU_MONITOR_INTERVAL_MS:-200}" \
  --output-dir "${OUTPUT_DIR:-results/long_context}" \
  "$@"
