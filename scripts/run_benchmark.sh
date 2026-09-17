#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/run_benchmark.sh
#   ./scripts/run_benchmark.sh --base-url http://10.0.0.2:8000 --model my-model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

"${PYTHON_BIN:-python3}" -m llm_benchmark.cli \
  --concurrency "${CONCURRENCIES:-1,4,8,16,32}" \
  --output-dir "${OUTPUT_DIR:-results}" \
  "$@"
