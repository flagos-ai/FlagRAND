#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export FLAGRAND_PERF_PROFILE="${FLAGRAND_PERF_PROFILE:-smoke}"
export FLAGRAND_DEVICE="${FLAGRAND_DEVICE:-0}"
export FLAGRAND_PLATFORM=ascend-cann900

cd "$PROJECT_DIR"
exec "${PYTHON:-python3}" tools/run_tests.py \
  --gpus "$FLAGRAND_DEVICE" \
  --stages stable \
  --output-dir "${FLAGRAND_RESULTS_DIR:-ascend_910c_results}" \
  --dump-output
