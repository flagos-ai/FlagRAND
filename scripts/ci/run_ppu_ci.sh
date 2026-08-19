#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export FLAGRAND_PERF_PROFILE="${FLAGRAND_PERF_PROFILE:-smoke}"
export FLAGRAND_PLATFORM=thead

cd "$PROJECT_DIR"
exec "${PYTHON:-python3}" tools/run_tests.py \
  --gpus "${FLAGRAND_DEVICE:-0}" \
  --stages stable \
  --output-dir "${FLAGRAND_RESULTS_DIR:-ppu_results}" \
  --dump-output
