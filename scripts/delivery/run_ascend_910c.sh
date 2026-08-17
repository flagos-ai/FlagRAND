#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${FLAGRAND_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export FLAGRAND_PERF_PROFILE="${FLAGRAND_PERF_PROFILE:-acceptance}"
export FLAGRAND_DEVICE="${FLAGRAND_DEVICE:-0}"
export FLAGRAND_SPEEDUP_MIN="${FLAGRAND_SPEEDUP_MIN:-0.95}"
export FLAGRAND_RUN_ID="$RUN_ID"
export FLAGRAND_RESULT_ARCHIVE="${FLAGRAND_RESULT_ARCHIVE:-/output/FlagRAND_910C_results_${RUN_ID}.tar.gz}"

mkdir -p "$(dirname "$FLAGRAND_RESULT_ARCHIVE")"
exec bash "$PROJECT_DIR/scripts/delivery/run_ascend_910c_v6.sh"
