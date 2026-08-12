#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export FLAGRAND_PERF_PROFILE="${FLAGRAND_PERF_PROFILE:-smoke}"
export FLAGRAND_DEVICE="${FLAGRAND_DEVICE:-0}"
export FLAGRAND_SPEEDUP_MIN="${FLAGRAND_SPEEDUP_MIN:-0.95}"
export FLAGRAND_PACKAGE_RESULTS=0

exec bash "$PROJECT_DIR/scripts/delivery/run_ascend_910c_v6.sh"
