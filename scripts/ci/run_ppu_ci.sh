#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export FLAGRAND_PERF_PROFILE="${FLAGRAND_PERF_PROFILE:-smoke}"
export FLAGRAND_PACKAGE_RESULTS=0

exec bash "$PROJECT_DIR/scripts/delivery/run_ppu_v6.sh"
