#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export FLAGRAND_PERF_PROFILE="${FLAGRAND_PERF_PROFILE:-acceptance}"
exec bash "$PROJECT_DIR/scripts/ci/run_backend_ci.sh" hygon
