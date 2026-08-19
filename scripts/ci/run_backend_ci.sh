#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="$1"
PROFILE="${FLAGRAND_PERF_PROFILE:-smoke}"
OPERATOR="${FLAGRAND_TEST_OP:-}"
RUN_ID="${FLAGRAND_RUN_ID:-${GITHUB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}_${GITHUB_RUN_ATTEMPT:-1}}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
LOG_DIR="$PROJECT_DIR/ci_logs/$BACKEND/$RUN_ID"

cd "$PROJECT_DIR"
source scripts/ci/backend_env.sh "$BACKEND"
mkdir -p "$LOG_DIR"

if [[ -n "${AVAILABLE_GPUS:-}" ]]; then
    FIRST_DEVICE="${AVAILABLE_GPUS%%,*}"
    if [[ "$BACKEND" == "ascend-cann900" ]]; then
        export ASCEND_RT_VISIBLE_DEVICES="$FIRST_DEVICE"
        export FLAGRAND_DEVICE=0
    else
        export CUDA_VISIBLE_DEVICES="$FIRST_DEVICE"
    fi
fi

run_capture() {
    local log_file="$1"
    shift
    "$@" 2>&1 | tee "$log_file"
    return "${PIPESTATUS[0]}"
}

run_capture "$LOG_DIR/environment.log" "$PYTHON_BIN" - <<'PY'
import logging
import pkgutil

import torch
import triton
import triton.backends

from triton.runtime.driver import driver

import flagrand
from flagrand.runtime import current_backend, current_vendor, device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("flagrand.ci.environment")
logger.info("flagrand: %s", flagrand.__file__)
logger.info("torch: %s", torch.__version__)
logger.info("triton: %s", triton.__version__)
logger.info("triton target: %s", driver.active.get_current_target())
logger.info(
    "triton backend modules: %s",
    ", ".join(sorted(module.name for module in pkgutil.iter_modules(triton.backends.__path__))),
)
logger.info("vendor: %s", current_vendor())
logger.info("backend: %s", current_backend())
logger.info("device: %s", device.name)
logger.info("device count: %s", device.device_count)
PY

"$PYTHON_BIN" -m compileall -q src benchmark tests tools
"$PYTHON_BIN" tools/validate_operator_registry.py | tee "$LOG_DIR/operator_registry.log"
"$PYTHON_BIN" -m pytest -q tests/accuracy/test_unified_runner.py | tee "$LOG_DIR/unified-runner-tests.log"

case "$BACKEND" in
    ascend-cann900)
        RESULTS_DIR="$PROJECT_DIR/ascend_910c_results/ci_$RUN_ID"
        ;;
    iluvatar)
        RESULTS_DIR="$PROJECT_DIR/corex_results/ci_$RUN_ID"
        ;;
    thead)
        RESULTS_DIR="$PROJECT_DIR/ppu_results/ci_$RUN_ID"
        ;;
esac

export FLAGRAND_PLATFORM="$BACKEND"
export FLAGRAND_PERF_PROFILE="$PROFILE"

RUNNER_ARGS=(
    --gpus 0
    --output-dir "$RESULTS_DIR"
    --dump-output
)
if [[ -n "$OPERATOR" ]]; then
    RUNNER_ARGS+=(--ops "$OPERATOR")
else
    RUNNER_ARGS+=(--stages stable)
fi

run_capture "$LOG_DIR/unified-runner.log" \
    "$PYTHON_BIN" tools/run_tests.py "${RUNNER_ARGS[@]}"
