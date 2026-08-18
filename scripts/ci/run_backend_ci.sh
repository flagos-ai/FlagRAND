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

import torch
import triton

import flagrand
from flagrand.runtime import current_backend, current_vendor, device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("flagrand.ci.environment")
logger.info("flagrand: %s", flagrand.__file__)
logger.info("torch: %s", torch.__version__)
logger.info("triton: %s", triton.__version__)
logger.info("vendor: %s", current_vendor())
logger.info("backend: %s", current_backend())
logger.info("device: %s", device.name)
logger.info("device count: %s", device.device_count)
PY

"$PYTHON_BIN" -m compileall -q src benchmark tests tools
"$PYTHON_BIN" tools/validate_operator_registry.py | tee "$LOG_DIR/operator_registry.log"

if [[ -n "$OPERATOR" ]]; then
    run_capture "$LOG_DIR/pytest-$OPERATOR.log" \
        "$PYTHON_BIN" -m pytest -q tests/accuracy -m "$OPERATOR"
    case "$BACKEND" in
        ascend-cann900)
            run_capture "$LOG_DIR/benchmark-$OPERATOR.log" \
                "$PYTHON_BIN" -m benchmark.run_ascend_910c \
                --device "${FLAGRAND_DEVICE:-0}" \
                --profile smoke \
                --distributions "$OPERATOR" \
                --results-dir "$PROJECT_DIR/ascend_910c_results/ondemand_$RUN_ID"
            ;;
        iluvatar)
            run_capture "$LOG_DIR/benchmark-$OPERATOR.log" \
                "$PYTHON_BIN" -m benchmark.run_corex \
                --profile smoke \
                --distributions "$OPERATOR" \
                --results-dir "$PROJECT_DIR/corex_results/ondemand_$RUN_ID"
            ;;
        thead)
            run_capture "$LOG_DIR/benchmark-$OPERATOR.log" \
                "$PYTHON_BIN" -m benchmark.run_ppu \
                --profile smoke \
                --distributions "$OPERATOR" \
                --skip-torch-reference \
                --results-dir "$PROJECT_DIR/ppu_results/ondemand_$RUN_ID"
            ;;
    esac
    exit 0
fi

case "$BACKEND" in
    ascend-cann900)
        export FLAGRAND_PERF_PROFILE="$PROFILE"
        run_capture "$LOG_DIR/hardware-ci.log" bash scripts/ci/run_ascend_910c_ci.sh
        ;;
    iluvatar)
        run_capture "$LOG_DIR/pytest.log" \
            "$PYTHON_BIN" -m pytest -q tests/accuracy \
            -m "not ascend_910c and not normal_f64 and not lognormal_f64"
        COREX_PROFILE="$PROFILE"
        if [[ "$COREX_PROFILE" == "acceptance" ]]; then
            COREX_PROFILE=matrix
        fi
        run_capture "$LOG_DIR/benchmark.log" \
            "$PYTHON_BIN" -m benchmark.run_corex \
            --profile "$COREX_PROFILE" \
            --results-dir "$PROJECT_DIR/corex_results/ci_$RUN_ID"
        ;;
    thead)
        export FLAGRAND_PERF_PROFILE="$PROFILE"
        run_capture "$LOG_DIR/hardware-ci.log" bash scripts/ci/run_ppu_ci.sh
        ;;
esac
