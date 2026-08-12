#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DELIVERY_VERSION="${FLAGRAND_DELIVERY_VERSION:-5}"
RELEASE_TAG="v${DELIVERY_VERSION}"
RELEASE_LABEL="V${DELIVERY_VERSION}"
RUN_ID="${FLAGRAND_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${FLAGRAND_PERF_PROFILE:-acceptance}"
DEVICE="${FLAGRAND_DEVICE:-0}"
SPEEDUP_MIN="${FLAGRAND_SPEEDUP_MIN:-0.95}"
PACKAGE_RESULTS="${FLAGRAND_PACKAGE_RESULTS:-1}"
LOG_DIR="$PROJECT_DIR/ascend_910c_logs/${RELEASE_TAG}_$RUN_ID"
RESULT_DIR="$PROJECT_DIR/ascend_910c_results/${RELEASE_TAG}_$RUN_ID"
RESULT_ARCHIVE="${FLAGRAND_RESULT_ARCHIVE:-/home/FlagRAND_910C_${RELEASE_TAG}_results_${RUN_ID}.tar.gz}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"

if [[ "$PROFILE" != "smoke" && "$PROFILE" != "acceptance" ]]; then
    echo "FLAGRAND_PERF_PROFILE must be smoke or acceptance, got: $PROFILE" >&2
    exit 2
fi

if [[ ! -f "$CANN_ENV" ]]; then
    CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [[ ! -f "$CANN_ENV" ]]; then
    echo "CANN environment script not found" >&2
    exit 10
fi

if [[ -e "$LOG_DIR" || -e "$RESULT_DIR" ]]; then
    echo "Refusing to overwrite an existing run: $RUN_ID" >&2
    exit 20
fi

source "$CANN_ENV"
install -d -m 0755 "$LOG_DIR" "$RESULT_DIR"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_DIR/.triton_cache_ascend910c_isolated_${RELEASE_TAG}}"
export FLAGRAND_USE_LIBTUNER=0
export FLAGRAND_DEVICE="$DEVICE"
mkdir -p "$TRITON_CACHE_DIR"
RESULTS_PACKAGED=0

run_capture() {
    local log_file="$1"
    shift
    "$@" 2>&1 | tee "$log_file"
    return "${PIPESTATUS[0]}"
}

package_results() {
    if [[ "$PACKAGE_RESULTS" != "1" || "$RESULTS_PACKAGED" == "1" ]]; then
        return
    fi
    tar -czf "$RESULT_ARCHIVE" -C "$PROJECT_DIR" \
        "ascend_910c_logs/${RELEASE_TAG}_$RUN_ID" "ascend_910c_results/${RELEASE_TAG}_$RUN_ID"
    chmod 0644 "$RESULT_ARCHIVE"
    tar -tzf "$RESULT_ARCHIVE" >/dev/null
    echo
    echo "Container result archive: $RESULT_ARCHIVE"
    sha256sum "$RESULT_ARCHIVE"
    RESULTS_PACKAGED=1
}

package_on_exit() {
    local exit_code=$?
    trap - EXIT
    set +e
    package_results
    exit "$exit_code"
}

trap package_on_exit EXIT

echo "========== ASCEND 910C $RELEASE_LABEL ENVIRONMENT =========="
"$PYTHON_BIN" - <<'PY' | tee "$LOG_DIR/environment.log"
import logging
import os

import torch
import torch_npu
import triton
from triton.runtime import driver

import flagrand

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("flagrand.ascend910c.environment")
device = int(os.environ["FLAGRAND_DEVICE"])
torch.npu.set_device(device)
logger.info("flagrand: %s", flagrand.__file__)
logger.info("torch: %s", torch.__version__)
logger.info("torch_npu: %s", torch_npu.__version__)
logger.info("triton: %s", getattr(triton, "__version__", "unknown"))
logger.info("NPU available: %s", torch.npu.is_available())
logger.info("NPU count: %s", torch.npu.device_count())
logger.info("device: %s", torch.npu.get_device_name(device))
logger.info("target: %s", driver.active.get_current_target())
PY

"$PYTHON_BIN" -m compileall -q src benchmark tests tools
"$PYTHON_BIN" tools/validate_operator_registry.py | tee "$LOG_DIR/operator_registry.log"

echo
echo "========== ASCEND 910C $RELEASE_LABEL PREFLIGHT =========="
if run_capture "$LOG_DIR/preflight.log" \
    bash scripts/run_ascend_910c_cann9.sh \
        --device "$DEVICE" \
        --preflight-only \
        --results-dir "$RESULT_DIR/preflight"; then
    preflight_rc=0
else
    preflight_rc=$?
fi

echo
echo "========== ASCEND 910C $RELEASE_LABEL PYTEST CORRECTNESS =========="
if run_capture "$LOG_DIR/pytest.log" "$PYTHON_BIN" -m pytest -q tests/accuracy; then
    pytest_rc=0
else
    pytest_rc=$?
fi

performance_rc=1
if [[ "$preflight_rc" -eq 0 && "$pytest_rc" -eq 0 ]]; then
    if [[ "$PROFILE" == "smoke" ]]; then
        echo
        echo "========== ASCEND 910C $RELEASE_LABEL PERFORMANCE: SMOKE =========="
        if run_capture "$LOG_DIR/smoke.log" \
            bash scripts/run_ascend_910c_cann9.sh \
                --device "$DEVICE" \
                --profile smoke \
                --results-dir "$RESULT_DIR/smoke"; then
            performance_rc=0
        else
            performance_rc=$?
        fi
    else
        apis=(
            raw32
            uniform_f32
            uniform_f64
            normal_f32
            normal_f64
            lognormal_f32
            lognormal_f64
        )
        mkdir -p "$LOG_DIR/per_api" "$RESULT_DIR/per_api"
        performance_rc=0
        for api in "${apis[@]}"; do
            log_file="$LOG_DIR/per_api/${api}.log"
            echo
            echo "========== ASCEND 910C $RELEASE_LABEL PER-API: $api =========="
            if run_capture "$log_file" \
                bash scripts/run_ascend_910c_cann9.sh \
                    --device "$DEVICE" \
                    --profile full \
                    --acceptance-profile \
                    --distributions "$api" \
                    --speedup-min "$SPEEDUP_MIN" \
                    --results-dir "$RESULT_DIR/per_api/$api"; then
                api_rc=0
            else
                api_rc=$?
            fi
            summary_count="$(grep -Ec "^API pass[[:space:]]+${api}[[:space:]]" "$log_file" || true)"
            if [[ "$api_rc" -ne 0 || "$summary_count" -ne 1 ]]; then
                performance_rc=1
            fi
            echo "api=$api rc=$api_rc pass_summary_count=$summary_count"
        done
    fi
fi

echo
echo "========== ASCEND 910C $RELEASE_LABEL FINAL STATUS =========="
grep -E '^(Preflight:|Completed |Results:)' "$LOG_DIR/preflight.log" 2>/dev/null || true
if [[ -d "$LOG_DIR/per_api" ]]; then
    for log_file in "$LOG_DIR"/per_api/*.log; do
        grep -E '^(API |Completed |Results:|Fatal error:)' "$log_file" || true
    done
fi
echo "preflight_rc=$preflight_rc"
echo "pytest_rc=$pytest_rc"
echo "performance_rc=$performance_rc"

package_results
trap - EXIT

if [[ "$preflight_rc" -ne 0 || "$pytest_rc" -ne 0 || "$performance_rc" -ne 0 ]]; then
    echo "ASCEND 910C $RELEASE_LABEL STATUS: FAIL" >&2
    exit 1
fi

echo "ASCEND 910C $RELEASE_LABEL STATUS: PASS"
