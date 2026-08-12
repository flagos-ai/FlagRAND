#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DELIVERY_VERSION="${FLAGRAND_DELIVERY_VERSION:-5}"
RELEASE_TAG="v${DELIVERY_VERSION}"
RELEASE_LABEL="V${DELIVERY_VERSION}"
RUN_ID="${FLAGRAND_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${FLAGRAND_PERF_PROFILE:-acceptance}"
PACKAGE_RESULTS="${FLAGRAND_PACKAGE_RESULTS:-1}"
LOG_DIR="$PROJECT_DIR/ppu_logs/${RELEASE_TAG}_$RUN_ID"
RESULT_DIR="$PROJECT_DIR/ppu_results/${RELEASE_TAG}_$RUN_ID"
RESULT_ARCHIVE="${FLAGRAND_RESULT_ARCHIVE:-/tmp/FlagRAND_PPU_${RELEASE_TAG}_results_${RUN_ID}.tar.gz}"

if [[ "$PROFILE" != "smoke" && "$PROFILE" != "acceptance" ]]; then
    echo "FLAGRAND_PERF_PROFILE must be smoke or acceptance, got: $PROFILE" >&2
    exit 2
fi

if [[ -e "$LOG_DIR" || -e "$RESULT_DIR" ]]; then
    echo "Refusing to overwrite an existing run: $RUN_ID" >&2
    exit 20
fi

install -d -m 0755 "$LOG_DIR" "$RESULT_DIR"
cd "$PROJECT_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/PPU_SDK/CUDA_SDK}"
export LD_LIBRARY_PATH="/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_DIR/.triton_cache_ppu_isolated_${RELEASE_TAG}}"
export FLAGRAND_USE_LIBTUNER=0
export FLAGRAND_PPU_LLC_COMPAT_LOG="$LOG_DIR/ppu_llc_compat.log"
export FLAGRAND_PPU_COMPILE_LOG="$LOG_DIR/ppu_compile.log"
unset FLAGRAND_FORCE_PORTABLE
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
        "ppu_logs/${RELEASE_TAG}_$RUN_ID" "ppu_results/${RELEASE_TAG}_$RUN_ID"
    chmod 0644 "$RESULT_ARCHIVE"
    tar -tzf "$RESULT_ARCHIVE" >/dev/null
    echo
    echo "Result archive: $RESULT_ARCHIVE"
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

echo "========== PPU $RELEASE_LABEL ENVIRONMENT =========="
"$PYTHON_BIN" - <<'PY' | tee "$LOG_DIR/environment.log"
import logging

import torch
import triton
from triton.runtime import driver

import flagrand
from flagrand.runtime import current_backend, supports_cached_binary_launch, uses_portable_f32_math

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("flagrand.ppu.environment")
logger.info("flagrand: %s", flagrand.__file__)
logger.info("torch: %s", torch.__version__)
logger.info("triton: %s", getattr(triton, "__version__", "unknown"))
logger.info("device available: %s", torch.cuda.is_available())
logger.info("device: %s", torch.cuda.get_device_name(0))
logger.info("target: %s", driver.active.get_current_target())
logger.info("backend: %s", current_backend())
logger.info("cached binary launch: %s", supports_cached_binary_launch())
logger.info("portable f32 math: %s", uses_portable_f32_math())
PY

test -f /usr/local/PPU_SDK/targets/x86_64-linux/lib/libacrand.so
"$PYTHON_BIN" -m compileall -q src benchmark tests tools

"$PYTHON_BIN" tools/validate_operator_registry.py | tee "$LOG_DIR/operator_registry.log"

echo
echo "========== PPU $RELEASE_LABEL PREFLIGHT =========="
if run_capture "$LOG_DIR/preflight.log" \
    bash scripts/run_ppu.sh --preflight-only --results-dir "$RESULT_DIR/preflight"; then
    preflight_rc=0
else
    preflight_rc=$?
fi

echo
echo "========== PPU $RELEASE_LABEL PYTEST CORRECTNESS =========="
if run_capture "$LOG_DIR/pytest.log" "$PYTHON_BIN" -m pytest -q tests/accuracy; then
    pytest_rc=0
else
    pytest_rc=$?
fi

performance_rc=1
if [[ "$preflight_rc" -eq 0 && "$pytest_rc" -eq 0 ]]; then
    echo
    echo "========== PPU $RELEASE_LABEL PERFORMANCE: $PROFILE =========="
    if [[ "$PROFILE" == "acceptance" ]]; then
        performance_args=(
            --profile full
            --acceptance-profile
            --acceptance-baseline ppu_acrand
            --acceptance-min "${FLAGRAND_ACCEPTANCE_MIN:-0.95}"
            --acceptance-max "${FLAGRAND_ACCEPTANCE_MAX:-1.50}"
            --strict-acceptance
            --skip-torch-reference
            --results-dir "$RESULT_DIR/acceptance"
        )
    else
        performance_args=(
            --profile smoke
            --skip-torch-reference
            --results-dir "$RESULT_DIR/smoke"
        )
    fi
    if run_capture "$LOG_DIR/performance.log" bash scripts/run_ppu.sh "${performance_args[@]}"; then
        performance_rc=0
    else
        performance_rc=$?
    fi
fi

echo
echo "========== PPU $RELEASE_LABEL FINAL STATUS =========="
grep -E '^(Preflight:|ACCEPTANCE |Acceptance status:|Completed |Results:)' \
    "$LOG_DIR/preflight.log" "$LOG_DIR/performance.log" 2>/dev/null || true
echo "preflight_rc=$preflight_rc"
echo "pytest_rc=$pytest_rc"
echo "performance_rc=$performance_rc"

package_results
trap - EXIT

if [[ "$preflight_rc" -ne 0 || "$pytest_rc" -ne 0 || "$performance_rc" -ne 0 ]]; then
    echo "PPU $RELEASE_LABEL STATUS: FAIL" >&2
    exit 1
fi

echo "PPU $RELEASE_LABEL STATUS: PASS"
