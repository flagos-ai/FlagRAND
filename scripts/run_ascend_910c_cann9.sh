#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANN_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh

if [[ ! -f "$CANN_ENV" ]]; then
    CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
fi

if [[ ! -f "$CANN_ENV" ]]; then
    echo "CANN environment script not found: $CANN_ENV" >&2
    exit 10
fi

source "$CANN_ENV"

if [[ "${ASCEND_HOME_PATH:-}" != *"cann-9.0"* ]]; then
    echo "Expected CANN 9.0, but ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset}" >&2
    echo "Use the validated CANN 9.0 container with /usr/local/Ascend/cann-9.0.0." >&2
    exit 11
fi

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_DIR/.triton_cache_ascend910c_isolated_v5}"
export FLAGRAND_USE_LIBTUNER=0
mkdir -p "$TRITON_CACHE_DIR" "$PROJECT_DIR/ascend_910c_results"

exec python3 -u -m benchmark.run_ascend_910c "$@"
