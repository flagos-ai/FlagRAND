#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="$1"

case "$BACKEND" in
    nvidia)
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
        export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
        export FLAGRAND_VENDOR=nvidia
        ;;
    ascend-cann900)
        set +u
        source /usr/local/Ascend/cann/set_env.sh
        set -u
        export FLAGRAND_VENDOR=ascend
        ;;
    iluvatar)
        export COREX_HOME=/usr/local/corex
        export CUDA_HOME="$COREX_HOME"
        export PATH="$COREX_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$COREX_HOME/lib:$COREX_HOME/lib64:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="$COREX_HOME/lib:$COREX_HOME/lib64:${LIBRARY_PATH:-}"
        export CPATH="$COREX_HOME/include${CPATH:+:$CPATH}"
        export FLAGRAND_VENDOR=iluvatar
        ;;
    thead)
        set +u
        source /usr/local/PPU_SDK/envsetup.sh
        set -u
        export FLAGRAND_VENDOR=thead
        ;;
    *)
        echo "Unsupported FlagRAND backend: $BACKEND" >&2
        return 2
        ;;
esac

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
