#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="$1"

case "$BACKEND" in
    ascend-cann900)
        source /usr/local/Ascend/cann/set_env.sh
        export FLAGRAND_VENDOR=ascend
        ;;
    iluvatar)
        export COREX_HOME=/usr/local/corex
        export PATH="$COREX_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$COREX_HOME/lib:$COREX_HOME/lib64:${LD_LIBRARY_PATH:-}"
        export FLAGRAND_VENDOR=iluvatar
        ;;
    thead)
        source /usr/local/PPU_SDK/envsetup.sh
        export FLAGRAND_VENDOR=thead
        ;;
    *)
        echo "Unsupported FlagRAND backend: $BACKEND" >&2
        return 2
        ;;
esac

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
