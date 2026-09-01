#!/usr/bin/env bash
set -e

source /workspace/FlagRAND/scripts/ci/backend_env.sh metax
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
exec "$@"
