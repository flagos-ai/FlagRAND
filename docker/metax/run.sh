#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${FLAGRAND_METAX_IMAGE:-flagrand-metax:c550-flagtree0.7.0.post2-metax3.6-py312}"
CONTAINER="${FLAGRAND_METAX_CONTAINER:-flagrand-muxi-dev}"
VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

docker run -it \
    --name "$CONTAINER" \
    --device=/dev/mxcd \
    --device=/dev/dri \
    --shm-size=4g \
    --env "MACA_VISIBLE_DEVICES=$VISIBLE_DEVICES" \
    --env "CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES" \
    --volume "$PROJECT_DIR:/workspace/FlagRAND" \
    --workdir /workspace/FlagRAND \
    "$IMAGE" \
    "$@"
