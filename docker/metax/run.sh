#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${FLAGRAND_METAX_IMAGE:-flagrand-metax:c550-maca3.7.0.107}"
CONTAINER="${FLAGRAND_METAX_CONTAINER:-flagrand-muxi-dev}"

docker run -it \
    --name "$CONTAINER" \
    --device=/dev/mxcd \
    --device=/dev/dri \
    --volume "$PROJECT_DIR:/workspace/FlagRAND" \
    --workdir /workspace/FlagRAND \
    "$IMAGE" \
    "$@"
