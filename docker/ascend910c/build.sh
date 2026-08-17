#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${FLAGRAND_ASCEND_IMAGE:-flagrand-ascend910c:cann9.0-flagtree3.5}"
BUILD_JOBS="${FLAGRAND_BUILD_JOBS:-8}"

cd "$PROJECT_DIR"
docker build \
    --platform linux/arm64 \
    --build-arg MAX_JOBS="$BUILD_JOBS" \
    --file docker/ascend910c/Dockerfile \
    --tag "$IMAGE" \
    .

echo
echo "Ascend 910C image: $IMAGE"
echo "IMAGE BUILD: PASS"
