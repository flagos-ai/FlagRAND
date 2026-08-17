#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${FLAGRAND_ASCEND_IMAGE:-flagrand-ascend910c:cann9.0-flagtree3.5}"
BUILD_JOBS="${FLAGRAND_BUILD_JOBS:-8}"

cd "$PROJECT_DIR"

build_args=(
    --network=host
    --platform linux/arm64
    --build-arg "MAX_JOBS=$BUILD_JOBS"
    --file docker/ascend910c/Dockerfile
    --tag "$IMAGE"
)

for proxy_name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    proxy_value="${!proxy_name-}"
    if [[ -n "$proxy_value" ]]; then
        build_args+=(--build-arg "$proxy_name=$proxy_value")
    fi
done

docker build "${build_args[@]}" .

echo
echo "Ascend 910C image: $IMAGE"
echo "IMAGE BUILD: PASS"
