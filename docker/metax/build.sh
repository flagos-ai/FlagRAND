#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${FLAGRAND_METAX_IMAGE:-flagrand-metax:c550-maca3.7.0.107}"
BASE_IMAGE="${FLAGRAND_METAX_BASE_IMAGE:-cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.20.0-maca.ai3.7.0.107-torch2.8-py310-ubuntu22.04-amd64}"

cd "$PROJECT_DIR"

build_args=(
    --network=host
    --platform linux/amd64
    --build-arg "METAX_BASE_IMAGE=$BASE_IMAGE"
    --file docker/metax/Dockerfile
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
echo "MetaX image: $IMAGE"
echo "IMAGE BUILD: PASS"
