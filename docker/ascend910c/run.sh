#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${FLAGRAND_ASCEND_IMAGE:-flagrand-ascend910c:cann9.0-flagtree3.5}"
DEVICE="${FLAGRAND_DEVICE:-0}"
RUN_ID="${FLAGRAND_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${FLAGRAND_OUTPUT_DIR:-$PROJECT_DIR/ascend_910c_delivery_results/$RUN_ID}"

mkdir -p "$OUTPUT_DIR"

docker run --rm \
    --runtime=ascend \
    --privileged \
    --shm-size=8g \
    --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    --volume /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
    --volume /usr/local/sbin:/usr/local/sbin:ro \
    --volume /etc/ascend_install.info:/etc/ascend_install.info:ro \
    --volume "$OUTPUT_DIR:/output" \
    --env FLAGRAND_DEVICE="$DEVICE" \
    --env FLAGRAND_RUN_ID="$RUN_ID" \
    --env FLAGRAND_RESULT_ARCHIVE="/output/FlagRAND_910C_results_${RUN_ID}.tar.gz" \
    "$IMAGE" \
    bash scripts/delivery/run_ascend_910c.sh

echo
echo "Ascend 910C results: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
