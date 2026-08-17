#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${FLAGRAND_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${FLAGRAND_OUTPUT_DIR:-$PROJECT_DIR/ascend_910c_delivery_results/$RUN_ID}"
BUILD_LOG="$OUTPUT_DIR/image_build.log"
RUN_LOG="$OUTPUT_DIR/acceptance.log"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

echo "========== FLAGRAND ASCEND 910C DELIVERY =========="
echo "Source:  $PROJECT_DIR"
echo "Results: $OUTPUT_DIR"
echo "Build:   $BUILD_LOG"
echo "Run:     $RUN_LOG"
echo

bash docker/ascend910c/build.sh 2>&1 | tee "$BUILD_LOG"

FLAGRAND_RUN_ID="$RUN_ID" \
FLAGRAND_OUTPUT_DIR="$OUTPUT_DIR" \
    bash docker/ascend910c/run.sh 2>&1 | tee "$RUN_LOG"

echo
echo "FLAGRAND ASCEND 910C DELIVERY: PASS"
echo "Results: $OUTPUT_DIR"
