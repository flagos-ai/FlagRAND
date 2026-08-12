#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export FLAGRAND_DELIVERY_VERSION=6
exec bash "$PROJECT_DIR/scripts/delivery/run_ppu_v5.sh" "$@"
