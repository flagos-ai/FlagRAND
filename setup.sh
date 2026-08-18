#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$1"

cd "$PROJECT_DIR"
source scripts/ci/backend_env.sh "$BACKEND"

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
.venv/bin/python tools/validate_operator_registry.py
