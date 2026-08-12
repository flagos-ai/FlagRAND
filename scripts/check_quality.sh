#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"

echo "========== OPERATOR REGISTRY =========="
"$PYTHON_BIN" tools/validate_operator_registry.py

echo
echo "========== PYTHON STATIC COMPILATION =========="
"$PYTHON_BIN" -m compileall -q src benchmark tests tools

if command -v pre-commit >/dev/null 2>&1; then
    echo
    echo "========== PRE-COMMIT =========="
    pre-commit run --all-files --show-diff-on-failure
else
    echo
    echo "PRE-COMMIT: SKIP (pre-commit is not installed in this environment)"
fi

echo
echo "QUALITY CHECK: PASS"
