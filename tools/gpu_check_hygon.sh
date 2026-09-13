#!/usr/bin/env bash
set -euo pipefail
"${HY_SMI_PATH:-/opt/hyhal/bin/hy-smi}"
GPU_COUNT=$("${PYTHON_BIN:-.venv/bin/python}" -c 'import torch; print(torch.cuda.device_count())')
AVAILABLE_GPUS=$(seq -s, 0 "$((GPU_COUNT - 1))")
echo "Available GPUs: $AVAILABLE_GPUS"
