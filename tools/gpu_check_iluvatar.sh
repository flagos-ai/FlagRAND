#!/usr/bin/env bash
set -euo pipefail

ixsmi
GPU_COUNT=$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')
AVAILABLE_GPUS=$(seq -s, 0 "$((GPU_COUNT - 1))")
echo "Available GPUs: $AVAILABLE_GPUS"
