#!/usr/bin/env bash
set -euo pipefail

npu-smi info
GPU_COUNT=$(.venv/bin/python -c 'import torch, torch_npu; print(torch.npu.device_count())')
AVAILABLE_GPUS=$(seq -s, 0 "$((GPU_COUNT - 1))")
echo "Available GPUs: $AVAILABLE_GPUS"
