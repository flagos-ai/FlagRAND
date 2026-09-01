#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

mx-smi
"$PYTHON_BIN" - <<'PY'
import torch
import triton
from triton.runtime import driver

from benchmark.muxi_matrix.mcrand import mcrand_library_path, mcrand_version

print(f"Torch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"Triton target: {driver.active.get_current_target()}")
print(f"mcRAND library: {mcrand_library_path()}")
print(f"mcRAND version: {mcrand_version()}")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: name={torch.cuda.get_device_name(index)}")
PY
GPU_COUNT=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
AVAILABLE_GPUS=$(seq -s, 0 "$((GPU_COUNT - 1))")
echo "Available GPUs: $AVAILABLE_GPUS"
