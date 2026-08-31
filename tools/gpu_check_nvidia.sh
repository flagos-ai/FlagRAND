#!/usr/bin/env bash
set -euo pipefail

nvidia-smi
.venv/bin/python - <<'PY'
import torch
import triton

print(f"Torch: {torch.__version__}")
print(f"Torch CUDA runtime: {torch.version.cuda}")
print(f"Triton: {triton.__version__}")
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(
        f"GPU {index}: name={properties.name} "
        f"compute_capability={properties.major}.{properties.minor} "
        f"memory={properties.total_memory}"
    )
PY
GPU_COUNT=$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')
AVAILABLE_GPUS=$(seq -s, 0 "$((GPU_COUNT - 1))")
echo "Available GPUs: $AVAILABLE_GPUS"
