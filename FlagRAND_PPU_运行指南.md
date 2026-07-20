# FlagRAND PPU 运行指南

本文假定 PPU 机器上已经存在完整代码目录：

```bash
cd ~/flagrand_ppu_work/FlagRAND-pr1
```

## 1. 环境设置

每次重新进入容器或终端后，先执行：

```bash
export PYTHONPATH="$PWD/src:$PWD"
unset FLAGRAND_FORCE_PORTABLE
unset FLAGRAND_USE_LIBTUNER

export CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
export LD_LIBRARY_PATH="/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/targets/x86_64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p ppu_logs ppu_results
```

正式性能测试不要设置 `FLAGRAND_FORCE_PORTABLE=1`。该变量只作为兼容性排障开关；打开后会禁用 cached binary launcher，并使部分 f32 normal/lognormal 路径走 portable math，可能降低性能。

## 2. 环境检查

```bash
python - <<'PY'
import torch
import triton
from triton.runtime import driver
import flagrand

print("torch:", torch.__version__)
print("triton:", getattr(triton, "__version__", "unknown"))
print("flagrand:", flagrand.__file__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("target:", driver.active.get_current_target())
PY
```

预期至少包含：

```text
cuda available: True
device: PPU-ZW810E
```

再确认正式性能路径没有被强制切到 portable：

```bash
python - <<'PY'
from flagrand.runtime import supports_cached_binary_launch, uses_portable_f32_math

print("cached binary launch:", supports_cached_binary_launch())
print("portable f32 math:", uses_portable_f32_math())
PY
```

正式性能测试预期：

```text
cached binary launch: True
portable f32 math: False
```

## 3. 正确性验证

先跑 preflight：

```bash
python ppu_flagrand_matrix.py --preflight-only \
  2>&1 | tee ppu_logs/preflight.log
```

预期：

```text
Preflight: pass
```

再跑 pytest：

```bash
python -m pytest -q tests/test_api.py tests/test_philox_cuda.py \
  2>&1 | tee ppu_logs/pytest.log
```

预期全量通过。

## 4. 性能测试 Shape

正式聚合纳入小 N、奇数、非 4 倍数和 1M 附近 shape，但不纳入 16M 级别大 N。

Raw/Uniform shape：

```text
1,2,3,4,5,7,8,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025,4095,4096,4097,65535,65536,65537,1048575,1048576,1048577
```

Normal/LogNormal shape：

```text
2,4,6,8,10,14,16,18,126,128,130,1022,1024,1026,4094,4096,4098,65534,65536,65538,1048574,1048576,1048578
```

## 5. Raw/Uniform 性能

```bash
python ppu_flagrand_matrix.py \
  --profile smoke \
  --distributions raw32,uniform_f32,uniform_f64 \
  --sizes 1,2,3,4,5,7,8,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025,4095,4096,4097,65535,65536,65537,1048575,1048576,1048577 \
  --skip-torch-reference \
  --results-dir ppu_results/generic_uniform_wide \
  2>&1 | tee ppu_logs/generic_uniform_wide.log
```

## 6. Normal/LogNormal 性能

```bash
python ppu_flagrand_matrix.py \
  --profile smoke \
  --distributions normal_f32,normal_f64,lognormal_f32,lognormal_f64 \
  --sizes 2,4,6,8,10,14,16,18,126,128,130,1022,1024,1026,4094,4096,4098,65534,65536,65538,1048574,1048576,1048578 \
  --skip-torch-reference \
  --results-dir ppu_results/generic_normal_wide \
  2>&1 | tee ppu_logs/generic_normal_wide.log
```
