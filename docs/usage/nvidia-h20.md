# FlagRAND NVIDIA H20 运行指南

NVIDIA 是 FlagRAND 的标准 CUDA 后端，H20 是首个正式验证硬件。H20 使用现有通用
Philox4x32-10 Triton kernel、CUDA fast math 和 cached binary launcher，不维护独立的 H20 RNG 副本。

## 环境

节点需要预装可用的 NVIDIA driver、CUDA、PyTorch 和 Triton。H20 对应 Hopper/GH100，预期 compute
capability 为 `9.0`。`setup.sh nvidia` 使用节点当前 Python 的 system site packages，不替换硬件运行时。

```bash
export FLAGRAND_SYSTEM_PYTHON="${FLAGRAND_SYSTEM_PYTHON:-python3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
./setup.sh nvidia
source scripts/ci/backend_env.sh nvidia
bash tools/gpu_check_nvidia.sh
```

如果 `libcurand.so` 不在 `CUDA_HOME` 或动态链接器搜索路径中，设置：

```bash
export FLAGRAND_NVIDIA_CURAND_PATH=/absolute/path/to/libcurand.so
```

## 正确性契约

- raw32 与 NVIDIA `CURAND_RNG_PSEUDO_PHILOX4_32_10` 做 seed/offset 精确比较；
- Uniform f32/f64 与原生 cuRAND 做逐元素比较；
- Normal/LogNormal 验证 replay、流推进和数学统计契约；
- 通用 CUDA 测试继续覆盖 bulk/split、任意 offset、dtype、contiguous 和 current stream。

原生 cuRAND baseline 使用 persistent generator、当前 CUDA stream 和预分配输出。

## 预检

```bash
.venv/bin/python -m benchmark.run_nvidia --preflight-only --device 0
```

预检会编译七 API 的对齐与非对齐路径，比较 offset 0 到 7 的 raw32 位流，并逐元素比较
Uniform f32/f64 的 offset 0 到 3 及 aligned fast path。若 Uniform 不一致，`manifest.json` 会保存
原生值、FlagRAND 值、IEEE-754 位型和对应 raw32 words，供定位转换或 pair 顺序问题。

## Smoke

```bash
.venv/bin/python -m benchmark.run_nvidia --profile smoke --device 0
```

## 正式矩阵

```bash
.venv/bin/python -m benchmark.run_nvidia --profile matrix --device 0
```

正式指标为 `total_ms_median`。每个 shape 的 speedup 为
`NVIDIA cuRAND total_ms_median / FlagRAND total_ms_median`，只在单 API 内聚合，不计算跨 API 平均值，
也不设置额外性能阈值。

## 统一验收

```bash
export FLAGRAND_RUN_ID="h20_$(date +%Y%m%d_%H%M%S)"
export FLAGRAND_PERF_PROFILE=acceptance
bash scripts/ci/run_backend_ci.sh nvidia
```

结果写入：

```text
nvidia_results/ci_<RUN_ID>/summary.json
```

各 benchmark 子进程还会保存 `manifest.json`、`records.jsonl`、`records.csv` 和 `summary.md`。
