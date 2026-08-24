# FlagRAND PPU-ZW810E 运行指南

PPU 路径保留标准 Philox4x32-10 位流和七个 public API，并同时对比 PPU 的 `libcurand.so` 兼容实现与原生 `libacrand.so`。正式性能 baseline 使用 `ppu_acrand`。

## 当前统一 CI / 验收入口

统一后端名称为 `thead`。当前 CI 依赖由 `conf/backends.yaml` 固定为 Python 3.12、
`torch==2.9.0+ppu2.0.0.oe` 和 `triton==3.5.0+ppu2.0.0.oe`。

```bash
git clone --branch main --single-branch git@github.com:flagos-ai/FlagRAND.git
cd FlagRAND

./setup.sh thead
source scripts/ci/backend_env.sh thead
bash tools/gpu_check_thead.sh

RUN_ID="ppu_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_RUN_ID="$RUN_ID" \
FLAGRAND_PERF_PROFILE=acceptance \
bash scripts/ci/run_backend_ci.sh thead

.venv/bin/python -m json.tool "ppu_results/ci_${RUN_ID}/summary.json"
```

以下章节保留专用 PPU SDK 环境、底层矩阵与 LibTuner 的诊断方法。

## 环境准备

使用节点已有的 PPU SDK、torch 和 Triton 环境，不要由 pip 替换硬件依赖。离线节点直接通过 `PYTHONPATH` 加载源码：

```bash
cd /path/to/FlagRAND-pr1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import flagrand; print(flagrand.__file__)'
```

已验证两套厂商环境：历史环境为 Triton 3.5、target backend `cuda`；当前环境为 FlagTree
`0.6.0+ppu3.6`、Triton 3.6、target backend `ppu`、PPU SDK 2.0.0。当前 PPU
backend 生成带 `ppu.` namespace 的 LLVM inline assembly，而配套 `ppu-llc` 接受无 namespace
opcode。`flagrand.runtime.ppu_compiler` 只在 active target 为 `ppu` 时，对送入编译器的临时副本执行
namespace 归一化，并补齐 backend 异常路径需要的编译日志变量；不修改系统 Triton、PPU SDK 或原始编译输入。

PPU 3.6 使用经过实机 ABI 验证的 cached binary launcher，同时保留 portable FP32 math。正式运行预期：

```text
backend: ppu
cached binary launch: True
portable f32 math: True
```

如需留存兼容层审计日志，可设置：

```bash
export FLAGRAND_PPU_LLC_COMPAT_LOG="$PWD/ppu_logs/ppu_llc_compat.log"
export FLAGRAND_PPU_COMPILE_LOG="$PWD/ppu_logs/ppu_compile.log"
```

如动态库不在系统搜索路径，通过 `PPU_CURAND_LIBRARY` 和 `PPU_ACRAND_LIBRARY` 指定完整文件路径。

## 正确性预检

```bash
bash scripts/run_ppu.sh --preflight-only
```

预检覆盖 raw32 seed=0 golden vector、bulk/split、offset 流连续性、七 API replay，以及 baseline 动态库可用性。完整 pytest 正确性：

```bash
python3 -m pytest -q tests/accuracy
```

## 端到端性能

正式交付推荐直接使用一键入口，它会完成环境核验、正确性、性能和结果打包：

```bash
FLAGRAND_PERF_PROFILE=acceptance bash scripts/delivery/run_ppu_v6.sh
```

Smoke：

```bash
bash scripts/run_ppu.sh --profile smoke --skip-torch-reference
```

固定宽 shape 验收矩阵：

```bash
bash scripts/run_ppu.sh \
  --profile full \
  --acceptance-profile \
  --acceptance-baseline ppu_acrand \
  --acceptance-min 0.95 \
  --strict-acceptance \
  --skip-torch-reference
```

2026-08-10 在 PPU 3.6 环境上的正式结果为：preflight pass，324 条性能记录零失败，七个 API
全部通过 `0.95x` 单 API 聚合门禁。

CI 等价入口：

```bash
FLAGRAND_PERF_PROFILE=acceptance bash scripts/ci/run_ppu_ci.sh
```

正式指标是 `total_ms_median`：包含 public API、下发、设备执行和末尾同步。每个 shape 的 speedup 为 `PPU baseline total_ms_median / FlagRAND total_ms_median`；先逐 shape 计算，再在单 API 内取算术平均，不计算跨 API 平均值。结果写入 `ppu_results/`，硬件回归时应完整保留 manifest、JSONL、CSV 与 Markdown summary。

LibTuner 专项说明见 [PPU LibTuner Guide](ppu_libtuner.md)，既有性能与正确性数据见 [PPU 报告](../reports/ppu_performance_and_correctness.md)。
