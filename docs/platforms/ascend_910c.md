# FlagRAND 华为 Ascend 910C（CANN 9.0）运行指南

## 验证环境

当前交付以节点容器 `FlagBLAS_ywb` 中的 CANN 9.0.0 环境为准。实测组合为 Python 3.11.15、torch 2.9.0+cpu、torch_npu 2.9.0.post2，以及由 FlagTree 分发的 Triton 3.5.1；Triton target 为 `GPUTarget(backend='npu', arch='Ascend910_9382', warp_size=0)`。包元数据可能不显示 `triton-ascend`，不能因此误判 backend。

不得在容器内用 pip 替换 torch、torch_npu、Triton 或 CANN。离线节点不执行 editable build，直接设置源码路径：

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import flagrand; print(flagrand.__file__)'
```

## 七个 API 的实现契约

| API | 910C 路径 | 正式 baseline |
|---|---|---|
| raw32 | Triton 计数器哈希，signed int32 输出 | `Tensor.random_` / ACLNN Random |
| uniform_f32 | 同一计数器哈希融合 24-bit Uniform 映射 | `Tensor.uniform_` / ACLNN Uniform |
| normal_f32 | 字节分解、Irwin-Hall-3 与奇多项式校正 | `Tensor.normal_` / ACLNN Normal |
| lognormal_f32 | Normal 近似路径融合 `exp` | ACLNN Normal 后接 `exp_` |
| uniform_f64 | CANN 原生 FP64 兼容路径 | 同一 CANN 原生路径 |
| normal_f64 | CANN 原生 FP64 兼容路径 | 同一 CANN 原生路径 |
| lognormal_f64 | CANN FP64 Normal 后接 FP64 `exp_` | 同一 CANN 原生路径 |

910C 的 FP32 性能路径是独立 Triton 实现，但不承诺与 cuRAND Philox 位流逐位相同；正确性口径是 replay、bulk/split、offset 连续性及统计特征。三个 FP64 API 是为了完全对齐华为 baseline 而采用的兼容路径，不构成独立实现性能对比。实测 Uniform/Normal 虽存储为 binary64，但随机有效分辨率与 float32 相同；LogNormal 的 FP64 `exp` 会生成非零低位，这并不增加输入 Normal 的随机熵。

固定 kernel block 来自 910C 独立 sweep：raw32=1024、uniform_f32=4096、normal_f32=2048、lognormal_f32=2048。CLI 的 `--block-size` 仍保留用于接口与诊断记录，但正式快路径使用上述逐 API 常量。

### Normal/LogNormal FP32 host fast path

设备端 kernel 稳定后，小 shape 的瓶颈是 Python 下发链而不是 NPU 计算。分层测量显示 direct compiled
launch 的端到端时间约为 `41.36 us`，而原 public API 会随 host 状态升至约 `42.9–50.7 us`；1000 次
连续调用均命中 cached binary launcher，没有 compile/JIT launch。v6 因此在 public API 完成 generator、
dtype、contiguous 和 launch option 校验后，直接调用 910C Normal cached launcher，避免重复 backend
查找、通用 distribution 分派和 chunk 分派。普通调用只计算一次 seed、offset 和 epoch；跨 `2^32`
counter epoch 时自动回退到完整 windowed 路径，replay、offset 与 bulk/split 语义不变。

2026-08-11 的 910C 定向实测中，正确性为 `13 passed`；`normal_f32` 三轮 API speedup 分别为
`1.175x`、`1.131x`、`1.130x`，`lognormal_f32` 为 `1.315x`。完整定位、实现与证据见
[Normal FP32 host fast path 报告](../reports/ascend_910c_normal_host_fast_path_20260811.md)。

## 一键正式回归

交付包解压进 `FlagBLAS_ywb` 容器后，在容器内执行：

```bash
set -euo pipefail
cd /path/to/FlagRAND-pr1
FLAGRAND_PERF_PROFILE=acceptance \
FLAGRAND_DEVICE=0 \
FLAGRAND_SPEEDUP_MIN=0.95 \
    bash scripts/delivery/run_ascend_910c_v6.sh
```

脚本依次完成环境核验、Python 静态编译、正确性预检、pytest、七个 API 的独立进程正式验收和结果打包。预检确认 CANN 9.0、910C/9382 证据、Triton `npu`/`ascend` backend、ACLNN RNG 符号、replay、流连续性、F64 精度契约以及 baseline replay。诊断开关 `--allow-non-cann9` 和 `--allow-unverified-910c` 会把结果标为非正式交付数据。

## 端到端性能

Smoke：

```bash
bash scripts/run_ascend_910c_cann9.sh --device 0 --profile smoke
```

910C 正式性能验收只采用逐 API 独立进程模式。它隔离不同 API 的进程生命周期、运行时状态和设备执行顺序；每个 API 使用完整 acceptance shape、full warmup/repeat，并独立执行 `0.95x` 门禁。不得用同进程七 API 矩阵替代正式结果。

单 API 诊断命令为：

```bash
bash scripts/run_ascend_910c_cann9.sh \
  --device 0 \
  --profile full \
  --acceptance-profile \
  --distributions uniform_f32 \
  --speedup-min 0.95
```

CI 等价入口：

```bash
FLAGRAND_PERF_PROFILE=acceptance \
FLAGRAND_DEVICE=0 \
bash scripts/ci/run_ascend_910c_ci.sh
```

正式指标为 `total_ms_median`，包含 Python public API、下发、设备执行和末尾同步。每个 repeat 前后同步，首次编译由 warmup 排除；`device_ms_median` 和 enqueue 只作诊断。每个 shape 的 speedup 为 `CANN total_ms_median / FlagRAND total_ms_median`，只在单 API 内聚合，不计算跨 API 平均值。

结果保存在 `ascend_910c_results/<timestamp>_<profile>_910c_philox/`，包括 `manifest.json`、`records.jsonl`、`records.csv` 和 `summary.md`。正式回传必须保留整个目录，不要只保存终端输出。

逐 API 前台验收及结果下载的完整命令见 [阶段四硬件回归手册](../delivery/phase4_hardware_regression.md)。JumpServer 网络不参与 NPU 计时，默认无需后台执行。

## 常见故障

- `backend='npu'` 是当前 FlagTree 发行版的正常值，不应强制要求字符串 `ascend`。
- BiSheng 报 UB overflow 时先减小单 program block；不要把编译失败误判为算法错误。
- 当前 lowering 会把部分 uint32 右移当成算术移位，代码通过掩码显式恢复逻辑右移。
- 直接 uint32-to-float32 vcast 不可用；先把已约束到 24 bit 的值 bitcast 为 int32，再转换为 float32。
- 不要清空 Triton cache 后只测首次调用，也不要使用 `ASCEND_LAUNCH_BLOCKING=1` 生成正式数据。
- macOS 复制产生的 `._*.py` 是 AppleDouble 文件，不是源码；打包和 `.gitignore` 均应排除。
