# PPU MTGP32 SDK 基线构建与使用

2026-10-28：SDK device 函数加 host 封装已完成 PPU 上的 14 阶段构建、
11 项状态与续接检查、11 项 N=1Mi 统计检查，以及八 API × 七个代表 shape 的 56 个性能配对。
完整 shape 矩阵与性能优化继续进行。SDK Poisson 在 λ≥1000 的正态近似仍需单独说明，
1Mi 的统计结果不能替代已发现的大样本 CDF 偏差。

## 基线组成

基线标识是 `ppu_sdk_mtgp32_device`，源码为
[`mtgp32_device.hg`](../../../benchmark/ppu_matrix/mtgp32_device.hg)，Python 调用层为
[`mtgp32_device.py`](../../../benchmark/ppu_matrix/mtgp32_device.py)。
SDK 提供 MTGP32 递推、参数初始化和分布转换；host 封装管理状态、参数、尾部输出与当前 Torch stream。

| 输出 API | SDK device 函数 |
| --- | --- |
| raw32 | `hgrand(state)` |
| uniform_f32 / uniform_f64 | `hgrand_uniform` / `hgrand_uniform_double` |
| normal_f32 / normal_f64 | `hgrand_normal` / `hgrand_normal_double`，随后应用 mean、stddev |
| lognormal_f32 / lognormal_f64 | `hgrand_log_normal` / `hgrand_log_normal_double` |
| poisson | `hgrand_poisson(state, lambda)` |

状态由 `hgrandMakeMTGP32Constants` 和 `hgrandMakeMTGP32KernelState` 初始化。
封装采用 200 个 SDK 状态，每个参与的线程块使用 256 个线程，同一状态由一个线程块更新。
输出按固定的 200 states × 256 lanes 顺序循环；短输出仅启动涉及的状态，参与的线程块仍完整执行
256 个 lane 的 SDK 调用。最后一组中未消费的输出保存在尾部缓存，后续同参数调用继续消费。

每种 API 和分布参数组合分别持有状态、cursor 和尾部缓存。normal/lognormal 的 key 包括
mean、stddev，Poisson 的 key 包括 lambda；同一组合连续调用保持自己的进度，切换组合不会消费
其他组合的状态。各组合从相同 seed 的 SDK 初始状态开始，这不表示它们在统计上相互独立。
支持 seed reset，不提供非零绝对 offset。切换 CUDA stream 时，调用方安排先后依赖。

FlagRAND 当前 MTGP32 实现采用 192 个状态，而此基线采用 SDK 的 200 个状态；两者比较分布正确性
和各自的重复性，不要求输出逐位相同。基线的状态数量和分布转换方式应随性能结果一起说明。

## 与原生 host acRAND 的区别

| 路径 | 调用方式 | 结果中的标识 |
| --- | --- | --- |
| 原生 host acRAND | `libacrand.so` 中的 `acrandGenerate*` | `ppu_acrand` |
| 本文 MTGP32 基线 | 自建共享库调用 SDK 的 `hgrand*` device 函数，host 层管理状态和尾部 | `ppu_sdk_mtgp32_device` |

历史设备记录中，acRAND 1400 的八项 MTGP32 host API 返回 `Unimplemented API`。SDK 头文件内的
MTGP32 device 函数是另一条调用路径；本封装不调用 `acrandGenerate*`。统一矩阵对 MTGP32 选择
device 基线，对 Philox、XORWOW、MRG32K3A 仍选择原生 host acRAND。

## clone 后自动构建与使用

仓库包含 SDK device+host 封装源码及构建脚本，运行时使用目标节点或厂商镜像已有的
PPU SDK、HGGC、宿主 C++ 编译器、Torch 和 FlagTree。clone 包含此改动的版本后，
`tools/run_tests.py` 自动构建并选择本次 checkout 的基线；无需复制其他节点的 `.so`
或历史 `native-support.json`。2026-11-02 已在 PPU-ZW810E 上以纯源码包解包模拟全新 checkout，
完成目标环境安装和自动构建：HGGC dryrun 与 14 个编译/链接阶段返回码均为 0，实际记录了
13 份 SDK 头文件的路径与 SHA。统一 25 API smoke 含 123 个配置，其中 MTGP32 八 API 共 36 个，
50 个 accuracy/performance 阶段返回码均为 0。本次用源码包模拟全新 checkout，尚未 commit/push。
直接性能模块与直接类入口也分别自动构建基线，返回码均为 0；257 个 raw 输出的 reset
及 17+240 分段续接，输出与状态比较均一致。smoke 的 123 个配置统计异常为 0；
完整 1780 配置的五项原 SDK Poisson 统计异常、runner 返回 1 的历史记录见
[随机数算子运行与测量](ppu-rng-families.md)。自动构建不改变原统计判断。

```bash
git clone https://github.com/flagos-ai/FlagRAND.git
cd FlagRAND
export PATH=/usr/local/PPU_SDK/CUDA_SDK/bin:/usr/local/PPU_SDK/bin:/usr/local/PPU_SDK/asight/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$PWD/src:$PWD"
export FLAGRAND_VENDOR=thead
export FLAGRAND_PLATFORM=thead
export FLAGRAND_ACRAND_PATH=/usr/local/PPU_SDK/lib/libacrand.so
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 -m pip install --no-build-isolation -e '.[test,ppu]'

FLAGRAND_PERF_PROFILE=smoke python3 tools/run_tests.py \
  --ops mtgp32 --gpus 0 --dump-output --output-dir results/ppu-mtgp32-smoke
```

runner 在启动测试 worker 前构建一次 SDK 基线，输出为
`results/ppu-mtgp32-smoke/native-build/mtgp32/libflagrand_mtgp32_device.so`，
库路径由环境传递给各 API 子进程。同一阶段还用独立子进程预构建/加载 FlagRAND 的
PPU Tensor copy 扩展，构建在正式计时之外。该 Torch 扩展使用当前 Torch ABI 和 C++17，
与下述 HGGC C++14 device 基线是两个构建目标。

单独调用 `python3 -m benchmark.run_rng_delivery --platform thead` 测量 MTGP32 时，
同样在本次性能结果目录内自动构建。直接创建 `Mtgp32DeviceGenerator` 的默认构建目录是
仓库 `build/ppu-mtgp32`。

### SDK 路径与编译选项

默认调用 PATH 中的 `hgcc`，设备目标为 `ppu001`。SDK 不在默认目录时，例如：

```bash
export FLAGRAND_PPU_SDK_ROOT=/path/to/PPU_SDK
export FLAGRAND_PPU_HGCC="$FLAGRAND_PPU_SDK_ROOT/bin/hgcc"
export PATH="$FLAGRAND_PPU_SDK_ROOT/CUDA_SDK/bin:$FLAGRAND_PPU_SDK_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FLAGRAND_PPU_SDK_ROOT/CUDA_SDK/lib64:$FLAGRAND_PPU_SDK_ROOT/targets/x86_64-linux/lib:$FLAGRAND_PPU_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}"
export FLAGRAND_ACRAND_PATH="$FLAGRAND_PPU_SDK_ROOT/lib/libacrand.so"
```

`FLAGRAND_PPU_SDK_ROOT` 供 Tensor copy 扩展定位 SDK，HGGC 由 `FLAGRAND_PPU_HGCC`
选择。构建脚本也接受 `--compiler`；目标可通过 `--arch` 或 `FLAGRAND_PPU_ARCH` 指定。
正常 PPU-ZW810E 环境继续使用默认 `ppu001`。

[`build_ppu_mtgp32_baseline.py`](../../../tools/build_ppu_mtgp32_baseline.py) 使用 HGGC 自身的
`--dryrun` 分阶段命令，以 `-O3 -std=c++14 -arch=ppu001 -keep` 生成完整阶段命令，并在最终 host `g++` 链接加入 `-shared`。
HGGC 本身不接受顶层 `-shared`；host 对象与 device-link 注册对象均保留 `-fPIC`。
针对已有环境的 GCC 头文件兼容问题，只在 host splitter 阶段加入
`-D__HGGC_GCC_X86IMM_BUILTINS_H__`，device cc1 编译选项保持原样。

自动构建的 `native-build/mtgp32/compiler` 目录保存每个阶段的标准输出、错误输出及 `build.json`。
手动调用时，该目录由 `--work` 指定。JSON 包含源码与共享库
SHA256、实际执行命令、返回码，并从保留的预处理文件中收集实际使用的 SDK 头文件路径和 SHA256。
`library_version` 记录的是已安装 acRAND 组件版本；device 头文件版本以 `build.json` 的记录识别。

### 可选的已有库覆盖

需要独立 SDK 实验时，可手动构建并显式指定库和对应构建记录：

```bash
python3 tools/build_ppu_mtgp32_baseline.py \
  --source benchmark/ppu_matrix/mtgp32_device.hg \
  --output build/ppu-mtgp32/libflagrand_mtgp32_device.so \
  --work build/ppu-mtgp32/compiler
export FLAGRAND_PPU_MTGP32_DEVICE_LIBRARY="$PWD/build/ppu-mtgp32/libflagrand_mtgp32_device.so"
export FLAGRAND_PPU_MTGP32_BUILD_RECORD="$PWD/build/ppu-mtgp32/compiler/build.json"
```

`FLAGRAND_PPU_MTGP32_DEVICE_LIBRARY` 优先于自动构建；不设置时由运行入口从源码构建，
不再要求 `benchmark/ppu_matrix` 目录中预先存在共享库。`FLAGRAND_PPU_MTGP32_BUILD_RECORD`
可选，用于将显式库的对应构建记录加入 manifest。
`FLAGRAND_ACRAND_PATH` 仍用于其他生成器的 host 基线和 acRAND 组件版本查询。

## 统一测试入口

保持上述环境，从统一入口运行三种代表 shape，构建由入口完成：

```bash
FLAGRAND_PERF_PROFILE=smoke python3 tools/run_tests.py \
  --ops xorwow,mrg32k3a,mtgp32,poisson \
  --gpus 0 --dump-output --output-dir results/ppu-rng-smoke
```

完整 shape 集合使用相同入口：

```bash
FLAGRAND_PERF_PROFILE=acceptance python3 tools/run_tests.py \
  --ops xorwow,mrg32k3a,mtgp32,poisson \
  --gpus 0 --dump-output --output-dir results/ppu-rng-full
```

MTGP32 单独运行可以将 `--ops` 改为 `mtgp32`，包含该生成器的八个 API。完整集合中
raw/uniform 各有 44 个 shape，normal/lognormal 各有 42 个 shape，Poisson 为 44 个 shape ×
5 个 lambda。`poisson` 分组另外汇总四个生成器的 Poisson。`--gpus 0` 是当前可见设备中的逻辑编号。

`native-support.json` 由首次运行自动记录原生 host acRAND 的探测结果；无需历史文件。
相同设备和库版本下可通过 `FLAGRAND_PPU_NATIVE_SUPPORT` 复用已有记录。其历史 MTGP32 unsupported 记录不用于
判断新 device 库的状态；MTGP32 的编译、加载、输出与计时结果来自本次实际构建和运行。

## 精度与 Poisson 范围

已读取的 SDK MTGP32 double 重载仍只消费一个 32 位随机字。uniform FP64 返回 double，取值网格
为 `2^-32`；normal/lognormal FP64 在 32 位随机输入上执行双精度数学转换。因此结果元数据记录
`fp64_random_input_bits=32`，不能把 double 返回类型等同于 53 个随机位。FlagRAND 自身的 FP64
随机精度要求继续保持，性能对比同时注明这一差异。

SDK 的 `hgrand_poisson(MTGP32)` 在 lambda<1000 时使用逆变换，在 lambda≥1000 时使用
`uint32(sqrt(lambda) * normal_icdf_double(raw32) + lambda + 0.5)`，即取整正态近似。
本基线保留 SDK 公式，未替换为其他采样算法。大样本下应同时查看均值、方差和 Poisson CDF 比较，
不能由重复性或均值正常推断近似分布等同于 Poisson。已有 device 探针观察到大样本 CDF 偏差；
新封装的各 shape 仍需独立记录实际统计结果。

统一矩阵保留统计异常点的状态和原始耗时；只有候选与基线均符合既定正确性要求的点才生成
有效加速比。不能通过删除 lambda=1000、大 shape 或修改阈值来填补完整集合。该封装已取得上述
56 点设备结果；后续 84 点缓存接入对照再次记录了原 SDK 在 lambda=1000、N=16Mi 的 CDF 偏差。
它们均不是最终完整 shape 矩阵。

独立诊断参考使用同一 SDK raw32 递推和 `_hgrand_poisson_ITR_double`，标识为
`ppu_sdk_mtgp32_itr_reference`，单独保存构建和计时记录。六个 lambda 的 16Mi 统计检查已有结果；
该参考用于比较逆变换路径，不能覆盖原样 SDK 库或以原样 SDK 名义报告其性能。

## 结果文件

以下 `native-build` 路径位于本次 `tools/run_tests.py --output-dir` 下；直接调用性能模块时
位于它创建的 `<run>` 目录下。

| 文件 | 内容 |
|---|---|
| `native-build/mtgp32/libflagrand_mtgp32_device.so` | 从当前仓库源码构建的 SDK device+host 基线 |
| `native-build/mtgp32/build.log` | 完整构建输出 |
| `native-build/mtgp32/compiler/build.json` | 编译阶段、返回码、库/源码/实际 SDK 头文件 SHA256 |
| `native-build/mtgp32/compiler/stage-*.stdout` / `stage-*.stderr` | 各阶段输出 |
| `native-build/mtgp32/tensor-copy.log` | 统一 runner 预构建/加载 Tensor copy 扩展的输出 |

manifest 同时保存本次所选 SDK 基线的构建记录；显式覆盖时使用所提供且与库路径对应的记录。
这些信息保留在原始产物中，公开 summary 格式不变。

公开 `summary.json` 沿用 Philox 格式，不新增字段。根字段为 `timestamp`、`total_duration`、`env`、
`result`、`profile`；每个交付项保留 `customized`、`accuracy`、`performance`。性能点为
`base`、`gems`、`speedup`，dtype 内平均值仍为算术平均。`base` 取所选基线的完整调用时间，
`gems` 取 FlagRAND 的完整调用时间，`speedup=base/gems`。

原始 `records.jsonl` 中，MTGP32 基线标识为 `ppu_sdk_mtgp32_device`，并记录 `baseline_metadata`，
其中包括 200 个状态、每块 256 线程、参数组合的状态范围、尾部策略、FP64 随机输入位数、
Poisson 近似范围和动态库 SHA256。候选的配对字段为 `speedup_vs_ppu_sdk_mtgp32_device`。
`manifest.json` 的 `acceptance_baselines` 逐算子记录基线身份；包含多类基线时，
`acceptance_baseline` 为 `per_operator`。构建目录中的 `build.json` 一并保存。

主要指标仍为 `total_ms_median`，计时涵盖持久生成器、预分配输出上的完整公开调用与末尾同步。
初始化、首次编译和首次分配在 warmup 中完成。Event 间隔与 Asys kernel duration 分开解释。
`performance_statistics.json` 保持仓库现有独立统计格式；统一流程及文件布局见
[随机数算子运行与测量](ppu-rng-families.md)。
