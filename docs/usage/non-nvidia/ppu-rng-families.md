# PPU-ZW810E 随机数算子运行与测量

本文覆盖 XORWOW、MRG32K3A、MTGP32 的 raw32、uniform、normal、lognormal，以及
Philox 和这三个生成器的 Poisson，共 25 个 API。uniform、normal、lognormal 分别覆盖
FP32 和 FP64。实现接口、算子注册和测试入口沿用仓库现有结构。

2026-11-01：当前本地组合实现完成同版 25 API、1780 个配置，生成 3560 条原始记录。
逐 API 在原 shape/λ 有效配对上的算术平均均高于 0.95。263 项独立正确性比较差异为 0，
44 项沿用原跳过条件；1780 个 FlagRAND 输出满足原统计条件。原 SDK 的五项 Poisson
统计异常仍保留，有效速度配对为 1775。完整 runner 返回 1，对应四个 Poisson performance
阶段；其余 46 个阶段返回 0。运行时长 51 分 03 秒。

| MTGP32 API | 有效配对 | 算术平均加速比 |
|---|---:|---:|
| raw32 | 44 | 1.185289123× |
| uniform_f32 | 44 | 1.069422464× |
| uniform_f64 | 44 | 1.013041542× |
| normal_f32 | 42 | 0.955373760× |
| normal_f64 | 42 | 1.088344197× |
| lognormal_f32 | 42 | 0.970356347× |
| lognormal_f64 | 42 | 1.097701566× |
| poisson | 218 | 2.274494301× |

XORWOW 七个普通 API 均值为 1.273734～1.800004，MRG32K3A 为 1.125247～1.703976。
Philox、XORWOW、MRG32K3A、MTGP32 的 Poisson 均值依次为 16.309837、5.065898、4.502046、2.274494。
全部 1775 个有效点中，1166 点低于 0.95；上述结果是每个 API 的平均值。
当前实现包含原生缓存命中路径、uniform FP64 融合尾部、FP32 normal/lognormal 派生缓存
及公开入口局部赋值后移。129 个源码条目在 25 份实际 manifest 中共 3225 项 SHA 一致。
本轮代码在本地工作树，未推送 GitHub。

2026-10-28：MTGP32 的 PPU SDK device 函数加 host 封装已接入统一测试代码，
基线标识为 `ppu_sdk_mtgp32_device`。该封装已完成 14 阶段设备构建、11 项状态续接检查、
11 项 1Mi 统计检查和 56 个代表性能配对。后续缓存公共入口的 84 点比较也已完成；
组合版本随后完成包含边界 shape 的 25 API 完整矩阵，共 1780 个配对配置、3560 条原始记录。
独立正确性阶段执行 263 项比较、差异为 0，另 44 项按原条件跳过。原生基线有 5 条统计异常，
因此有效速度配对为 1775。XORWOW、MRG32K3A 的七个普通 API 算术平均均超过 0.95；
这一历史版本中 MTGP32 的七个普通 API 为 0.613604–0.810714。
构建方式、状态语义及 SDK 精度限制见 [MTGP32 SDK 基线](ppu-mtgp32-baseline.md)。

2026-10-29 的历史组合版本包含 raw 尾部直写、uniform FP32 转换直写、对齐 normal
raw 预取，以及 PPU 对数和 FP64 exp 函数，完成 MTGP32
全部八个 API：520 个配置点、1040 条原始记录。82 项独立正确性比较差异为 0，
另 44 项沿用原跳过条件。520 个 FlagRAND 输出均满足原统计条件；SDK 在 λ=1000、
N=4194304 和 16777216 的两条 CDF 偏差继续保留，有效速度配对为 518。

| MTGP32 API | 有效配对 | 算术平均加速比 |
|---|---:|---:|
| raw32 | 44 | 0.856777× |
| uniform FP32 | 44 | 0.751802× |
| uniform FP64 | 44 | 0.676750× |
| normal FP32 | 42 | 0.658243× |
| normal FP64 | 42 | 0.805510× |
| lognormal FP32 | 42 | 0.664481× |
| lognormal FP64 | 42 | 0.797466× |
| Poisson | 218 | 2.220148× |

七个普通 API 的均值仍低于 0.95；Poisson 的 218 个有效点中有 147 个低于 0.95。
这里保留全部 shape 和参数，不用整体均值替代逐点结果。六个组合源码在八个 API 的实际
manifest 中共 48 项 SHA 一致。原 SDK 的两条统计异常使 Poisson performance 和完整 runner
返回 1；其余 15 个阶段返回 0，外层脚本返回 0 不代表完整 runner 返回 0。
设备首尾快照随结果保存，不能据端点推断整段独占。

此前大 raw 非对齐路径改为直接写用户输出，只保存未消费尾部。该单因素公开入口的 1Mi−1
两序中位数比值为 1.253706/1.189607，1Mi+1 为 1.150848/1.143195，均以同轮旧实现
时间除以新实现时间计算。全 44 个 raw32 shape 的独立 SDK 复测算术平均为 0.832919×，
其中 42 点低于 0.95；这一轮与上表的运行时刻不同，不能由跨轮均值变化推导改动的因果效果。
上表使用随后当前组合版本的完整测量，未将这项内部比值乘入任何结果。

## 本轮设备与环境

2026-10-24 实际运行环境为 PPU-ZW810E，单卡显存 97920 MiB、64 个计算单元，
驱动 `1.3.2-d7f5a2`，PPU-SMI 报告 HGGC 13.0。软件为 Python 3.12.3、
Torch 2.9.0、Triton 3.6.0、FlagTree `0.6.0+ppu.gitd96f5339`，acRAND 版本 1400。
这套环境的 Triton target 报告 `cuda / arch=80 / warp_size=32`，实际使用 PPU 编译器，
不能由这个兼容接口名称推断成 NVIDIA 硬件。

使用节点或厂商镜像已有的 Torch、FlagTree、完整 PPU SDK 编译工具和宿主 C++ 编译器。
clone 包含此改动的仓库版本后，从仓库根目录安装并设置运行环境：

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
```

这些命令描述已运行的 SDK 环境；它们不安装或替换系统硬件依赖。现有 Philox CI 的
历史依赖与运行方式见 [PPU 指南](ppu.md)。

仓库自带 MTGP32 SDK device+host 基线源码和构建脚本。`tools/run_tests.py` 在选择 PPU
MTGP32 API 时，先从本次 checkout 编译基线，再把库路径传给各测试子进程；无需预先提供
其他机器上的 `.so`、补全代码或 `native-support.json`。2026-11-02 已在 PPU-ZW810E 上用
纯源码包解包模拟全新 checkout，完成安装、自动构建和 25 API smoke：123 个配置、246 条原始记录，
263 项准确性比较差异为 0、44 项按原条件跳过，50 个阶段及统一 runner 返回码均为 0。
本次变更尚未 commit/push。另行运行直接性能模块和直接 `Mtgp32DeviceGenerator` 类入口，
均自动构建基线并返回 0；257 个 raw 输出的 reset 及 17+240 分段续接，输出与状态比较均一致。
smoke 的 123 个配置统计异常为 0；上文完整 1780 配置的五项原 SDK Poisson 统计异常及
runner 返回 1 的记录保持原样。自动构建不改变原统计判断。

SDK 位于其他目录时，同时设置工具搜索路径、动态库搜索路径和编译器。例如：

```bash
export FLAGRAND_PPU_SDK_ROOT=/path/to/PPU_SDK
export FLAGRAND_PPU_HGCC="$FLAGRAND_PPU_SDK_ROOT/bin/hgcc"
export PATH="$FLAGRAND_PPU_SDK_ROOT/CUDA_SDK/bin:$FLAGRAND_PPU_SDK_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FLAGRAND_PPU_SDK_ROOT/CUDA_SDK/lib64:$FLAGRAND_PPU_SDK_ROOT/targets/x86_64-linux/lib:$FLAGRAND_PPU_SDK_ROOT/lib:${LD_LIBRARY_PATH:-}"
export FLAGRAND_ACRAND_PATH="$FLAGRAND_PPU_SDK_ROOT/lib/libacrand.so"
```

`FLAGRAND_PPU_SDK_ROOT` 指定 Tensor copy 扩展使用的 SDK；`FLAGRAND_PPU_HGCC` 指定
基线编译器，默认从 PATH 查找 `hgcc`。基线目标默认 `ppu001`，可通过
`FLAGRAND_PPU_ARCH` 设置对应 SDK 支持的目标。

### PPU Tensor copy 扩展

PPU 的 MTGP32 缓冲复制使用随包分发的 `runtime/csrc/ppu_tensor_copy.cpp`。
统一 runner 在启动测试 worker 前，用独立子进程提前构建/加载扩展，日志保存在
`<output-dir>/native-build/mtgp32/tensor-copy.log`。需要单独预构建时可运行：

```bash
python3 -m flagrand.runtime.ppu_tensor_copy
```

`ppu` 可选依赖包含 Ninja 和 setuptools。构建通过当前 Torch 的 C++ extension 工具、
宿主 C++17 编译器及实际 `_GLIBCXX_USE_CXX11_ABI` 值完成，不使用其他机器的二进制。
SDK 默认仍是 `/usr/local/PPU_SDK`；非标准安装目录可设置
`FLAGRAND_PPU_SDK_ROOT`。需要自选编译器或缓存位置时，使用 Torch 原有的 `CXX` 和
`TORCH_EXTENSIONS_DIR`。这部分没有设备 kernel，不依赖 FlagGems CMake 工程。

首次创建 PPU buffer 时也会延迟构建/加载；上述 runner 预构建位于正式计时之外，
不要求用户先执行独立预构建命令。扩展接收当前 Tensor 和同一个 Stream，动态读取输出指针，
执行 SDK D2D copy 并调用 Tensor `record_stream`。小请求缓存命中时，原生扩展核对
当前 raw cache、分布参数、seed、offset 和实际当前 Stream，再复制已生成的结果；
缓存未命中时继续执行 Triton 递推及分布转换。内部源引用和跨 stream 等待保持。其他 backend 不因导入 FlagRAND 而构建该扩展。
wheel 与源码包通过 package-data 保留 C++ 源文件，安装后仍在目标 Torch 环境构建。

### MTGP32 SDK 基线自动构建

默认每次 `tools/run_tests.py` 运行只构建一次 MTGP32 基线，输出为
`<output-dir>/native-build/mtgp32/libflagrand_mtgp32_device.so`。该目录的 `build.log` 和
`compiler/build.json` 保存实际构建过程；各 API 的 manifest 记录对应构建来源。
单独使用 `python3 -m benchmark.run_rng_delivery --platform thead` 运行 MTGP32 时，
也会在本次性能输出目录中自动编译。直接使用 `Mtgp32DeviceGenerator` 时，默认构建到
仓库的 `build/ppu-mtgp32`。

SDK 实验需要指定已有库时，仍可手动构建并覆盖默认选择：

```bash
python3 tools/build_ppu_mtgp32_baseline.py \
  --source benchmark/ppu_matrix/mtgp32_device.hg \
  --output build/ppu-mtgp32/libflagrand_mtgp32_device.so \
  --work build/ppu-mtgp32/compiler
export FLAGRAND_PPU_MTGP32_DEVICE_LIBRARY="$PWD/build/ppu-mtgp32/libflagrand_mtgp32_device.so"
export FLAGRAND_PPU_MTGP32_BUILD_RECORD="$PWD/build/ppu-mtgp32/compiler/build.json"
```

显式 `FLAGRAND_PPU_MTGP32_DEVICE_LIBRARY` 优先于自动构建；上述两个变量均为可选。
对应 `build.json` 保存分阶段编译命令、返回码、库文件与实际 SDK 头文件的 SHA256。
完整构建说明见 [MTGP32 SDK 基线](ppu-mtgp32-baseline.md)。

## 统一测试入口

三种 shape 的初测：

```bash
FLAGRAND_PERF_PROFILE=smoke python3 tools/run_tests.py \
  --ops xorwow,mrg32k3a,mtgp32,poisson \
  --gpus 0 --dump-output --output-dir results/ppu-rng-smoke
```

完整 shape 集合：

```bash
FLAGRAND_PERF_PROFILE=acceptance python3 tools/run_tests.py \
  --ops xorwow,mrg32k3a,mtgp32,poisson \
  --gpus 0 --dump-output --output-dir results/ppu-rng-full
```

`--gpus 0` 使用当前可见设备中的逻辑编号 0。同一卡上的性能测量应串行执行。
完整集合为 raw/uniform 各 44 个 shape，normal/lognormal 各 42 个 shape；每个生成器的
Poisson 为 44 个 shape × 5 个 lambda（0.1、1、10、30、1000）。初测只有
1024、65536、1048576 三个 shape，不能替代完整集合。

单个 API 的性能迭代可以使用：

```bash
python3 -m benchmark.run_rng_delivery --platform thead --profile matrix \
  --distributions mrg32k3a_raw32 --results-dir results/ppu-mrg-raw
```

MTGP32 八个 API（包含 `mtgp32_poisson`）使用上述 SDK device 基线。Philox、XORWOW 和
MRG32K3A 使用 `ppu_acrand`；这些 host API 的可用性首次运行时由独立进程记录，结果写入 `native-support.json`，
因此新 checkout 不需要历史探测文件。相同设备和库版本下，可通过 `FLAGRAND_PPU_NATIVE_SUPPORT=/path/to/native-support.json`
复用该记录。更换原生库后应重新生成记录。该记录描述 host acRAND，不表示新建的 MTGP32
device 基线已经编译或运行。

## 结果格式与性能口径

| 文件 | 内容 |
|---|---|
| `summary.json` | 同仓库 Philox 的公开格式；逐 shape 时间与 dtype 内算术平均加速比 |
| `api_summary.json` | 各 API 的正确性、性能执行结果、返回码和原始输出路径 |
| `performance_statistics.json` | 逐 API 几何平均、有效配对数、完整性和 0.95× 目标结果 |
| `<API>/performance_raw/<run>/records.jsonl` | 每个 shape 的正确性统计、全部计时样本、基线标识与元数据 |
| `<API>/performance_raw/<run>/manifest.json` | 软件环境、shape、lambda、源码 SHA、逐算子基线和测量设置 |
| `native-build/mtgp32/libflagrand_mtgp32_device.so` | 本次 runner 从仓库源码构建的 SDK device+host 基线 |
| `native-build/mtgp32/build.log` | 基线构建输出 |
| `native-build/mtgp32/compiler/build.json` | 编译命令、阶段返回码、库和实际 SDK 头文件 SHA256 |
| `native-build/mtgp32/tensor-copy.log` | runner 在测试前构建/加载 Tensor copy 扩展的输出 |

主要计时为 `total_ms_median`：持久生成器和预分配输出上的完整公开调用，包含主机下发、
设备执行和末尾同步。首次编译、初始化、首次表构建及首次 workspace 分配在 warmup 中完成。
每个 shape 的加速比为同设备、同生成器、同 API 的所选基线时间除以 FlagRAND 时间；
MTGP32 为 SDK device 函数加 host 封装的完整调用，其他生成器为原生 host acRAND。
本说明的逐 API 结果在全部有效 shape 配对上取算术平均，Poisson 同时覆盖全部 lambda。
Event 间隔另行保留，它可能包含下发空隙，不能直接称为纯 kernel 时间。

公开 `summary.json` 格式保持不变：根字段为 `timestamp`、`total_duration`、`env`、`result`、
`profile`；交付项保留 `customized`、`accuracy`、`performance`。性能点仍使用 `base`、`gems`、
`speedup`，dtype 内的 `speedup` 仍为算术平均。基线身份和 SDK 限制保存在原始记录、manifest
及构建记录中，不向公开 summary 添加字段。独立 `performance_statistics.json` 继续保留现有统计字段。

历史 acRAND 1400 的八个 MTGP32 host API 报告 `Unimplemented API`，该事实仍只对应 host 接口。
新封装直接调用 SDK device 函数，不调用 `acrandGenerate*`，其性能不能标成原生 acRAND host 性能。
SDK 的 MTGP32 FP64 分布使用 32 位随机输入；Poisson 在 lambda≥1000 时使用取整正态近似。
因此仍按既定统计要求记录每个点，不以 double 返回类型推断 53 个随机位，也不将近似分布视为精确 Poisson。
新封装已完成代表点及组合版本的完整 shape 计时。后续源码变化不沿用此前均值，
原样 SDK 与强制 ITR 参考的库和结果分别保存。

Philox、XORWOW 和 MRG32K3A Poisson 在 N=16777216、lambda=1000 的原生输出存在统计偏差；
MTGP32 原样 SDK 在 lambda=1000、N=4194304 和 16777216 的 CDF 也偏离原条件。
这些记录及原始耗时保留，不纳入有效配对的速度比。`summary.json` 中测试执行状态与
`performance_statistics.json` 中性能目标分别查看；有效配对的平均值不等于完整集合达标。
当某个 API 有缺失或未通过的原生配对时，`api_complete=false`、`api_meets_target=null`；
其有效配对的几何平均仍然保留。主 `summary.json` 的公开字段不因此扩展。

## 编译产物与设备时间

PPU 内核保留 Triton IR、LLVM IR、目标二进制及 `hgobjdump` 输出，结合 Asight Systems
中的 kernel duration 判断设备改动。Asight Compute 历史记录包含 PCM 计数器被占用，
10 月 28 日的采集返回 `DeviceNotReady`；当前没有本轮动态 occupancy 或 stall 计数，
因此不从静态寄存器数量推导这些动态指标。

MTGP32 的 PPU FP32 normal/lognormal 使用 `fast_logf`，FP64 使用原生 `log`，
FP64 lognormal 另使用原生 `exp`。FP64 保留 53 位随机输入，sqrt 和 sincos 沿用原实现。
三项数学选择通过独立选项和具名 JIT helper 接入 buffer、cached 和 uncached 路径。
工厂函数若仅把同一个局部名称绑定到两个
builtin，当前 Triton 的依赖 hash 会忽略 builtin 身份，可能复用先编译的数学实现。
具名 JIT helper 将不同实现纳入编译依赖；设备上的两种首次编译顺序、公开路径、分块续流
和状态比较均已执行。原生 exp 接入后的 119 组数值比较满足原 `rtol=1e-12, atol=1e-12`
条件；normal_f64 新旧输出逐位相同，lognormal_f64 新旧最大差异为 1 ULP。
实际公开路径的 1Mi/16Mi 转换 kernel 时间约从 34.45/522.21μs 降到 27.75/415.18μs。
对应 PASM 的 VREG 从 48 降到 40、SREG 从 96 降到 80，FP64 FMA 从 156 条降到 136 条；
条件分支从 6 条增到 10 条。这些是本地数学改动的内部对照，新组合的 SDK 全 shape 均值另行记录。

raw 尾部路径保留原预取公式、192 个状态和 256-word 协作步长；只在原策略的未消费
尾部不超过 256 word、且仍使用 global-state 递推的范围内直写输出。指针选择后的一次
store 在 PASM 中消除了双条件 store 的执行掩码切换；其单独内核对照约快 5.4%。
完整公开调用收益另外测量，不能把这项内部 kernel 比值解释为 SDK 加速比。缓存只保留
未消费区间后，backing tensor 长度可改变，逻辑 offset、物理 frontier、全部 state 和
未消费 raw 的比较保持原约定。

uniform_f32 的相应非对齐路径在递推 kernel 中直接写出转换结果，并将未消费的 word
保存为 raw 尾部；前缀最多 256 word，在同一个 kernel 转换。原本的全量 raw 写入、读取
和独立转换被合并，递推、参数表和轮间 barrier 保持原实现。同轮 1Mi±1 的 kernel
总时间约从 9.9μs 降到 7.3μs；实际公开调用的负样本和完整 44-shape SDK 结果仍保留。

FP32 normal/lognormal 对没有 live cache、raw 消费长度为 256 的整数倍的中等请求，
在现有容量能容纳至少两次请求时使用原有 raw 预取。默认容量为 1Mi raw word，派生输出
缓存阈值仍为 65536；这项入口变化不修改分布数学和预取容量。262144 输出的连续 64 次
调用由 64 次 raw 生成变为 16 次 refill，相邻原有预取路径保持原来的 refill 相位。
设备耗时按完整调用区间的 kernel 总和比较，不把错位的 refill/hit 逐次比值作算术平均。
本轮保留 FP64 控制点的完整调用回退记录，组合效果由同版本完整八 API 矩阵表示。

已执行的设备实验包括 MRG32K3A 的 uint32 递推状态与小 shape 的 compact 状态访问、
MTGP32 的共享状态、同步安排，以及相邻 word 交换的 PPU warp shuffle，
以及 XORWOW Poisson 的 state 写回布局。编译器、设备时间、完整公开 API 时间和原生比较
分别留存；研究中的同源码前后比值不替代对应正式基线的比值。
