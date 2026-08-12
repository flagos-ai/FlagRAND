# Ascend 910C Normal/LogNormal FP32 Host Fast Path 报告

## 1. 结论

910C 上 `normal_f32` 的不稳定未发生在随机数 kernel、Triton cache 或 NPU 设备执行阶段，而发生在
Python public API 到 cached launcher 之间的 host dispatch。v6 将通过校验的 NPU FP32
Normal/LogNormal 常见路径直接连接到 cached launcher，同时保留 counter epoch 边界的完整回退。

实机定向验收结果如下：

| 项目 | 结果 |
|---|---:|
| pytest 正确性 | 13 passed |
| `normal_f32` API speedup，第 1 轮 | 1.175x |
| `normal_f32` API speedup，第 2 轮 | 1.131x |
| `normal_f32` API speedup，第 3 轮 | 1.130x |
| `lognormal_f32` API speedup | 1.315x |
| 性能记录失败数 | 0 |

三轮 `normal_f32` 全部高于 `0.95x` 门禁，最低值仍有 `0.180x` 的绝对余量。该结果取代 v5
稳定性复测中的 `0.865x / 0.871x / 0.970x`，不再依赖第三轮偶然过线。

## 2. 问题表现

v5 的 Normal kernel 在 `N≈1M` 时已经显著快于 ACLNN，但小、中 shape 的 public API 总时间约为
`49–50 us`，导致 shape 平均后的 API speedup 在不同轮次落到 `0.865x`、`0.871x` 和 `0.970x`。
同一份二进制在不同轮次的 kernel 数学、block 和输出均未变化，因此首先需要区分三类可能性：

1. Triton 重新编译或 cached launcher 失效；
2. NPU kernel 或 stream 执行时间波动；
3. Python public API、通用分派和 launcher 包装层的 host 下发波动。

## 3. 定位方法与证据

### 3.1 Cached launch 计数

连续 1000 次调用的五轮探针中，每轮均满足：

- `direct_launches=1000`；
- `compile_or_jit_launches=0`；
- launcher cache 只有一个有效 entry；
- `requires_regular_jit=false`。

因此首次编译、cache miss 和 regular JIT 不是本次波动来源。

### 3.2 Host 分层测量

在相同 output、generator 状态和同步口径下逐层绕过包装代码，代表性结果为：

| 层级 | enqueue 中位数 | total 中位数 |
|---|---:|---:|
| compiled launch，固定 stream | 10.25 us | 41.36 us |
| cached launcher hot match | 23.13 us | 41.33 us |
| chunk dispatch | 28.62 us | 41.36 us |
| Huawei distribution dispatch | 35.23–42.99 us | 41.39–43.07 us |
| public cuRAND API | 42.88–50.59 us | 42.94–50.65 us |
| CANN ACLNN baseline | 24.18–27.05 us | 103.16–103.23 us |

direct、launcher 和 chunk 三层的 total 基本固定在 `41.3 us`，说明设备端执行稳定；显著变化只出现在
Huawei 通用分派与 public API 两层。Fused public fast-path 探针进一步显示，在不改 kernel 的前提下可
节省约 `6.8–12.7 us` enqueue 开销。这足以解释 v5 的全部不稳定差额。

### 3.3 为什么不继续做 kernel profiling

此处 direct launch 的 total 已稳定，且原路径和 fast path 使用同一个已编译 kernel。继续采集
AI Core 指令、UB 或 vector pipeline 只能解释共同的约 `41 us` 设备底座，无法解释 public API 额外的
host 时间。因此本轮使用 launcher 计数和分层端到端探针定位，比继续下钻设备指令更直接。

## 4. 实现方法

### 4.1 Public API 单次校验

`generate_normal` 与 `generate_lognormal` 保留原 dtype 错误和未处理路径语义，但成功路径不再额外经过
通用 `_require_handled` 调用。NPU float32 分支一次完成：

- Philox generator 类型与 RNG type 校验；
- NPU contiguous output 校验；
- `block_size`、`num_warps` option 白名单和取值校验；
- lazy-import 后的 910C fast launcher 调用。

Ascend 模块仍只会在 `out.device.type == "npu"` 时加载，不进入 PPU/CUDA import 或执行路径。

### 4.2 Common-case direct cached launch

新增的 `generate_huawei910c_normal_f32_fast` 只处理已经由 public dispatcher 验证的调用。它直接读取
seed 和 offset，计算低 32 位 counter、epoch key、signed-i32 launch 参数和固定 block，然后调用现有
`_NORMAL_LAUNCHER`。Normal 与 LogNormal 仍使用相同 kernel，只通过 constexpr `LOGNORMAL` 区分。

该优化没有更换 hash、Normal 近似多项式、`exp`、block 或 cached compiled binary，因此输出数学路径
保持不变。

### 4.3 Counter epoch 安全回退

当一次调用跨越 `2^32` counter 边界时，fast path 不尝试展开第二个 epoch，而是回退到原
`generate_huawei910c` windowed 实现。新增测试从 `(1 << 32) - 8` 生成 18 个值，分别验证 Normal 与
LogNormal 的 replay 逐位一致和最终 offset。这样常见路径获得最短 host 链，极端大 offset 仍保持完整语义。

## 5. 验证结果

### 5.1 正确性

910C 定向 pytest 覆盖 replay、bulk/split、offset、统计特征和 counter epoch 边界，结果为
`13 passed`。出现的一条 `allow_internel_format=False` 信息来自 `torch.empty_like` 的 torch_npu base-format
回退，是既有 warning，不是 CPU 算子 fallback、精度错误或失败。

### 5.2 性能稳定性

优化后三轮 `normal_f32` 均通过，API 均值为 `1.175x / 1.131x / 1.130x`。小、中 shape 的 FlagRAND
端到端时间主要落在约 `36–37 us`，`N≈1M` 约为 `41–43 us`。`lognormal_f32` 使用同一 host fast path，
完整 acceptance shape 的 API 均值为 `1.315x`。

### 5.3 双平台终验

v6 包已完成 910C 七 API 逐进程 acceptance 和 PPU 七 API acceptance。910C `normal_f32` 的正式
API speedup 为 `1.042x`，`lognormal_f32` 为 `1.451x`；PPU 七 API 同样全部通过，证明共享 public
dispatcher 的改动没有破坏 PPU 路径。两端 preflight、pytest 和 performance 返回码均为 0。
