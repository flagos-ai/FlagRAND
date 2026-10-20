# 第一节 加速比统计

测试设备为海光 **BW1000**，运行环境为 DTK 26.04、PyTorch `2.10.0`、
Triton `3.6.0`（FlagTree `0.6.0+hcu.git46341ffa`），Triton target 为
`hip / arch=gfx936 / warp_size=64`，HIP 版本为 `6.3.26113`。
性能 baseline 为海光环境的原生 **hipRAND**，动态库路径为
`/opt/dtk/lib/libhiprand.so`，`hiprandGetVersion` 返回值为 `300200`。

计时使用持久 generator 和预分配输出张量，generator 状态在连续调用间推进，
编译预热不计时。每个 shape 实际预热 **16 次**，正式重复 **11 组**，
单组目标计时预算 **200 ms**。每组调用次数由 pilot 估算，最多 1000 次；
本次实际为 450–1000 次，因此 200 ms 是目标预算，不是每组必须达到的时长。

正式指标为 `total_ms_median`，覆盖 Python host API、HIP 提交、设备执行及该组末尾同步，
再折算单次调用时间。加速比计算式为：

`hipRAND total_ms_median / FlagRAND total_ms_median`

大于 1.0× 表示 FlagRAND 更快。API 平均加速比是该 API 全部实测 shape 加速比的
算术平均，不计算跨 API 总平均。相邻 shape 交替安排 FlagRAND 与 hipRAND 的测量顺序。
矩阵与 NVIDIA 的 shape 集合一致，共 300 个 API/shape 组合，最大 N=16,777,216。

| API | dtype | shape 数 | API 平均加速比 | 最小加速比 | 最大加速比 |
|---|---|---:|---:|---:|---:|
| raw32 | int32 | 44 | 1.072× | 0.471× | 2.569× |
| uniform_f32 | float32 | 44 | 1.517× | 0.588× | 2.575× |
| uniform_f64 | float64 | 44 | 1.290× | 0.887× | 3.031× |
| normal_f32 | float32 | 42 | 1.138× | 0.857× | 3.227× |
| normal_f64 | float64 | 42 | 1.008× | 0.623× | 1.204× |
| lognormal_f32 | float32 | 42 | 0.984× | 0.941× | 1.123× |
| lognormal_f64 | float64 | 42 | 1.001× | 0.648× | 1.196× |

本项目沿用已确定的验收口径：每个 API 平均加速比 **≥0.95×，不设上限**。
本次 **7/7 API 达标**。如果仅按示例的 0.95×–1.50× 区间统计，则为 **6/7**；
`uniform_f32=1.517×` 超过区间上沿，表示速度更快，不改变本项目的达标结论。

上述结论按 API 平均值判断。个别输入仍低于原生库：raw32 的 N=1,048,575 为
0.471×；normal_f32 的 N=16,777,216 为 0.857×。
normal_f32 有 36/42 个 shape 低于 1.0×，其平均值受到少数较大加速 shape 的影响；
所有慢点均保留在原始记录中。

# 第二节 正确性统计

正确性策略先明确 FlagRAND 的 portable Philox 序列约定，再分别执行已知答案、独立 CPU 参考、
重放与连续性、边界、stream、binary64 参数和分布统计校验。
FlagRAND 与 hipRAND 各自验证自身序列；本次不要求两者生成的随机数逐元素相等。
详细判据见 [海光正确性验证策略](./Hygon_BW1000_FlagRAND_正确性验证策略.md)。

| 校验范围 | 数据量 | 结果 |
|---|---:|---|
| FlagRAND preflight | 7 组、35 个探针 | 35 passed，0 failed |
| FlagRAND acceptance cases | 300 个 | 300 passed，0 failed |
| hipRAND baseline cases | 300 个 | 300 passed，0 failed |
| acceptance 全部记录 | 600 条 | 600 passed，0 failed |
| 矩阵准确性 pytest，按测试 ID 去重 | 119 个 | 111 passed，8 skipped，0 failed |
| 前置海光/API/后端协议测试 | 27 个 | 27 passed，0 failed |
| 统一 runner 测试 | 14 个 | 14 passed，0 failed |
| 七 API 的 accuracy/performance 阶段 | 14 个 | 14 passed，0 failed |

8 个 skipped 均为 CoreX 专用测试，在海光上不适用。
共享 pytest 用例按测试 ID 去重，矩阵 case、preflight 探针和前置测试分别统计。

Preflight 覆盖七 API 的 offset=0、1、2、3、2^34−4。
专项准确性测试通过了 seed=0 的 raw32 golden vector、七 API 的整段与分段连续生成一致性、
大 seed、非对齐输出、counter 高位进位，以及 FP64 参数和 exp 精度校验。
stream 测试覆盖 FlagRAND uniform 和七 API 的原生 hipRAND 非默认 stream 重放。

每个 acceptance case 都检查同 seed 两次生成的整段输出完全一致；
FlagRAND 另对前 min(N,128) 个值进行独立 CPU 参考比较。
raw32 要求完全相等；FP32 使用 `rtol=2e-5, atol=2e-6`；
FP64 使用 `rtol=1e-12, atol=1e-13`。数值参考抽样与整段重放检查分开记录。

分布统计取前 min(N,2^20) 个值：uniform 检查有限值、(0,1] 范围、均值和方差；
normal 检查有限值、均值和标准差；lognormal 检查有限值、正值以及取对数后的均值和标准差。
矩阵在样本量达到 4096 后进行矩检查，另有 N=262144 的专项分布测试。

# 第三节 实现方法

| API / 组件 | 实现方法 |
|---|---|
| raw32 | Triton kernel 内执行 Philox4x32-10。4-word 对齐路径每个 counter 写出 4 个 int32；非对齐 offset 或 shape 使用四 word 滑动窗口与 mask。默认 BLOCK=256、num_warps=4。 |
| uniform_f32 | 将 Philox 生成与 uint32 到 (0,1] 的 FP32 变换融合；对齐路径每个 counter 生成 4 个输出，非对齐路径使用滑动窗口。默认 256/4。 |
| uniform_f64 | 每个 double 消费两个 uint32，以显式的 53-bit 组合映射生成均匀数；每个 counter 生成两个 double，非对齐路径按 word offset 取窗口。默认 256/4。 |
| normal_f32 | Philox 与两组 Box–Muller 变换融合在一个内核中；使用 portable FP32 数学路径。海光默认 128/2，适配 64-thread wave 及实测配置差异。 |
| normal_f64 | 每两个 uniform_f64 经 Box–Muller 生成两个 double。HIP libdevice 提供 double log/sqrt，三角函数采用区间约减后的 double 多项式。标准分布使用编译期分支；非标准 mean/stddev 按 IEEE-754 binary64 bits 传入。默认 256/4。 |
| lognormal_f32 | 复用 FP32 Philox + Box–Muller 融合路径，在同一内核内执行 exp。默认 128/2。 |
| lognormal_f64 | 复用 FP64 normal 路径，在同一内核内用 HIP libdevice 执行 double exp，保留 mean/stddev 的 binary64 精度。默认 256/4。 |
| public API 与 kernel launch | 参照 NVIDIA 后端采用直接 public API 分发；缓存已编译 Triton specialization，key 区分设备、指针对齐、seed、counter 类型/对齐、shape、SHIFT、参数和分块。每次调用获取当前 stream；现代 Triton 避免重复求值旧版路径的环境检查。 |
| hipRAND baseline | Python 通过 ctypes 调用 hiprandCreateGenerator、seed/offset/stream 设置及七类生成函数；输出直接写入 Torch 的 HIP 张量。使用真实 hipRAND 原生库。 |
| benchmark 与交付 | 两侧均复用 generator、预分配输出并绑定当前 stream；正式矩阵使用相同计时器。统一 runner 产出同格式 summary.json，包含 PHILOX4_32_10、RAW_BITS、UNIFORM、NORMAL、LOGNORMAL 五组及逐 shape 的 base/gems/speedup；其中 gems 字段保存 FlagRAND 时间，时间单位为 ms。 |

编译证据显示，旧 FP64 normal 的 128/4 配置产生一次布局转换，使用 2048 B 的
launch-time LDS，并出现 `ds_write → s_barrier → ds_read`；256/4 配置消除了该转换和 LDS，
最终写出从 `buffer_store_dwordx2` 变为 `buffer_store_dwordx4`。
配置扫描中，N=16,777,216 的图回放时间从 649.89 μs 降至 305.17 μs。
此处图回放计时用于解释设备侧变化；第一节的正式加速比始终采用完整 host API 时间。

本次数据对应实现提交 `5fb6caa`，
验收 run 为 `4abbd70eb84941879227f0c0c428e351`，统一 runner 耗时 `0:10:37`。
测试使用一个 BW1000 节点的物理 GPU 2。

复现代码从 [FlagRAND 公开仓库](https://github.com/flagos-ai/FlagRAND) 获取，
环境和命令见 [Hygon 运行指南](hygon.md)及
[硬件验收流程](../../getting-started/hardware-acceptance.md)。
统一入口在仓库的 `hygon_results/ci_<RUN_ID>/` 下生成 `summary.json`，
各 API 的 `performance_raw/` 保留计时样本。该历史轮次还保存了逐 API/shape 分析、
实现与打包文件清单；重新测试的结果应记录自己的源码版本和 run ID，不能直接沿用本文数值。
