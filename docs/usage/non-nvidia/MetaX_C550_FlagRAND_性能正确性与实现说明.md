# 第一节 加速比统计

测试设备为 MetaX C550，运行环境为 PyTorch `2.8.0+metax3.7.0.7`，Triton target 为
`maca / arch=80 / warp_size=64`。性能 baseline 为沐曦原生 mcRAND，动态库路径为
`/opt/maca/lib/libmcrand.so`，mcRAND 版本值为 `100000`。

计时使用持久 generator 和预分配输出张量，编译预热不计时。每个 shape 预热 8 次、
正式重复 5 组、单组目标计时预算 80 ms。正式指标为 `total_ms_median`，加速比计算式为
`mcRAND total_ms_median / FlagRAND total_ms_median`；大于 1.0x 表示 FlagRAND 更快。
API 平均加速比是该 API 全部实测 shape 加速比的算术平均，不计算跨 API 总平均。

| API | dtype | shape 数 | API 平均加速比 | 最小加速比 | 最大加速比 |
|---|---:|---:|---:|---:|---:|
| raw32 | int32 | 16 | 1.189x | 1.131x | 1.314x |
| uniform_f32 | float32 | 16 | 1.163x | 1.067x | 1.281x |
| uniform_f64 | float64 | 16 | 1.082x | 0.994x | 1.297x |
| normal_f32 | float32 | 17 | 0.833x | 0.760x | 0.946x |
| normal_f64 | float64 | 11 | 0.876x | 0.805x | 1.065x |
| lognormal_f32 | float32 | 17 | 0.840x | 0.769x | 0.972x |
| lognormal_f64 | float64 | 15 | 0.899x | 0.865x | 0.987x |

验收目标范围为 `0.95x–1.50x`。raw32、uniform_f32、uniform_f64 的 API 平均加速比
位于目标范围内；本次验收汇总为 3/7 APIs in target range。

| API | 最小加速比 shape | 最小加速比 | 最大加速比 shape | 最大加速比 |
|---|---:|---:|---:|---:|
| raw32 | N=1,025 | 1.131x | N=1,048,576 | 1.314x |
| uniform_f32 | N=1,023 | 1.067x | N=1,048,576 | 1.281x |
| uniform_f64 | N=1,023 | 0.994x | N=1,048,577 | 1.297x |
| normal_f32 | N=126 | 0.760x | N=1,048,576 | 0.946x |
| normal_f64 | N=128 | 0.805x | N=262,144 | 1.065x |
| lognormal_f32 | N=1,022 | 0.769x | N=1,048,574、1,048,576 | 0.972x |
| lognormal_f64 | N=4,096 | 0.865x | N=262,144 | 0.987x |

# 第二节 正确性统计

| 校验范围 | 数据量 | 结果 |
|---|---:|---:|
| FlagRAND preflight | 1 组 | pass |
| FlagRAND acceptance cases | 108 个 | 108 passed，0 failed |
| mcRAND baseline cases | 108 个 | 108 passed，0 failed |
| acceptance 全部记录 | 216 条 | 216 passed，0 failed |

preflight 已通过 seed=0 的 raw32 Philox golden vector，以及七类 API 的整段生成与分段连续
生成一致性检查。acceptance case 已通过相同 seed 重放一致性检查；uniform 输出通过有限值、
范围和均值检查；normal 输出通过有限值、均值和标准差检查；lognormal 输出通过有限值、
正值以及取对数后的均值和标准差检查。

# 第三节 实现方法

| API | 实现方法 |
|---|---|
| raw32 | Triton kernel 内执行 Philox4x32-10 counter-based 生成。4-word 对齐时每个 counter 直接写出 4 个 int32；非对齐 offset 或 shape 使用四 word 滑动窗口。 |
| uniform_f32 | 在 Philox kernel 内将 uint32 直接映射为 `(0,1]` float32；对齐路径每个 counter 生成 4 个输出，非对齐路径使用滑动窗口。 |
| uniform_f64 | 每个 double 使用两个 Philox uint32 组成 53-bit 随机精度；对齐路径每个 counter 生成 2 个 double，非对齐路径使用滑动窗口。 |
| normal_f32 | Philox 与两组 Box–Muller 变换融合在同一个 Triton kernel 中；MACA 后端使用可移植的 `tl.log`、`tl.sqrt`、`tl.sin`、`tl.cos` 数学路径。 |
| normal_f64 | 每两个 uniform_f64 通过 Box–Muller 生成两个 double 正态结果；标准分布 `mean=0、stddev=1` 使用编译期分支。 |
| lognormal_f32 | 复用 normal_f32 的融合 Philox + Box–Muller 路径，在同一个 kernel 内对结果执行 `exp`。 |
| lognormal_f64 | 复用 normal_f64 路径，在同一个 kernel 内对正态结果执行 `exp`。 |
| mcRAND baseline | Python 通过 ctypes 调用 `mcrandCreateGenerator`、seed/offset/stream 设置函数和七类生成函数；输出直接写入当前 CUDA-compatible Torch tensor。 |
| benchmark | FlagRAND 与 mcRAND 均复用 generator、预分配输出并绑定当前 Torch stream；smoke 预热覆盖非对齐 Philox 的 4 个 SHIFT 与 scalar specialization 状态。 |
