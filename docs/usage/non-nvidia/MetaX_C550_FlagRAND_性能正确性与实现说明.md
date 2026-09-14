# MetaX C550 · FlagRAND 性能、正确性与实现说明

日期：2026-09-14。最终实测源码提交：`13cc585`，分支：`codex/cross-chip-tuning`。测试为一个 C550 节点的物理 GPU 1，不能据此声称已逐块检验所有 C550。

## 第一节：加速比统计

测试设备 MetaX C550，PyTorch `2.8.0+metax3.7.0.7`，Triton `3.1.0`，target `maca / arch=80 / warp_size=64`。原生 baseline 为 mcRAND，动态库 `/opt/maca/lib/libmcrand.so`，版本值 `100000`。

复用持久 generator 和预分配输出，绑定当前 Torch stream。保持原有 MetaX acceptance shape 清单及 full 计时：预热 5 次、正式 11 组、单组目标 200 ms，pilot 决定每组 1–1000 次调用。正式指标为同步的 `total_ms_median`；加速比为 mcRAND / FlagRAND。每个 API 对**全部配置 shape**的加速比取算术平均，不计算跨 API 总平均。此次未修改计时参数、shape 清单、正确性阈值或 native baseline。

| API | dtype | shape 数 | API 平均加速比 | 最小加速比 | 最大加速比 |
|---|---|---:|---:|---:|---:|
| raw32 | int32 | 16 | 1.916× | 1.816× | 2.124× |
| uniform_f32 | float32 | 16 | 1.880× | 1.251× | 2.107× |
| uniform_f64 | float64 | 16 | 1.785× | 1.661× | 2.147× |
| normal_f32 | float32 | 17 | 1.532× | 1.462× | 1.762× |
| normal_f64 | float64 | 11 | 1.524× | 1.309× | 1.583× |
| lognormal_f32 | float32 | 17 | 1.549× | 1.457× | 1.843× |
| lognormal_f64 | float64 | 15 | 1.574× | 0.990× | 1.680× |
验收标准为每个 API 均值 >=0.95×，没有上限；最终 7/7 API 达标。保留唯一慢于 mcRAND 的实测点：lognormal_f64、N=262144，0.990×；未删除或重跑挑选更好的数值。

R1 的 N=1025 曾出现 raw32 0.627×、uniform_f32 0.695×。减少计数器值特化后，最终分别为 1.844×、1.836×，FlagRAND 的同步调用中位数为 15.925 μs、16.031 μs，pilot 都恢复到每组 1000 次。诊断图内耗时前后仍约 4 μs，不能把这段端到端改善解释成 GPU 算术吞吐提升数倍。

部分其它输入仍有首组长停顿，逐组样本全部保留。此表是**未改变既有计时流程的实测结果**，不宣称所有编译特化均已在预热阶段覆盖，也不把 host API 时间当成纯 GPU 内核时间。进一步诊断应使用有限输入，而非重复完整基线。

## 第二节：正确性统计

| 校验范围 | 数量 | 结果 |
|---|---:|---|
| 完整交付 accuracy / performance 阶段 | 14 | 全部通过 |
| FlagRAND acceptance cases | 108 | 108 pass，0 fail |
| mcRAND baseline cases | 108 | 108 pass，0 fail |
| acceptance 全部记录 | 216 | 216 pass，0 fail |
| 去重 accuracy 用例 | 162 | 154 passed，8 skipped，0 failed |
| 新增 raw32 独立 CPU oracle 进位检查 | 18 | 全部通过，已包含于上述 154 项 |
| 新增七 API 跨边界整段/分段一致性检查 | 56 | 全部通过，已包含于上述 154 项 |
| benchmark preflight | 7 次 | 每次 golden vector 与七 API split/bulk 检查均通过 |
| 统一 runner 测试 | 14 | 全部通过 |

8 个 skipped 是 CoreX 专属用例，不是把 C550 的失败转为跳过。保留原有 seed=0 raw Philox golden vector、同 seed 重放、任意起始 offset、当前 stream、分布范围与统计检查；mcRAND 检查自身的确定性和分布性质，不要求两库所有分布输出逐元素相同。

新增检查覆盖 offset=2³³−4、2³³、2³⁴−4、2³⁴+1 等，以及对齐和非对齐长度；split/bulk 还覆盖边界前一 word 的偏移。R3 新增 oracle 曾发现进位错误；R4 又暴露 constexpr 转换错误。两次失败独立保留，最终没有通过缩小检查范围来达标。

## 第三节：实现方法与可复用经验

| API / 部件 | 当前实现 |
|---|---|
| raw32 | Triton 内执行 Philox4x32-10；对齐时每个 counter 写四个 int32，非对齐时使用四 word 滑动窗口。 |
| uniform_f32 | Philox 与 (0,1] float32 映射融合；保留对齐与滑动窗口路径。 |
| uniform_f64 | 两个 uint32 构成 53-bit 随机精度，再生成 double 输出。 |
| normal_f32 | Philox 与 Box–Muller 融合；MACA 使用 tl.log、tl.sqrt、tl.sin、tl.cos。 |
| normal_f64 | 两个 uniform_f64 经过 Box–Muller；保留 FP64 多项式 sin/cos 及标准分布编译期分支。 |
| lognormal_f32 / f64 | 复用对应正态路径，并在同一 kernel 内执行 exp。 |
| 计数器进位 | `_counter_at` 在向量加法之前，将 base_counter 和 index 通过 tl.cast 转成 uint64；覆盖 14 处对齐、非对齐及已有 libtuner 入口。 |
| 减少额外编译 | raw32、uniform_f32 的 generic kernel 使用 do_not_specialize=["base_counter"]；SHIFT、BLOCK、dtype 等必要区分仍保留。 |
| native baseline | ctypes 调用 mcRAND 的 generator、seed/offset/stream 和七类生成 API，直接写入预分配的 Torch 张量。 |

本次 manifest 记录 libtuner 可用但未启用（requested=false）。相关调用点同步修正，自动调参路径未另行实卡验收。

三条有实测证据的经验：

1. **先区分 CPU 提交与 GPU 工作。** R2 的 raw32 N=1025 CPU profile 明确记录 MACA 二进制生成，整个 measure 流程中调用由 7 次减到 3 次；相同输入的图内时间几乎不变。值特化还会影响 pilot，从而改变每组实际调用次数；只看一个平均加速比会掩盖原因。
2. **升位必须发生在加法之前。** 旧 TTIR 是 tensor<i32> add 后再 extsi；跨边界时已丢失正确值。最终 TTIR 显示先扩展标量与向量，再执行 tensor<i64> add。独立 CPU oracle 和跨边界 split/bulk 检查确认修复；仅做同 seed 重放无法发现这种共同错误。
3. **JIT 的值特化会影响可用的语言接口。** counter=1 在 MACA 路径上可能成为 constexpr，不能直接调用对象 .to；tl.cast 先把标量/constexpr 转为 tensor，再执行转换。这里修复的是我们自己的 Triton 写法。

最终轮次回收了 244 份编译缓存产物，包含元数据、TTIR、TTGIR、mcfatbin。编译元数据中，24 份 `_normal_f64_kernel` 及一份 `_raw_kernel` 申请 4096 字节 shared；所观察 FP64 TTGIR 伴随一次 convert_layout。这是资源与布局证据，尚没有分块对照或厂商硬件计数器测量，不能直接断言它就是性能瓶颈。

R2 runtime 报告的 raw kernel n_regs=16/18、shared=0、spills=0 对应进位修正前的策略实验。最终缓存未提供 n_regs/n_spills，未将历史值填作最终值；没有采集到可读 LLVM IR、ISA 反汇编或厂商搬运/计算分项时间。

完整配套证据位于平台工程 `reports/flagrand-metax-baseline-20260914/`：R1 初始基线、R2 策略诊断、R3/R4 失败、R5 最终 summary、每个 shape 的原始样本与编译产物。所有研究轮次经 node agent 在随机时刻启动、计入既有日预算，并沿用节点级让卡、白天预留卡和连续空闲恢复策略。
