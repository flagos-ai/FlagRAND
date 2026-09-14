# Iluvatar BI-V150 FlagRAND 性能、正确性与实现说明

设备 BI-V150，CoreX 4.4.0；Torch 2.7.1，Triton 3.1.0，target cuda / arch=71 / warp_size=64。原生库 `/usr/local/corex/lib64/libcurand.so`，版本10102。源码691e13f，最终验收GPU2；本次范围不是7/7 API。

## 第一节 正确性策略与结果

保留公共API的seed重放、整段/分段生成、范围、均值和方差检查。新增独立CPU Philox oracle与CoreX浮点映射检查：offset 0、4、2³³−4、2³⁴−4，整块/尾块、偏移输出地址、非默认stream，以及CoreX %4特化键交错输入。raw32和uniform_f64逐位相同，FP32变换rtol=2e-5、atol=2e-6，容差未放宽。

去重106 passed、0 failed，其中新增41例全部通过。FlagRAND记录216 passed、原生库172 passed和44 unsupported，共432条。原始normal_f32在offset=2³³−4、N1024时1020/1024错误，N1026正确；修复六条CoreX计数器/word-index路径，相加前扩展uint64。修复前后TTIR分别显示i32相加后扩展、i64相加，独立oracle回归通过。

## 第二节 加速比统计

持久且持续推进的generator，预分配输出，预热5次；正式11组、目标200ms、pilot每组最多1000次。total_ms_median为正式指标，加速比=原生/FlagRAND；每个API对全部配置shape加速比作算术平均，>=0.95×达标，不计算跨API平均。

| API | 全部 / 已测 shape | 平均加速比 | 最小 / 最大 | 慢点数 |
|---|---:|---:|---:|---:|
| lognormal_f32 | 42 / 42 | 1.431× | 1.351488 / 1.923 | 0 |
| normal_f32 | 42 / 42 | 1.370× | 1.302098 / 1.971 | 0 |
| raw32 | 44 / 44 | 2.191× | 1.243910 / 2.423 | 0 |
| uniform_f32 | 44 / 44 | 1.563× | 1.041455 / 2.607 | 0 |
| uniform_f64 | 44 / 0 | 不完整 | 无对照 | 0 |

四个可对照API全部shape快于原生。uniform_f64的独立正确性与44个FlagRAND生成shape通过，原生三个double接口均实测返回106（DOUBLE_PRECISION_REQUIRED）；不提供虚构加速比。normal_f64、lognormal_f64未纳入CoreX已实现矩阵，仍是交付余项。用整数构造53-bit binary64均匀随机数，不等于已实现FP64正态变换。

## 第三节 实现方法

| API / 层次 | 选定实现 |
|---|---|
| raw32 | Philox4x32-10；整块四word向量写出、尾块四word窗口；512/4；缓存整数键按CoreX %4规则分类 |
| uniform_f32 | CoreX既有r0/r2/r1/r3映射及融合FP32转换；512/4 |
| uniform_f64 | 两个uint32形成53-bit有效位，通过整数操作构造IEEE-754 binary64；512/4 |
| normal_f32 | Philox+两组Box–Muller融合；256/4；相加前扩展64-bit counter |
| lognormal_f32 | Philox+Box–Muller+exp融合；512/4；相加前扩展64-bit counter |
| 启动 | legacy CachedKernelLauncher，每次取得当前stream，不缓存随机状态或输出 |

## 第四节 优化证据与局限

R1完整矩阵raw32平均0.638×、normal_f32平均0.705×，所有慢点保留。短诊断N4096、raw512/4，普通JIT完整调用78.65μs、缓存23.41μs，图内6.84→6.62μs；1000次CPU profile为0.137→0.045秒，支持主机提交路径是主要瓶颈。短诊断5组/20ms不替代正式验收。

修复前normal N16M，512/4→256/4图内212.88→199.35μs，编译元数据寄存器37→28，shared=0、spills=0；未用硬件occupancy counter证明因果。已安装CoreX JIT按整数%4特化，其他后端默认%16，故为天数单独传入alignment=4。

已有TTIR、TTGIR、LLIR；缓存.asm为空。ixobjdump拒绝无Fatbin section的cubin，llvm-objdump识别elf64-iluvatar但缺少bi--mesa3d反汇编器，未获得可读ISA。LLIR中的NVVM名称属于编译接口，不能等同于NVIDIA硬件。

R2曾因诊断辅助函数漏传normal参数及GPUTarget JSON写出失败；原始数据保留，R3只补缺失项，再定位真实carry错误。R1–R3为同节点GPU1，R4为GPU2；每轮原生/FlagRAND均在同一设备对照。

完整轨迹在平台工作区 `reports/flagrand-iluvatar-bi-v150-20260914/`：含rounds.json、release.json、统一summary、所有原始组样本、CPU profile、编译产物与失败。BI-V100和其他节点需独立验证，不能继承BI-V150结论。
