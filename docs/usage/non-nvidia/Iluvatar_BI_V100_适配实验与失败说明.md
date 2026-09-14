# Iluvatar BI-V100 适配实验记录

2026-09-14，节点 u98，驱动/IX-ML 3.2.3。此次尚未建立原生库基线，不能报告有效加速比或声称 API 已通过。

五轮有限实验总计费 55.096566 秒，使用 GPU1 和 GPU3，通过平台 node agent 随机启动并共享日预算。完整数据在平台 reports/flagrand-iluvatar-bi-v100-20260914/，含源码快照、候选补丁、运行配置、统一 summary.json 和失败编译产物。

| 尝试 | 结果 |
|---|---|
| CoreX4.4 整体镜像 | 缺少 flagcx 扩展；关闭自动加载后，设备初始化报803 |
| 新镜像只读使用宿主 libcuda/libixthunk | 新NCCL缺少cuMemRetainAllocationHandle；再用旧NCCL则新Torch缺少ncclMemFree |
| 原Torch2.1/CoreX3.2.3 + 独立天数Triton3.1.0 | 识别 arch70 / warp64，seed0 raw golden-vector 可运行；通用raw与normal pair编译退出 |
| 自写BI-V100二维连续写出候选 | raw32 2 passed /20 failed，normal仍编译退出；未采用 |

Python 后端虽然将 arch70 映射为 ivcore10，实际编译器却提示不识别 ivcore10，并出现寄存器类别错误。对齐 raw 产物 shared=4096；失败路径与成功路径均有 llvm.bi.store.kop.i32，因此不能只凭指令名称下结论。R4/R5共保留31组编译产物；.asm为空，现有反汇编工具未能输出有效机器指令，VGPR/spill不可用。

部分错误发生在未改动的对齐路径。尚需以非零seed和错位输出对比regular/cached JIT，区分启动参数与代码生成问题；也应确认BI-V100对应的编译器构建。没有通过改变架构标识、删除测试、缩减shape或调整容差宣称达标。原生FP64探测未执行，不套用BI-V150结论。

本轮另修复测试记录：每个完成用例立即写入JSON；若随后编译器导致进程异常退出，保留已完成用例且阶段仍标为错误。统一runner以实际子进程异常退出验证，17项测试通过。该修复不改变性能计时。
