# FlagRAND 文档

FlagRAND 文档沿用 FlagGems 的用途分类，不在现行文档树中保留阶段包说明、版本审计或一次性调试报告。

| 分类 | 内容 |
|---|---|
| [getting-started](getting-started/) | 安装、硬件验收与结果校验 |
| [usage](usage/) | 公共 API 和不同硬件后端的运行说明 |
| [contribution](contribution/) | 算子开发、测试、性能与后端调试 |

正式硬件验收入口为
[hardware-acceptance.md](getting-started/hardware-acceptance.md)，覆盖 NVIDIA H20、天数智芯 BI-V150、
PPU-ZW810E、华为 Ascend 910C 和沐曦 MetaX C550。平台实现和诊断说明位于
[usage/non-nvidia](usage/non-nvidia/)。

NVIDIA H20 的安装、原生 cuRAND 对比和验收命令见
[nvidia-h20.md](usage/nvidia-h20.md)。

沐曦 MetaX C550 的 MACA 容器、mcRAND baseline、独立平台分流和原始矩阵见
[muxi.md](usage/non-nvidia/muxi.md)。
