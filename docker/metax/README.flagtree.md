# MetaX C550：原 Torch 与 FlagTree Triton 3.6

默认开发环境使用 `torch==2.8.0+metax3.7.0.7` 和
`flagtree===0.7.0.post2+metax3.6`，由 FlagTree 提供 Triton 3.6.0。
完整的 clone、环境准备、统一测试及结果读取步骤见
[C550 环境与复现](../../docs/usage/c550-environment.md)。

这一组合已在原基础镜像创建的独立容器中完成模块导入和一次 add kernel 执行，
257 个 FP32 输出全部正确，target 为 `maca / arch 80 / warp size 64`。
同环境的统一 r1 已完成 32 API、1,888 个配对点，442 项测试执行通过、8 项 CoreX 专属
跳过、0 失败；25/32 API 的几何平均不低于 0.95。独立 r2 完成所选八 API 的 150 个
配对点，8/8 达到目标，另有 MTGP32 全部 90 项正确性测试通过。r3 的新 FP32 共享状态
接线已通过两 seed 的专项比较，完整八 API 的 520 个配对点中 5/8 API 达到目标；
uniform FP64、lognormal FP32 和 Poisson 仍未达到 0.95。独立 r4 的 buffered 特化签名
接线也完成全部 520 点，5/8 API 达到目标；lognormal FP32/FP64 和 Poisson 仍不足。各轮源码范围不同，
不能拼接成整体达标结果；当前状态和复现命令见
[C550 环境与复现](../../docs/usage/c550-environment.md)。
r5 同一源码的四个固定组已完成32 API与1,888点：457项测试执行通过、8项CoreX专属跳过、
0失败，31/32 API达到0.95；唯一不足的是MTGP32 uniform FP64（0.702032）。
后续r6的短uniform缓存及同stream runtime复制已完成独立MTGP32八API、520点测试：
111项正确性执行通过，8/8 API达到0.95；uniform FP64为1.546692，八API等权几何平均为1.154102。
这不是新的完整32 API结果，也不将跨轮原生库时间变化算作单项代码收益。
已发布的 `6361eefda3a164a955f67fb16f725e7e27fd2bd8` 与
`9554017558051ea101c64d431093eeecd754af7a` 均在本机真实 clone 后将 checkout 传到
C550，在新环境中测试。前者的 MTGP32 uniform FP32 为0.745122；后者的 fresh-clone r2
中，MTGP32 Poisson 全220点为0.705456，均未达到0.95，保留各自完整结果。
当前实现将 `N≤65536` 的uniform FP32/FP64及Poisson接入同stream D2D派生缓存。
Poisson仅保留最近一个表身份与两种raw起点奇偶位置的数组，额外上界约4 MiB；三类派生
数组合计上界为 `16×(1048576+255)` 字节，约16 MiB，不含raw窗口与已有alias表。
大raw窗口不创建派生缓存；窗口耗尽、替换或seed重置时清理引用，存储寿命遵循当前stream。
独立r7在物理GPU6完成UF32全部44点，几何平均1.151110；UF64前置17项与UF32正式14项
正确性执行通过，0失败、0跳过。35个短输入点与其余9点的几何平均分别为1.273707、0.776587，
没有删去慢点。独立r8在物理GPU3完成MTGP32 Poisson全部220个shape/λ配对点、440条
原始计时记录，16项正确性执行全部通过、0失败、0跳过，相对同轮mcRAND的几何平均为
1.488941。两侧完整API时间的几何平均为13.733/20.448 µs（FlagRAND/mcRAND）；
175个短输入点与其余45点的加速比分别为1.725027、0.840051，慢点全部保留。
r8只覆盖此项API，修订后的最终提交仍需重新完成32 API复现；不将局部结果与其他轮次拼接为整体达标结果。
此前 Triton 3.1 或另一套 Torch 3.7.2 环境的结果分别保留。

## 构建

默认入口为 `docker/metax/Dockerfile`。`Dockerfile.flagtree` 保留相同配方，兼容已有命令。
在仓库根目录运行：

```bash
bash docker/metax/build.sh
```

默认镜像名为 `flagrand-metax:c550-flagtree0.7.0.post2-metax3.6-py312`。
`FLAGRAND_METAX_BASE_IMAGE` 可以指定已有的基础镜像；未指定时使用
`cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.20.2-maca.ai3.7.0.107-torch2.8-py312-ubuntu22.04-pd`。

本轮设备验证使用的是从归档导入的本地 image ID：

```text
sha256:b2662f3e02b13c9753399046ebb2b6e2b210f95a2f5bf65fb54fd3ffb4f32a1c
```

该镜像的 RepoTags 和 RepoDigests 均为空，image ID 不能代替仓库 manifest digest。
若当前 Docker 已保存该镜像，可以这样构建：

```bash
docker tag \
  sha256:b2662f3e02b13c9753399046ebb2b6e2b210f95a2f5bf65fb54fd3ffb4f32a1c \
  flagrand-metax-base:c550-maca3.7.0.107-py312
FLAGRAND_METAX_BASE_IMAGE=flagrand-metax-base:c550-maca3.7.0.107-py312 \
  bash docker/metax/build.sh
```

这里的 `docker tag` 只为宿主机已有镜像增加本地名称，不执行推送。

Dockerfile 在新镜像内移除旧 `triton`、`flagtree` 发行包，再从
[FlagTree 官方包仓库](https://resource.flagos.net/repository/flagos-pypi-hosted/simple)
安装固定版本 wheel。安装使用 `--no-deps`，保留基础镜像的 MetaX Torch；
测试依赖固定为 pytest 8.4.1、PyYAML 6.0.3、setuptools 69.5.1 和 wheel 0.45.1。
构建阶段使用 `/opt/conda/bin/python3` 打印导入版本，不启动 GPU kernel。

上述 Docker 构建配方尚未实际执行，来源标签的远端拉取也尚未验证。
已完成的设备检查来自同一基础镜像上按文档命令安装的独立容器。

## 运行

在宿主机仓库根目录指定分配的物理设备，再运行开发镜像：

```bash
export MACA_VISIBLE_DEVICES=0
bash docker/metax/run.sh
```

`run.sh` 挂载当前源码，设置 MetaX/CUDA 可见设备 mask 和 4 GiB 共享内存。
默认缓存目录为 `.triton_cache_metax36`、`.flagtree_cache_metax36` 和 `.cache_metax36`，
与之前的编译缓存分开。

容器中检查实际模块版本：

```bash
/opt/conda/bin/python3 - <<'PY'
import importlib.metadata
import torch
import triton
from triton.runtime import driver

print("Torch:", torch.__version__)
print("FlagTree:", importlib.metadata.version("flagtree"))
print("Triton:", triton.__version__, triton.__file__)
print("Target:", driver.active.get_current_target())
PY
```

随后按 [统一测试步骤](../../docs/usage/c550-environment.md#5-执行统一测试)
生成 `summary.json` 与 `api_summary.json`。实际设备归属、正确性和逐 API 几何平均
以该次运行产生的文件为准。
