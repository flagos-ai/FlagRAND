# MetaX C550：原 Torch 与 FlagTree Triton 3.6

默认开发环境使用 `torch==2.8.0+metax3.7.0.7` 和
`flagtree===0.7.0.post2+metax3.6`，由 FlagTree 提供 Triton 3.6.0。
完整的 clone、环境准备、统一测试及结果读取步骤见
[C550 环境与复现](../../docs/usage/c550-environment.md)。

这一组合已在独立新容器中完成最终源码的完整复现。

最终源码 `2e9233d912d7cf3536a583217c668947cb28e3fa` 已完成本机 clone、checkout 传到 C550、新环境测试的独立复现。本轮覆盖 32 个 API、1,888 个 shape/参数点，全部保留同设备 mcRAND 与 FlagRAND 的配对时间。32/32 个 API 的 shape 几何平均达到 0.95。正确性执行记录为 475 passed、0 failed、8 skipped；这些数值按各 API 的测试执行次数统计。

Poisson 四生成器等权几何平均为 1.588938，MTGP32 八 API 等权几何平均为 1.614291。MTGP32 uniform FP32/FP64 分别为 1.717579/1.590052。逐 API 目标与族级汇总分别列出，原有 summary 的算术平均字段保持不变。

汇总结果见 [C550 验收说明](../../docs/usage/c550-acceptance-20261021.md)，安装与测试入口见 [C550 环境与复现](../../docs/usage/c550-environment.md)。本轮从已有基础镜像安装固定版本依赖，以下 Dockerfile 构建配方没有在本轮执行。

架构优化结合实际二进制内的 LLVM、设备活动时间线及完整 Host API 计时：共享内存复用状态和小型 Poisson 表，XORWOW 采用两阶段 FP64 变换，有限范围的 exp 删除多余处理，短 uniform、normal/lognormal 和 Poisson 复用派生缓存，并通过当前 stream 执行 D2D 复制。短 raw32 复用已有 raw 窗口，缓存命中用 D2D 复制，refill 写输出路径保留，不增加派生数组。共享内存和两阶段变换以设备 kernel 时间对照解释；短调用缓存减少主机侧重复提交与分发，其完整 API 收益和 kernel、memcpy 活动分别记录，不将主机路径收益归为 kernel 加速。这些改动的机制与收益由独立控制/候选实验解释；本节的验收数值仅来自最终完整轮次，历史轮次的倍率变化不作为单项代码收益。

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
