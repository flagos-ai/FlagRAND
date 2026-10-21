# C550 环境与复现

本文用于在 MetaX C550 上复现 Philox、XORWOW、MRG32K3A、MTGP32 及其 Poisson 接口。
算子实现放在 `src/flagrand`，正确性比较放在 `tests/accuracy`，原生 mcRAND 对照和计时放在
`benchmark`，统一测试入口为 `tools/run_tests.py`。这一分工沿用 FlagGems 的算子、后端、
正确性和性能测试组织方式。

## 1. 当前交付环境配置

| 项目 | 当前配置 |
|---|---|
| 设备 | MetaX C550 |
| 来源镜像标签 | `cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.20.2-maca.ai3.7.0.107-torch2.8-py312-ubuntu22.04-pd` |
| 本轮本地 image ID | `sha256:b2662f3e02b13c9753399046ebb2b6e2b210f95a2f5bf65fb54fd3ffb4f32a1c` |
| Python 入口 | `/opt/conda/bin/python3` |
| PyTorch | `2.8.0+metax3.7.0.7` |
| FlagTree 发行包 | `0.7.0.post2+metax3.6` |
| 导入的 Triton | `3.6.0` |
| Triton target | `maca`，arch `80`，warp size `64` |
| 原生 RNG | `/opt/maca/lib/libmcrand.so`，版本值 `100000` |
| pytest / PyYAML | `8.4.1` / `6.0.3` |
| setuptools / wheel | `69.5.1` / `0.45.1` |

本轮通过已有镜像归档导入环境。测试节点的 `RepoTags` 和 `RepoDigests` 均为空；上述
image ID 标识本地镜像，不能将它拼成仓库的 `@sha256:...` 拉取地址。来源标签保留用于
查找对应发布物，从仓库拉取仍需访问权限，本轮尚未验证该拉取过程。
完整运行环境以每次测试输出的 `env`、环境日志和源码版本为准。

当前配置保留镜像预装的 MetaX PyTorch、MACA runtime 和 mcRAND，安装固定版本 FlagTree。

最终源码 `2e9233d912d7cf3536a583217c668947cb28e3fa` 已完成本机 clone、checkout 传到 C550、新环境测试的独立复现。本轮覆盖 32 个 API、1,888 个 shape/参数点，全部保留同设备 mcRAND 与 FlagRAND 的配对时间。32/32 个 API 的 shape 几何平均达到 0.95。正确性执行记录为 475 passed、0 failed、8 skipped；这些数值按各 API 的测试执行次数统计。

Poisson 四生成器等权几何平均为 1.588938，MTGP32 八 API 等权几何平均为 1.614291。MTGP32 uniform FP32/FP64 分别为 1.717579/1.590052。逐 API 目标与族级汇总分别列出，原有 summary 的算术平均字段保持不变。

逐 API 结果、shape 覆盖、计时口径和正确性计数见 [C550 验收说明](c550-acceptance-20261021.md)。本轮使用已有基础镜像创建新容器，没有执行仓库 Dockerfile 的 Docker build。[FlagTree 环境说明](../../docker/metax/README.flagtree.md)保留安装与构建配方。

架构优化结合实际二进制内的 LLVM、设备活动时间线及完整 Host API 计时：共享内存复用状态和小型 Poisson 表，XORWOW 采用两阶段 FP64 变换，有限范围的 exp 删除多余处理，短 uniform、normal/lognormal 和 Poisson 复用派生缓存，并通过当前 stream 执行 D2D 复制。短 raw32 复用已有 raw 窗口，缓存命中用 D2D 复制，refill 写输出路径保留，不增加派生数组。共享内存和两阶段变换以设备 kernel 时间对照解释；短调用缓存减少主机侧重复提交与分发，其完整 API 收益和 kernel、memcpy 活动分别记录，不将主机路径收益归为 kernel 加速。这些改动的机制与收益由独立控制/候选实验解释；本节的验收数值仅来自最终完整轮次，历史轮次的倍率变化不作为单项代码收益。

`N≤65536` 的 MetaX raw32 包括 fresh aligned 调用，复用已有有界 raw 窗口，缓存命中时通过当前 stream 的 MACA D2D 复制输出，refill 写输出路径保留。此路径没有新增派生数组；短调用的 raw 预取窗口约 4 MiB，与下述派生存储分别计算。`N≤65536` 的 MetaX FP32/FP64 uniform、normal/lognormal 与 Poisson 使用有界派生缓存。Poisson 仅保留最近一个表身份及两种 raw 起点奇偶位置的派生数组，额外占用至多约 4 MiB。normal/lognormal 四个 API 共用一个最近的 `_normal_cache` 条目，由 raw 窗口、mode、dtype、参数位模式及 raw 起点余数确定；条件变化时替换而不按参数累积，新增单个数组的上界为 `4×(1048576+255)` 字节，约 4 MiB。所有分布/精度混合调用的活跃派生张量上界为 `20×(1048576+255)=20976620` 字节（约 20 MiB），不含 raw window、已有 alias 表及尚未完成的 stream 工作暂时保留的旧存储；大 raw 窗口不建立派生缓存。raw window 耗尽、替换或 seed 重置时释放派生引用，已排队工作使用的存储按 stream 生命周期保留。

本轮下载的 CPython 3.12 / Linux x86_64 wheel 大小为 425,411,075 字节，
SHA256 为 `4e113cc9e94e5788400ccc11bbd3d8d29739255a517c801784115dff6bc4dda1`。
对应官方包为 `flagtree-0.7.0.post2+metax3.6-cp312-cp312-linux_x86_64.whl`。
复现时可以复用这个依赖 wheel，并将其目录以只读方式挂入新容器；FlagRAND 源码和编译缓存
应来自本次 clone。没有该 wheel 时，使用下文的官方包仓库安装命令。

## 2. 获取代码

在安装 MetaX 驱动并能访问 C550 的 Linux 主机上执行。`FLAGRAND_REF` 指向要复现的已发布分支、
标签或 `main`；同一次复现保留对应的 commit。

```bash
export FLAGRAND_REF=main
git clone --branch "$FLAGRAND_REF" --single-branch \
  https://github.com/flagos-ai/FlagRAND.git FlagRAND
cd FlagRAND
git rev-parse HEAD
```

本机编辑后送到 C550 测试，以及对方重新 clone 后测试，使用相同的仓库入口和环境配置。
GPU 测量在 C550 上运行；普通 CPU 主机可以整理代码、构建 Python 包和读取结果。

## 3. 使用基础镜像，或构建开发镜像

已有上述基础镜像时，可以在新容器中安装固定版本 FlagTree，无需重新构建镜像。
以下示例将宿主机分配的设备号传入容器，
容器中的测试使用逻辑设备 `0`：

```bash
export MACA_VISIBLE_DEVICES=0
export FLAGRAND_METAX_BASE_IMAGE='sha256:b2662f3e02b13c9753399046ebb2b6e2b210f95a2f5bf65fb54fd3ffb4f32a1c'

docker run --rm -it \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --shm-size=4g \
  -e "MACA_VISIBLE_DEVICES=$MACA_VISIBLE_DEVICES" \
  -e "CUDA_VISIBLE_DEVICES=$MACA_VISIBLE_DEVICES" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e FLAGRAND_USE_LIBTUNER=0 \
  -v "$PWD:/workspace/FlagRAND" \
  -w /workspace/FlagRAND \
  --entrypoint /bin/bash \
  "$FLAGRAND_METAX_BASE_IMAGE"
```

容器内替换 Triton 发行包，再安装测试依赖和当前项目。`--no-deps` 保留镜像提供的 Torch：

```bash
export PYTHON_BIN=/opt/conda/bin/python3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PYTHON_BIN" -m pip uninstall -y triton flagtree
"$PYTHON_BIN" -m pip install --only-binary=:all: --no-deps \
  --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  'flagtree===0.7.0.post2+metax3.6'
"$PYTHON_BIN" -m pip install --no-deps \
  setuptools==69.5.1 wheel==0.45.1 PyYAML==6.0.3 pytest==8.4.1
"$PYTHON_BIN" -m pip install -e . --no-deps --no-build-isolation
source scripts/ci/backend_env.sh metax
export FLAGRAND_MCRAND_PATH=/opt/maca/lib/libmcrand.so
export TRITON_CACHE_DIR="$PWD/.triton_cache_metax36"
export FLAGTREE_CACHE_DIR="$PWD/.flagtree_cache_metax36"
export XDG_CACHE_HOME="$PWD/.cache_metax36"
mkdir -p "$TRITON_CACHE_DIR" "$FLAGTREE_CACHE_DIR" "$XDG_CACHE_HOME"
```

上述线程数用于控制正确性比较中的 CPU 并行度，避免多卡测试进程相互争用 CPU；
它们不改变 GPU kernel 的线程布局。

若希望将这些依赖和源码一起打入镜像，在宿主机仓库根目录执行。
已有本轮本地基础镜像时，先为它指定一个用于构建的本地名称：

```bash
docker tag \
  sha256:b2662f3e02b13c9753399046ebb2b6e2b210f95a2f5bf65fb54fd3ffb4f32a1c \
  flagrand-metax-base:c550-maca3.7.0.107-py312
export FLAGRAND_METAX_BASE_IMAGE=flagrand-metax-base:c550-maca3.7.0.107-py312
bash docker/metax/build.sh
bash docker/metax/run.sh
```

默认开发镜像为 `flagrand-metax:c550-flagtree0.7.0.post2-metax3.6-py312`。`build.sh` 使用
`FLAGRAND_METAX_BASE_IMAGE` 指定的镜像；未设置时使用上面的来源标签。
Dockerfile 安装固定版本 FlagTree、Python 测试依赖和当前 FlagRAND，保留基础镜像的 Torch。
`Dockerfile.flagtree` 保留相同配方。`run.sh` 挂载当前代码目录，
因此修改源码后无需重建镜像。构建阶段打印实际导入版本，不运行 GPU 测试。
这份 Docker 构建配方尚未实际执行；当前验证来自已有基础镜像中的独立安装流程。

## 4. 记录导入与设备

容器内执行：

```bash
"$PYTHON_BIN" - <<'PY'
import sys
import importlib.metadata
import flagrand
import torch
import triton
from triton.runtime import driver
from benchmark.muxi_matrix.mcrand import mcrand_library_path, mcrand_version

print("Python:", sys.executable, sys.version)
print("FlagRAND:", flagrand.__file__)
print("Torch:", torch.__version__)
print("FlagTree:", importlib.metadata.version("flagtree"))
print("Triton:", triton.__version__, triton.__file__)
print("Device:", torch.cuda.get_device_name(0))
print("Target:", driver.active.get_current_target())
print("mcRAND:", mcrand_library_path(), mcrand_version())
PY
```

`flagrand.__file__` 应来自当前 checkout 的 `src/flagrand`。若需要验证 wheel 安装，
先在独立目录安装 wheel，再在不设置源码 `PYTHONPATH` 的进程中检查导入路径；
checkout 的测试流程默认使用当前源码。

## 5. 执行统一测试

先运行少量 shape，确认新 checkout 中的设备执行和结果输出：

```bash
export FLAGRAND_PLATFORM=metax
export FLAGRAND_PERF_PROFILE=smoke
export FLAGRAND_OPS='raw32,uniform_f32,uniform_f64,normal_f32,normal_f64,lognormal_f32,lognormal_f64,xorwow,mrg32k3a,mtgp32,poisson'
"$PYTHON_BIN" tools/run_tests.py \
  --ops "$FLAGRAND_OPS" \
  --gpus 0 \
  --dump-output \
  --output-dir "$PWD/muxi_results/c550_smoke"
```

其中 `xorwow`、`mrg32k3a`、`mtgp32` 展开为各生成器的八个接口，包含 Poisson；
`poisson` 展开为四个生成器的 Poisson 接口。测试入口会去重，连同前面的 Philox 七接口，
共测试 32 个生成器/API 组合。

完整测试使用固定的 acceptance shape 集：

```bash
export FLAGRAND_PERF_PROFILE=acceptance
export FLAGRAND_RESULTS="$PWD/muxi_results/c550_$(date +%Y%m%d_%H%M%S)"
"$PYTHON_BIN" tools/run_tests.py \
  --ops "$FLAGRAND_OPS" \
  --gpus 0 \
  --dump-output \
  --output-dir "$FLAGRAND_RESULTS"
```

原有 Philox 七接口保持其历史 acceptance shape 定义。新增接口使用 44 个 raw/uniform/Poisson
shape 和 42 个 normal/lognormal shape，覆盖小规模、非对齐尾部和大规模输入。
Poisson 另覆盖五个 lambda。具体值由仓库配置生成，并保存在每次原始结果中。

`summary.json` 是统一交付结果，`api_summary.json` 保存逐 API 的结果；各 API 子目录保留
原始计时、正确性比较和命令输出。测试程序正常结束表示这次运行已完成，
性能是否达标以逐 API 的 `api_meets_target` 及原始结果为准。
`env.visible_device_masks` 保存启动时的设备 mask，各 API 的 `device.logical_worker` 和
`device.visible_device_masks` 保存实际工作进程及其设备分配。
初轮与 fresh clone 的配对复现应按每项 API 的实际物理设备安排。多卡队列动态分配任务，
两轮使用相同 mask 并不自动保证每项 API 落在同一张卡上；单卡运行时所有 API 使用该卡。

读取新增交付项的完整性、正确性和逐 API 性能：

```bash
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["FLAGRAND_RESULTS"])
summary = json.loads((root / "summary.json").read_text())
print("Summary fields:", list(summary))
for name in ("XORWOW", "MRG32K3A", "MTGP32", "POISSON"):
    result = summary["result"][name]
    performance = result["performance"]
    print(name, "accuracy:", result["accuracy"]["status"],
          "complete:", performance["complete_api_family"])
    for api, geometric_mean in performance["api_geometric_means"].items():
        print(" ", api, geometric_mean, performance["api_meets_target"][api])
PY
```

## 6. 统计口径

统一测试会生成根目录的 `summary.json`，并保留每个接口的正确性结果、性能结果和原始计时。
结果根结构使用 `timestamp`、`total_duration`、`env`、`result`，并以 `profile` 记录本次测试范围。正式命令及新增交付项
由本仓库的测试入口和 `conf/operators.yaml` 定义。

每个点的加速比为同设备、同生成器、同分布及精度下的
`mcRAND 耗时 / FlagRAND 耗时`。现有 `speedup` 字段保留逐 shape 加速比的算术平均；
新增生成器同时记录逐 API 的几何平均，不能将两者互换。

本轮逐 API 的目标为几何平均不低于 `0.95`。Poisson 的 lambda 也属于配置的一部分，
需要与 shape 一起记录。完整 shape 集中的慢点、失败点和不支持项保留在结果中；
此前小 shape 展示所得到的 MTGP32 数值不作为本轮完整测试的结论。

首次调用的编译、状态初始化和 Poisson 表构造，与稳定运行计时分别记录。
完整 API 的耗时包含其执行所需的启动和变换；不能拿单个融合 kernel 的耗时替代整个 API。

## 7. 格式与包构建

本仓库已经配置 Black、isort 和 flake8。对本次修改的 Python 文件运行格式化即可：

```bash
python -m isort path/to/changed_file.py
python -m black path/to/changed_file.py
python -m flake8 path/to/changed_file.py
```

代码分层、格式工具和测试组织参考
[FlagGems 的固定源码版本](https://github.com/flagos-ai/FlagGems/tree/453da2f32f8a3798dea6cd3678922d39020f3e52)；
采用本仓库 `pyproject.toml` 中已有的版本和行宽设置。
生成器状态、数值变换和设备分派分别维护；厂商原生基线只出现在 benchmark 中。

Python 包构建：

```bash
python -m build
```

`setuptools` 从 `src` 收集 `flagrand` 及其子包。MTGP32 参数使用 Python 模块保存，
不依赖外部 `.pt` 数据文件。硬件验收使用 clone 得到的完整仓库，其中包含测试、benchmark、
配置和命令入口。
