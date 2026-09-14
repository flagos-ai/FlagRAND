# FlagRAND 硬件验收流程

适用后端：NVIDIA H20、天数智芯 BI-V150、PPU-ZW810E、华为 Ascend 910C、沐曦 MetaX C550、海光 BW1000。

节点应已部署对应厂商的驱动和运行时。联网环境从指定提交获取源码；默认分支为 `main`，
验证尚未合并的修改时先设置 `FLAGRAND_REF=codex/cross-chip-tuning`，并记录 `git rev-parse HEAD`。
离线节点可上传该提交的完整源码归档。每轮使用独立的结果目录，保留源码版本和原始失败记录。

验收通过条件：执行入口正常结束，生成 `summary.json`，交付项名称与 schema 正确，所有交付项
的准确性失败数为 0，且七个 API 分别对配置中的全部 shape 计算原生库耗时 / FlagRAND 耗时
的算术平均，每个 API 均值必须 `>= 0.95x`。不计算跨 API 平均、不设上限、不删除慢点。
缺失 API、缺失 shape 或原生库不支持必须保留，不能记作七接口验收通过。

仓库沿用 [FlagGems 的后端组织](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/runtime/backend)：
`runtime/backend/device_finder.py` 发现厂商，`runtime/backend/_<vendor>/` 保存厂商描述和算子覆盖，
公共 API、通用 Philox/变换和厂商原生 benchmark 分开组织。海光实现位于
`src/flagrand/runtime/backend/_hygon/ops.py`，hipRAND 对照位于 `benchmark/hygon_matrix/`；
运行时通过 `SpecOpRegistrar` 注册海光覆盖。

## 1. 天数智芯 BI-V150 / CoreX 4.4

### 1.1 获取源码

克隆指定分支，并记录本次验收的代码版本。

```bash
git clone --branch "${FLAGRAND_REF:-main}" --single-branch \
  git@github.com:flagos-ai/FlagRAND.git FlagRAND
cd FlagRAND
git rev-parse HEAD
```

### 1.2 安装 FlagRAND

CoreX 的 Torch 和 Triton 由节点运行时提供。项目虚拟环境继承系统包，仅安装当前 FlagRAND 源码。

```bash
/usr/local/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

### 1.3 检查运行环境

加载 CoreX 路径，检查 BI-V150 设备及项目虚拟环境中的 Torch。

```bash
source scripts/ci/backend_env.sh iluvatar
bash tools/gpu_check_iluvatar.sh
```

### 1.4 执行正式验收

`acceptance` profile 执行环境检查、统一 runner 自测、正确性矩阵、性能矩阵及结果聚合。

```bash
export FLAGRAND_RUN_ID="corex_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_PERF_PROFILE=acceptance \
  bash scripts/ci/run_backend_ci.sh iluvatar
export SUMMARY="$PWD/corex_results/ci_${FLAGRAND_RUN_ID}/summary.json"
```

完成后执行第 7 节校验结果。

## 2. PPU-ZW810E

### 2.1 获取源码

克隆指定分支，并记录本次验收的代码版本。

```bash
git clone --branch "${FLAGRAND_REF:-main}" --single-branch \
  git@github.com:flagos-ai/FlagRAND.git FlagRAND
cd FlagRAND
git rev-parse HEAD
```

### 2.2 安装 FlagRAND

根据 `conf/backends.yaml` 创建项目虚拟环境，安装 PPU 对应的 Torch、Triton 和 FlagRAND。

```bash
./setup.sh thead
```

### 2.3 检查运行环境

加载 PPU SDK，检查设备及项目虚拟环境中的 Torch。

```bash
source scripts/ci/backend_env.sh thead
bash tools/gpu_check_thead.sh
```

### 2.4 执行正式验收

`acceptance` profile 执行环境检查、统一 runner 自测、正确性矩阵、性能矩阵及结果聚合。

```bash
export FLAGRAND_RUN_ID="ppu_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_PERF_PROFILE=acceptance \
  bash scripts/ci/run_backend_ci.sh thead
export SUMMARY="$PWD/ppu_results/ci_${FLAGRAND_RUN_ID}/summary.json"
```

完成后执行第 7 节校验结果。FP64 精确构造专项显示 8 个 `skipped` 属于平台能力差异，不计为失败。

## 3. 华为 Ascend 910C / CANN 9.0

### 3.1 获取源码

克隆指定分支，并记录本次验收的代码版本。

```bash
git clone --branch "${FLAGRAND_REF:-main}" --single-branch \
  git@github.com:flagos-ai/FlagRAND.git FlagRAND
cd FlagRAND
git rev-parse HEAD
```

### 3.2 安装 FlagRAND

根据 `conf/backends.yaml` 创建项目虚拟环境，安装 CANN 9.0 对应的 Torch、FlagTree/Triton 和
FlagRAND。

```bash
./setup.sh ascend-cann900
```

### 3.3 检查运行环境

加载 CANN 9.0 环境，检查 NPU 设备及项目虚拟环境中的 `torch_npu`。

```bash
source scripts/ci/backend_env.sh ascend-cann900
bash tools/gpu_check_ascend.sh
```

### 3.4 执行正式验收

`acceptance` profile 执行环境检查、统一 runner 自测、正确性矩阵、性能矩阵及结果聚合。

```bash
export FLAGRAND_RUN_ID="ascend_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_PERF_PROFILE=acceptance \
  bash scripts/ci/run_backend_ci.sh ascend-cann900
export SUMMARY="$PWD/ascend_910c_results/ci_${FLAGRAND_RUN_ID}/summary.json"
```

完成后执行第 7 节校验结果。

## 4. NVIDIA H20

### 4.1 获取源码

克隆指定分支，并记录本次验收的代码版本。

```bash
git clone --branch "${FLAGRAND_REF:-main}" --single-branch \
  git@github.com:flagos-ai/FlagRAND.git FlagRAND
cd FlagRAND
git rev-parse HEAD
```

### 4.2 安装 FlagRAND

容器：`flagtree`。

H20 节点预装 NVIDIA driver、CUDA、PyTorch 和 Triton。项目虚拟环境继承容器当前 Python 的硬件包。

```bash
export FLAGRAND_SYSTEM_PYTHON="${FLAGRAND_SYSTEM_PYTHON:-python3}"
./setup.sh nvidia
```

### 4.3 检查运行环境

```bash
source scripts/ci/backend_env.sh nvidia
bash tools/gpu_check_nvidia.sh
```

### 4.4 执行正式验收

```bash
export FLAGRAND_RUN_ID="h20_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_PERF_PROFILE=acceptance \
  bash scripts/ci/run_backend_ci.sh nvidia
export SUMMARY="$PWD/nvidia_results/ci_${FLAGRAND_RUN_ID}/summary.json"
```

统一 runner 会在每个 API 的 `performance_raw/` 下保留 benchmark 原始目录，其中包括
`manifest.json`、`records.jsonl`、`records.csv`、`summary.json` 和 `summary.md`。验收根目录的
`summary.json` 是统一验收汇总；子目录中的 `summary.json` 是该 API 矩阵的 FlagOS 汇总，
`summary.md` 仅作人工阅读。

完成后执行第 7 节校验结果。

## 5. 沐曦 MetaX C550 / MACA AI 3.7.0.107

### 5.1 获取源码

在宿主机克隆指定分支，并记录本次验收的代码版本。MetaX 验收镜像不提供 Git，因此
源码由宿主机准备，再挂载进容器。

```bash
git clone --branch "${FLAGRAND_REF:-main}" --single-branch \
  git@github.com:flagos-ai/FlagRAND.git FlagRAND
cd FlagRAND
git rev-parse HEAD
```

### 5.2 构建并启动 MetaX 容器

仓库中的 MetaX 容器配置位于 `docker/metax/`，结构与 Ascend 容器配置一致：

```text
docker/metax/
├── Dockerfile
├── build.sh
├── entrypoint.sh
└── run.sh
```

Dockerfile 使用以下已验证的 MetaX C550 镜像作为基础镜像：

```text
cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.20.0-maca.ai3.7.0.107-torch2.8-py310-ubuntu22.04-amd64
```

在当前新克隆的项目目录构建 FlagRAND MetaX 开发镜像：

```bash
export FLAGRAND_METAX_IMAGE='flagrand-metax:c550-maca3.7.0.107'
bash docker/metax/build.sh
```

`docker/metax/Dockerfile` 保留基础镜像中的 MACA PyTorch、Triton、mcRAND 和运行时，仅增加
Git/SSH、PyYAML、pytest、FlagRAND 源码及验收环境变量。需要替换基础镜像时可设置：

```bash
export FLAGRAND_METAX_BASE_IMAGE='<MetaX MACA base image>'
bash docker/metax/build.sh
```

首次创建持久开发容器：

```bash
bash docker/metax/run.sh
```

启动脚本把 `/dev/mxcd`、`/dev/dri` 传入容器，将当前源码挂载到
`/workspace/FlagRAND`，默认容器名为 `flagrand-muxi-dev`。可通过
`FLAGRAND_METAX_CONTAINER` 修改名称。

后续进入同一容器：

```bash
docker start -ai flagrand-muxi-dev
```

本节上述流程使用基础镜像预装的 MACA PyTorch/Triton。
如需验证 Python 3.12 / Triton 3.6 环境，使用
[FlagTree MetaX 容器指南](../../docker/metax/README.flagtree.md)中的
`docker/metax/Dockerfile.flagtree`。该 py312 基础镜像的 Conda Python 已提供 Torch，
还需安装 `flagtree===0.6.1+metax3.6` 来提供 Triton。

### 5.3 检查运行环境和 mcRAND

在容器内固定从当前源码目录导入，并把 Triton 缓存留在项目目录中：

```bash
export PYTHONPATH="$PWD/src:$PWD"
export TRITON_CACHE_DIR="$PWD/.triton_cache"
export XDG_CACHE_HOME="$PWD/.cache"
export PYTHON_BIN=python3
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
```

加载 MetaX 后端环境，检查 MetaX C550、Torch、Triton target 和 mcRAND：

```bash
source scripts/ci/backend_env.sh metax
bash tools/gpu_check_metax.sh
```

也可以直接打印当前 Python 环境：

```bash
python3 - <<'PY'
import flagrand
import torch
from triton.runtime import driver

print("flagrand:", flagrand.__file__)
print("torch:", torch.__version__)
print("available:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0))
print("target:", driver.active.get_current_target())
PY
```

已验证的镜像环境显示 `torch==2.8.0+metax3.7.0.7`、`MetaX C550` 和
`GPUTarget(backend='maca', arch=80, warp_size=64)`。再检查原生 mcRAND baseline：

```bash
ldconfig -p | grep -E 'libmcrand|libmcruntime'

python3 - <<'PY'
from benchmark.muxi_matrix.mcrand import mcrand_library_path, mcrand_version

print("mcRAND library:", mcrand_library_path())
print("mcRAND version:", mcrand_version())
PY
```

已验证的镜像返回 `/opt/maca/lib/libmcrand.so` 和版本值 `100000`。

### 5.4 执行正确性预检

```bash
python3 -m benchmark.run_muxi --preflight-only
```

预检覆盖 seed=0 raw32 golden vector，以及七个 API 跨 aligned/generic 路径的 bulk/split 与
offset 流连续性。
通过时输出包含：

```text
Preflight: pass
```

### 5.5 执行正式验收

统一 runner 以 `metax` 为 canonical backend（`muxi` 和 `maca` 仅作别名），并使用 mcRAND
作为七个 Host API 的原生性能 baseline：

性能验收以每个 API 全部实测 shape 加速比的算术平均值 `>= 0.95x` 为通过条件，
不要求每个 shape 单独达标，不计算跨 API 平均值。加速比不设上限，也不因高加速比额外提示。

```bash
export FLAGRAND_RUN_ID="metax_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN=python3 \
FLAGRAND_PERF_PROFILE=acceptance \
  bash scripts/ci/run_backend_ci.sh metax
export SUMMARY="$PWD/muxi_results/ci_${FLAGRAND_RUN_ID}/summary.json"
```

统一 runner 在每个 API 的 `performance_raw/` 下保留 `benchmark.run_muxi` 原始产物，验收根目录的
`summary.json` 用于第 7 节的统一结果校验。

如需在同一进程中直接复核完整 MetaX 原始矩阵，执行：

```bash
python3 -m benchmark.run_muxi \
  --profile full \
  --acceptance-profile \
  --acceptance-baseline mcrand \
  --with-mcrand \
  --acceptance-min 0.95 \
  --strict-acceptance \
  --skip-torch-reference
```

固定 shape 集包含 108 个 FlagRAND case 和 108 个 mcRAND case，共 216 条记录。通过时
输出包含 `Acceptance status: pass (7/7 APIs meeting target)` 和
`Completed 216 records with 0 failure(s).`

pre-main 沐曦分支已产生过上述 216 条通过结果；将 NVIDIA 新 main 与 MetaX 独立分流
集成后，必须以新 `main` 的 C550 实测产物作为本次验收结论，不沿用 pre-main 性能数值。

完成后执行第 7 节校验结果。

## 6. 海光 BW1000 / DTK 26.04

### 6.1 获取源码

联网工作机可使用前述克隆方式。无法访问 GitHub 的计算节点，上传本次提交的源码归档：

```bash
# 在持有目标提交的工作机执行，归档包含 src、benchmark、tests、tools、conf 和 scripts。
git rev-parse HEAD
git archive --format=tar.gz -o /tmp/flagrand-acceptance-source.tar.gz HEAD
```

将该文件通过堡垒机上传后，在计算节点解压到本轮源码目录：

```bash
mkdir -p FlagRAND
tar -xzf /path/to/flagrand-acceptance-source.tar.gz -C FlagRAND
cd FlagRAND
```

记录来源提交和归档 SHA256。平台 node agent 上传同一份完整源码并记录运行 ID 时，
可以直接在其 `vendor/flagrand` 目录执行下述步骤。

### 6.2 使用 DTK 环境

复用匹配驱动的 DTK 容器/环境。已验证环境为 DTK 26.04、PyTorch 2.10.0、
HIP 6.3.26113、Triton 3.6.0 / FlagTree 0.6.0+hcu.git46341ffa，
Triton target 为 `hip / gfx936 / warp_size=64`。

直接从源码运行时无需联网安装：

```bash
export PYTHON_BIN=/usr/bin/python3
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/ci/backend_env.sh hygon
```

已有调度器分配的设备掩码应原样保留；独立执行时先选择空闲设备并设置
`CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`。
需要安装到独立项目环境时可运行 `./setup.sh hygon`，它继承 DTK 的系统硬件包。

### 6.3 检查设备与原生库

```bash
bash tools/gpu_check_hygon.sh
"$PYTHON_BIN" - <<'PY'
import torch
from triton.runtime import driver
print("torch:", torch.__version__, "HIP:", torch.version.hip)
print("target:", driver.active.get_current_target())
print("device:", torch.cuda.get_device_name(0))
PY
```

原生基准是 `/opt/dtk/lib/libhiprand.so`，已测版本值 `300200`；其他安装位置可设置
`FLAGRAND_HIPRAND_PATH`。实际库路径和版本随正式矩阵写入各 API 的 `manifest.json`。

### 6.4 正确性和正式验收

```bash
export FLAGRAND_RUN_ID="hygon_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_PERF_PROFILE=acceptance \
  bash scripts/delivery/run_hygon.sh
export SUMMARY="$PWD/hygon_results/ci_${FLAGRAND_RUN_ID}/summary.json"
```

该脚本调用与其他芯片相同的 `scripts/ci/run_backend_ci.sh hygon`。平台 node agent
直接调用此统一入口时，执行内容一致；测试作为单次有限研究任务、随机启动，并计入节点日预算。

执行环境检查、runner 自测及海光 API 回归后，统一 runner 为七个 API 分别执行正确性和性能。
正确性包含独立 CPU Philox oracle、seed 重放、非对齐 offset、整段/分段连续性、
64 位 counter 进位、非默认 mean/stddev 的 FP64 精度，以及分布统计检查。
FlagRAND 与 hipRAND 分别验证，不要求两家原生 generator 的输出顺序逐元素一致。

固定矩阵为 raw32/uniform_f32/uniform_f64 各 44 个 shape，其他四个 API 各 42 个 shape，
共 300 对、600 条原生/FlagRAND 记录。每个 shape 预热 5 次、正式 11 组、目标预算 200 ms。
两方复用 generator 和预分配输出，使用当前 HIP stream，编译预热不计入正式指标；
以同步端到端 `total_ms_median` 计算加速比。该计时协议保持已有海光基线不变。

完成后执行第 7 节，必须重新计算七个 API 的均值并检查完整原始矩阵。
详细硬件优化证据见 [海光指南](../usage/non-nvidia/hygon.md)。

## 7. 校验 summary.json 和每个 API 的原生性能

使用刚完成的验收根目录 `summary.json`，并保留其旁边的七个 API 子目录。
检查脚本只读取产物，可在计算节点或下载结果后的工作机运行，不占用 GPU。

```bash
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
"$PYTHON_BIN" tools/check_acceptance.py "$SUMMARY" \
  --output "$(dirname "$SUMMARY")/acceptance_check.json"
```

脚本核对根字段 `timestamp / total_duration / env / result`、五个交付项、正确性计数，
再按当前源码中各后端的固定配置核对每个 API 的全部 shape，直接从原生库和 FlagRAND 的
`records.jsonl` 重算 `total_ms_median` 比值，并与 `summary.json` 的逐 shape 值和均值核对。
`PHILOX4_32_10` 和 `RAW_BITS` 对应同一个 raw32 API，不算两种独立 API。

七个 API 均完整且均值 `>= 0.95x` 时，最后一行为：

```text
SUMMARY_JSON_CHECK=PASS
```

缺失或不支持的 API、非零正确性失败、原始记录与汇总不一致、API 均值不足，都会输出具体原因和
`SUMMARY_JSON_CHECK=FAIL`。保留原始 summary，不修改统计口径来满足检查。
BI-V150 当前 CoreX 原生 FP64 能力限制仍单独记录；四个可对照 API 通过不等于七个 API 全部通过。

查看完整结果：

```bash
"$PYTHON_BIN" -m json.tool "$SUMMARY"
```
