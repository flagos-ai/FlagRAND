# FlagRAND 硬件验收流程

适用后端：NVIDIA H20、天数智芯 BI-V150、PPU-ZW810E、华为 Ascend 910C、沐曦 MetaX C550。

节点应已部署对应厂商的驱动和运行时，并具备 FlagRAND 仓库的 SSH 读取权限。验收固定从远端
`main` 获取源码，不复用已有源码目录、项目虚拟环境或历史结果目录。

验收通过条件：执行入口正常结束，生成 `summary.json`，交付项名称与 schema 正确，所有交付项
的准确性失败数为 0。

## 1. 天数智芯 BI-V150 / CoreX 4.4

### 1.1 获取源码

克隆远端 `main`，并记录本次验收的代码版本。

```bash
git clone --branch main --single-branch \
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

完成后执行第 6 节校验结果。

## 2. PPU-ZW810E

### 2.1 获取源码

克隆远端 `main`，并记录本次验收的代码版本。

```bash
git clone --branch main --single-branch \
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

完成后执行第 6 节校验结果。FP64 精确构造专项显示 8 个 `skipped` 属于平台能力差异，不计为失败。

## 3. 华为 Ascend 910C / CANN 9.0

### 3.1 获取源码

克隆远端 `main`，并记录本次验收的代码版本。

```bash
git clone --branch main --single-branch \
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

完成后执行第 6 节校验结果。

## 4. NVIDIA H20

### 4.1 获取源码

克隆远端 `main`，并记录本次验收的代码版本。

```bash
git clone --branch main --single-branch \
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

完成后执行第 6 节校验结果。

## 5. 沐曦 MetaX C550 / MACA AI 3.7.0.107

### 5.1 获取源码

在宿主机克隆远端 `main`，并记录本次验收的代码版本。MetaX 验收镜像不提供 Git，因此
源码由宿主机准备，再挂载进容器。

```bash
git clone --branch main --single-branch \
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

```bash
export FLAGRAND_RUN_ID="metax_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN=python3 \
FLAGRAND_PERF_PROFILE=acceptance \
  bash scripts/ci/run_backend_ci.sh metax
export SUMMARY="$PWD/muxi_results/ci_${FLAGRAND_RUN_ID}/summary.json"
```

统一 runner 在每个 API 的 `performance_raw/` 下保留 `benchmark.run_muxi` 原始产物，验收根目录的
`summary.json` 用于第 6 节的统一结果校验。

如需在同一进程中直接复核完整 MetaX 原始矩阵，执行：

```bash
python3 -m benchmark.run_muxi \
  --profile full \
  --acceptance-profile \
  --acceptance-baseline mcrand \
  --with-mcrand \
  --acceptance-min 0.95 \
  --acceptance-max 1.50 \
  --strict-acceptance \
  --skip-torch-reference
```

固定 shape 集包含 108 个 FlagRAND case 和 108 个 mcRAND case，共 216 条记录。通过时
输出包含 `Acceptance status: pass (7/7 APIs in target range)` 和
`Completed 216 records with 0 failure(s).`

pre-main 沐曦分支已产生过上述 216 条通过结果；将 NVIDIA 新 main 与 MetaX 独立分流
集成后，必须以新 `main` 的 C550 实测产物作为本次验收结论，不沿用 pre-main 性能数值。

完成后执行第 6 节校验结果。

## 6. 校验 summary.json

以下命令使用前一节设置的 `SUMMARY` 路径，检查文件、FlagOS schema、交付项名称和数学正确性结果，
并打印各交付项的统计计数。

```bash
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
test -s "$SUMMARY"
"$PYTHON_BIN" - "$SUMMARY" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
summary = json.loads(path.read_text(encoding="utf-8"))
expected = ["PHILOX4_32_10", "RAW_BITS", "UNIFORM", "NORMAL", "LOGNORMAL"]

assert set(summary) == {"timestamp", "total_duration", "env", "result"}
assert list(summary["result"]) == expected

for name in expected:
    result = summary["result"][name]
    assert set(result) == {"customized", "accuracy", "performance"}
    assert result["accuracy"]["failed"] == 0
    accuracy = result["accuracy"]
    print(
        f"{name}: status={accuracy['status']} "
        f"passed={accuracy['passed']} skipped={accuracy['skipped']} failed={accuracy['failed']}"
    )

print(f"SUMMARY_JSON_PATH={path}")
print("SUMMARY_JSON_CHECK=PASS")
PY
```

验收成功时，最后一行固定为：

```text
SUMMARY_JSON_CHECK=PASS
```

查看完整结果：

```bash
"$PYTHON_BIN" -m json.tool "$SUMMARY"
```
