# FlagRAND 国产卡验收流程

适用后端：天数智芯 BI-V150、PPU-ZW810E、华为 Ascend 910C。

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

完成后执行第 4 节校验结果。

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

完成后执行第 4 节校验结果。FP64 精确构造专项显示 8 个 `skipped` 属于平台能力差异，不计为失败。

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

完成后执行第 4 节校验结果。

## 4. 校验 summary.json

以下命令使用前一节设置的 `SUMMARY` 路径，检查文件、FlagOS schema、交付项名称和数学正确性结果，
并打印各交付项的统计计数。

```bash
test -s "$SUMMARY"
.venv/bin/python - "$SUMMARY" <<'PY'
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
.venv/bin/python -m json.tool "$SUMMARY"
```
