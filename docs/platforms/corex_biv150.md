# FlagRAND 天数智芯 BI-V150 / CoreX 4.4 运行指南

## 验证环境

统一 CI 后端名称为 `iluvatar`。当前依赖由 `conf/backends.yaml` 固定为 Python 3.10、
`torch==2.7.1+corex.4.4.0`、`triton==3.1.0+corex.4.4.0` 和 CoreX 4.4；
运行前通过 `tools/gpu_check_iluvatar.sh` 验证 BI-V150 设备、Torch 和 Triton target。
`setup.sh iluvatar` 使用 `/usr/local/bin/python3` 创建带 `--system-site-packages` 的
项目虚拟环境，复用卡上预装的 CoreX Torch/Triton，不重新下载或替换厂商运行时。

## 完整验收

```bash
git clone --branch main --single-branch git@github.com:flagos-ai/FlagRAND.git
cd FlagRAND

./setup.sh iluvatar
source scripts/ci/backend_env.sh iluvatar
bash tools/gpu_check_iluvatar.sh

RUN_ID="corex_$(date +%Y%m%d_%H%M%S)"
FLAGRAND_RUN_ID="$RUN_ID" \
FLAGRAND_PERF_PROFILE=acceptance \
bash scripts/ci/run_backend_ci.sh iluvatar
```

验收结果位于：

```text
corex_results/ci_<RUN_ID>/summary.json
```

查看结果：

```bash
.venv/bin/python -m json.tool "corex_results/ci_${RUN_ID}/summary.json"
```

`summary.json` 的 `result` 只包含 `PHILOX4_32_10`、`RAW_BITS`、`UNIFORM`、
`NORMAL` 和 `LOGNORMAL`。FP32/FP64 的准确性与性能明细聚合在对应的分布交付项下。

## GitHub Actions

在 `unit-test` workflow 中选择：

```text
backend: iluvatar
profile: acceptance
```

任务会调度到标签为 `iluvatar` 的自托管 BI-V150 runner，并上传
`flagrand-iluvatar-<run_id>-<attempt>` 产物。
