# FlagRAND 算子开发与 CI 规范

## 后端目录

沿用 FlagGems 的公共入口、厂商发现和后端覆盖方式：

```text
src/flagrand/
├── ops/                         # 稳定的分布 API
├── rng/                         # 通用 Philox
├── fused/_internal/             # 通用融合生成和变换
└── runtime/backend/
    ├── device_finder.py         # 厂商发现
    ├── backend_utils.py         # 厂商描述
    └── _<vendor>/               # 厂商注册、算子覆盖和硬件特化
```

海光新增实现集中在 `_hygon/ops.py`；原生库只供 `benchmark/hygon_matrix/` 对照。
其他后端保留现有 `corex/`、`metax/`、`nvidia/`、`ascend/` 内核模块，由相应
`runtime/backend/_<vendor>/ops.py` 注册调用。新增厂商特化沿用后端目录，公共代码保持跨芯片可复用。

## 算子登记

每个公开 API 必须在 `conf/operators.yaml` 中登记，并指向 `flagrand.ops.philox` 下的稳定接口、正确性测试和性能测试。`tools/validate_operator_registry.py` 会检查七个 API 是否完整、ID 是否唯一、测试文件及对应 pytest marker 是否存在。

## 正确性口径

随机算子不使用逐元素 CPU 浮点对齐代替统计学验证。测试分为：Philox raw32 golden vector、bulk/split 与 offset 流一致性、seed/replay、输出范围与有限性、Uniform 矩、Normal/Log-normal 均值方差及分布尾部。平台原生 FP64 行为按各 baseline 的真实契约验证，不能把存储 dtype 等同于随机有效位数。

## 性能口径

`benchmark` 中的正式数据均为端到端口径。每个 case 先 warmup，再重复计时并取中位数；计时区间在操作前后同步设备，同时保存 enqueue、device 和 total 指标。验收使用每个 shape 的 `baseline total_ms_median / FlagRAND total_ms_median`，然后在单 API 内聚合，不计算跨 API 平均值。

## CI 分层

- `.github/workflows/linter.yml`：执行代码风格检查。
- `.github/workflows/build-check.yml`：构建并检查 wheel/sdist。
- `.github/workflows/unittest.yaml`：根据 `.github/backends.json` 选择硬件后端，并调用可复用的 `.github/workflows/backend-test.yaml`。
- 四个自托管 runner 标签分别为 `nvidia-h20`、`iluvatar`、`thead` 和 `cann9`，对应 `nvidia`、`iluvatar`、`thead`、`ascend-cann900` 后端。
- 海光使用 `hygon` 后端，复用统一 runner；原生基准是 DTK hipRAND。代码中的 Actions runner 标签为 `hygon`，需要仓库实际登记同名自托管 runner 后才能由 GitHub 启动；离线交付不依赖它。
- MetaX 使用统一后端名 `metax`（兼容 `muxi`、`maca` 别名）和相同的 CI 脚本；当前 C550
  节点尚未登记 GitHub Actions runner label，因此先按硬件验收文档在容器内手工运行。
- `scripts/ci/run_backend_ci.sh <backend>` 是统一硬件入口；它执行环境记录、注册表检查、统一 runner 测试、正确性与性能矩阵，并生成交付 `summary.json`。

正式 acceptance 示例：

```bash
./setup.sh thead
source scripts/ci/backend_env.sh thead
FLAGRAND_RUN_ID="ppu_$(date +%Y%m%d_%H%M%S)" \
FLAGRAND_PERF_PROFILE=acceptance \
bash scripts/ci/run_backend_ci.sh thead
```

后端与结果目录的映射为：

| 后端 | 结果目录 |
|---|---|
| `nvidia` | `nvidia_results/ci_<RUN_ID>/summary.json` |
| `hygon` | `hygon_results/ci_<RUN_ID>/summary.json` |
| `metax` | `muxi_results/ci_<RUN_ID>/summary.json` |
| `iluvatar` | `corex_results/ci_<RUN_ID>/summary.json` |
| `thead` | `ppu_results/ci_<RUN_ID>/summary.json` |
| `ascend-cann900` | `ascend_910c_results/ci_<RUN_ID>/summary.json` |

本地提交前执行：

```bash
python tools/validate_operator_registry.py
pre-commit run --all-files
pytest -q tests/accuracy
```

也可以执行统一静态入口：

```bash
bash scripts/check_quality.sh
```
