# FlagRAND 算子开发与 CI 规范

## 算子登记

每个公开 API 必须在 `conf/operators.yaml` 中登记，并指向 `flagrand.ops.philox` 下的稳定接口、正确性测试和性能测试。`tools/validate_operator_registry.py` 会检查七个 API 是否完整、ID 是否唯一、测试文件及对应 pytest marker 是否存在。

## 正确性口径

随机算子不使用逐元素 CPU 浮点对齐代替统计学验证。测试分为：Philox raw32 golden vector、bulk/split 与 offset 流一致性、seed/replay、输出范围与有限性、Uniform 矩、Normal/Log-normal 均值方差及分布尾部。平台原生 FP64 行为按各 baseline 的真实契约验证，不能把存储 dtype 等同于随机有效位数。

## 性能口径

`benchmark` 中的正式数据均为端到端口径。每个 case 先 warmup，再重复计时并取中位数；计时区间在操作前后同步设备，同时保存 enqueue、device 和 total 指标。验收使用每个 shape 的 `baseline total_ms_median / FlagRAND total_ms_median`，然后在单 API 内聚合，不计算跨 API 平均值。

## CI 分层

- `.github/workflows/ci.yml`：普通 PR 上执行注册表、Black/isort/flake8、CPU 可执行测试和 wheel/sdist 构建。
- `.github/workflows/hardware-ci.yml`：仅手动触发，调度带 `ppu` 或 `ascend-910c` label 的自托管 runner，执行设备正确性和端到端性能矩阵。
- `scripts/ci/run_ppu_ci.sh` 与 `run_ascend_910c_ci.sh` 调用相应的一键交付脚本，直接通过 `PYTHONPATH` 加载源码，不触发离线节点的 PEP 517 构建隔离，也不替换厂商 torch、Triton 和运行时组合。910C acceptance 固定为逐 API 独立进程，不提供同进程性能门禁。

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
