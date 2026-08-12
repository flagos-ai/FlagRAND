# FlagRAND v6 算子库规范审计报告

## 1. 审计范围

本次审计以《【试行】算子库规范》为准，覆盖融合后的 PPU-ZW810E 与 Huawei Ascend 910C 交付目录。
审计对象包括项目结构、七 API 注册表、代码风格、正确性测试、性能测试、pytest marker、CI、构建
产物、平台隔离、硬件运行手册，以及 v6 新增的 910C Normal/LogNormal FP32 host fast path。

## 2. v6 相对 v5 的受控增量

| 文件/区域 | 变更 | 隔离要求 |
|---|---|---|
| `src/flagrand/ascend/huawei910c.py` | 新增已校验 common-case direct cached launch；保留 epoch 回退 | 仅 Ascend 模块 |
| `src/flagrand/_curand_philox_fast_path.py` | NPU FP32 Normal/LogNormal 提前分派和 lazy import | 只在 NPU tensor 命中 |
| `src/flagrand/curand.py` | 精简两个 FP32 public API 成功路径的 Python 包装开销 | 错误类型与接口不变 |
| `tests/accuracy/test_philox_ascend_910c.py` | 增加跨 counter epoch replay/offset 测试 | 仅 910C marker |
| `scripts/delivery`、`scripts/ci` | 新增 v6 一键入口并使用独立 cache/result 命名 | PPU/910C 分离 |

PPU kernel、PPU 3.6 compiler compatibility proxy、CUDA/PPU backend 选择和平台专用动态库路径均未修改。

## 3. 规范符合性

| 规范项 | 状态 | v6 证据 |
|---|---|---|
| 项目结构 | PASS | 实现、测试、benchmark、配置、文档和脚本按既有目录分层 |
| 算子注册表 | PASS | 七个公开 API 均在 `conf/operators.yaml` 登记并关联测试和 benchmark |
| 接口兼容性 | PASS | public 函数签名、dtype 契约、异常类型和 generator offset 语义不变 |
| 平台隔离 | PASS（静态） | Ascend fast launcher 采用 NPU 条件分派和 lazy import；PPU 文件未改 |
| 正确性覆盖 | PASS | 新增 Normal/LogNormal counter epoch replay 与 offset 测试；既有统计测试保留 |
| 性能计时 | PASS | 正式指标仍为端到端 `total_ms_median`，每次测量前后同步 |
| 性能聚合 | PASS | baseline/FlagRAND 先按 shape 计算，再在单 API 内做算术平均，不跨 API 聚合 |
| 910C 正式入口 | PASS（实机） | 七 API 逐进程 acceptance 全部通过，每 API 均达到 `0.95x` 门禁 |
| PPU 正式入口 | PASS（实机） | 七 API acceptance 全部通过，PPU 3.6 compatibility proxy 生效 |
| 普通/硬件 CI | PASS（定义） | CI 调用 v6 一键入口，普通 CI 保留 registry、lint、pytest 和 build |
| 交付卫生 | PASS | 打包排除 cache、结果、日志、构建目录、AppleDouble 和 Python bytecode |

## 4. 硬件验收结果

910C 定向变更已在 CANN 9.0 / FlagTree Triton 3.5.1 / torch_npu 2.9.0.post2 环境验证：

- pytest：`13 passed`；
- `normal_f32` 三轮 API speedup：`1.175x`、`1.131x`、`1.130x`；
- `lognormal_f32` API speedup：`1.315x`；
- 所有定向性能记录：0 failure。

v6 交付包随后在两张实卡执行了完整一键回归，结果如下：

| API | Ascend 910C | PPU-ZW810E |
|---|---:|---:|
| raw32 | 1.353x | 1.061x |
| uniform_f32 | 1.017x | 0.951x |
| uniform_f64 | 0.989x | 1.118x |
| normal_f32 | 1.042x | 1.052x |
| normal_f64 | 0.976x | 0.959x |
| lognormal_f32 | 1.451x | 1.116x |
| lognormal_f64 | 1.000x | 0.960x |

两端均满足 `preflight_rc=0`、`pytest_rc=0`、`performance_rc=0`。910C 完成 216 条正式性能记录，
PPU 完成 324 条正式性能记录，均为 0 failure。

## 5. 最终结论

- PPU：`PPU V6 STATUS: PASS`；
- 910C：`ASCEND 910C V6 STATUS: PASS`；
- v6 已满足双平台七 API 正确性和端到端性能门禁，可以进入 Draft PR。

上传、解压、运行和结果下载命令见
[`phase4_hardware_regression.md`](phase4_hardware_regression.md)。
