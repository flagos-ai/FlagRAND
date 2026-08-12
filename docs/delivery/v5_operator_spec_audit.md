# FlagRAND v5 算子库规范审计报告

## 1. 审计范围

本次审计以《【试行】算子库规范》为准，覆盖融合后的 PPU-ZW810E 与 Huawei Ascend 910C
交付目录。审计对象包括项目结构、七 API 注册表、代码风格、正确性测试、性能测试、pytest marker、
CI、构建产物、平台隔离和硬件运行手册。

v5 不修改 `src/flagrand/**`。该目录已与 v4 交付包逐文件比较，内容完全一致；因此 PPU 3.6 与
910C 已通过实机验证的 kernel、host dispatch、cached launcher 和编译兼容逻辑均保持不变。

## 2. 规范符合性

| 规范项 | 状态 | v5 证据 |
|---|---|---|
| 项目结构 | PASS | 实现在 `src/flagrand/ops`、`fused`、`rng`、`runtime`、`ascend`；正确性位于 `tests/accuracy`；性能位于 `benchmark`；文档位于 `docs` |
| 算子注册表 | PASS | `conf/operators.yaml` 完整登记七个公开 API，ID 唯一，并关联公开接口、正确性文件和性能文件 |
| 注册表可移植性 | PASS | `tools/validate_operator_registry.py` 在有 PyYAML 时使用完整 YAML 解析；离线节点缺少 PyYAML 时使用只接受当前扁平注册表格式的标准库解析器 |
| Python 命名与日志 | PASS | 模块、函数和变量采用 snake_case，类采用 PascalCase，常量采用 UPPER_CASE；Python CLI 使用 `logging` |
| 代码风格 | PASS | Black、isort、flake8、YAML/TOML、尾随空格、文件结尾和 Python compileall 全部通过 |
| 正确性测试 | PASS（定义） | 覆盖 public API 契约、raw32 golden vector、seed/replay、bulk/split、offset、流语义、Uniform/Normal/LogNormal 统计特征和平台隔离；另有七个带独立 marker 的 910C pytest 测例 |
| pytest marker | PASS | 七个算子 marker 均已登记；性能测试改为一 API 一个参数项，`pytest -m <op>` 不再隐式执行完整七 API 矩阵；CUDA 正确性 marker 已细化到对应测例或参数 |
| 性能 shape | PASS | acceptance shape 统一维护，覆盖小、中、大及边界相邻 shape；Normal 类使用其独立合法 shape 集合 |
| Warmup 与重复测量 | PASS | smoke/full profile 分别配置 warmup、repeat 和 timing budget；正式统计使用 median |
| 计时与同步 | PASS | 每个 repeat 前后设备同步；正式指标为 `total_ms_median`，同时单独保存 enqueue 与 device event 指标 |
| 加速比聚合 | PASS | 先计算每 shape 的 baseline/FlagRAND，再在单 API 内取算术平均；禁止跨 API 平均 |
| PPU 正式入口 | PASS | `scripts/delivery/run_ppu_v5.sh` 一键完成环境、正确性、正式 acceptance、汇总和打包 |
| 910C 正式入口 | PASS | `scripts/delivery/run_ascend_910c_v5.sh` 只采用逐 API 独立进程，每个 API 单独执行 `0.95x` 门禁 |
| 普通 CI | PASS（定义） | `.github/workflows/ci.yml` 包含注册表、pre-commit、CPU-safe 测试、sdist/wheel 和 twine 检查 |
| 硬件 CI | PASS（定义） | `.github/workflows/hardware-ci.yml` 提供 PPU 与 910C 自托管任务；910C acceptance 调用同一逐 API 脚本 |
| 发布构建 | PASS | `python -m build --no-isolation` 成功生成 sdist/wheel，`twine check` 全部通过 |

## 3. v5 修正项

1. 删除 910C 同进程七 API 性能门禁及其运行手册章节。该协议会受到跨 API 运行时和设备状态影响，
   不再作为正式结论或候选结论输出。
2. 保留并固化逐 API 独立进程验收。七个 API 各自使用完整 acceptance shape、full warmup/repeat
   和 `0.95x` 门禁，任一子进程失败、缺少汇总或出现多个汇总都会使一键脚本失败。
3. 新增 PPU 与 910C 一键交付脚本。脚本通过仓库相对路径运行，不依赖某一次部署目录，并统一保存
   environment、preflight、pytest、performance 日志和结构化结果。
4. 硬件 CI 改为复用一键脚本，避免手工回归和 CI 使用不同命令。
5. 将 `benchmark/test_random_perf.py` 改成真正的单算子 marker 入口；正确性 marker 同步细化。
6. 注册表校验器增加无 PyYAML 的离线回退，避免硬件节点因无法联网安装构建依赖而跳过规范门禁。

## 4. 已执行验证与待执行验证

本地已经执行并通过：

- 七 API 注册表校验，包括强制标准库解析路径；
- Black、isort、flake8、YAML/TOML 与文本卫生检查；
- 全部 Python 文件 compileall 和全部 shell 文件 `bash -n`；
- `git diff --check`；
- sdist/wheel 构建和 `twine check`；
- v4/v5 `src/flagrand/**` 内容一致性比较。

本机没有 torch、Triton、PPU SDK 或 CANN，因此不伪造本地设备测试结果。v5 的最终硬件状态必须以
两张实卡执行包内一键脚本后的结果为准：

- PPU：`PPU V5 STATUS: PASS`，七 API acceptance 全部通过；
- 910C：`ASCEND 910C V5 STATUS: PASS`，七个独立日志分别且仅包含一个对应的 `API pass`。

上传、解压、一键运行和结果下载命令见
[`phase4_hardware_regression.md`](phase4_hardware_regression.md)。
