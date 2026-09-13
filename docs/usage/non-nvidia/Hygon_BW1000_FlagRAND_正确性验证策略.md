# 海光 BW1000 · FlagRAND 正确性验证策略

本策略对应 2026-09-14 已完成的海光完整验收，覆盖七个 public Philox API。
实现提交为 `5fb6caa`，验收 run 为 `4abbd70eb84941879227f0c0c428e351`。
本次先核对已执行的校验及原始结果，再据此形成性能、正确性与实现报告。

## 1. 明确序列约定和比较对象

FlagRAND 使用自身约定的 Philox4x32-10 序列：counter 映射为
`(0, 0, counter_low32, counter_high32)`；seed 拆成两个 32-bit key。
raw32 和 FP32 每个输出推进一个 uint32 word，FP64 每个输出推进两个 word。
正态与对数正态通过成对的 Box–Muller 变换生成，输出元素数要求为偶数。

CPU 参考实现以 NumPy 整数运算独立执行十轮 Philox，再执行明确的分布变换，
不调用待测 Triton 内核。FlagRAND 按此参考校验数值；hipRAND 按其原生 Philox
host generator 的序列做独立重放、stream 和分布校验。两者的随机数排序约定不同，
因此性能对照不要求 FlagRAND 与 hipRAND 逐元素相等。

## 2. 分层校验和通过条件

| 层次 | 输入及范围 | 校验方法与判定 | 已执行来源 |
|---|---|---|---|
| 已知答案 | seed=0 的 16 个 raw32 值 | 与固定 golden vector 逐元素完全相等 | `test_philox_cuda.py` |
| Preflight | 七 API；offset=0、1、2、3、2^34−4；raw/uniform N=7，normal/lognormal N=8 | 独立 CPU 参考，35 个探针全部通过 | 各 API 的 `manifest.json` |
| 正式 shape 矩阵 | 300 个 API/shape 组合，seed=1234；覆盖小尺寸、分块边界附近和最大 16,777,216 元素 | 每种实现重新创建同 seed generator，两次生成的整段输出完全相等；FlagRAND 另与 CPU 参考比较前 min(N,128) 个值 | `records.jsonl` |
| 连续性和子序列 | 七 API 的整段/分段生成；raw32 指定 offset 子序列 | 分段拼接与整段输出完全相等；offset 按所消费 uint32 word 数推进 | `test_philox_cuda.py` |
| 高位和非对齐 | 40-bit 以上 seed；offset=0、1、3、2^34−4；从 `storage[1:]` 取得连续但非 16-byte 对齐输出 | CPU 参考校验；另以 64 个 raw32/FP32 uniform 输出跨越 2^33−4、2^34−4 附近边界 | `test_philox_hygon.py`、`test_hygon.py` |
| Stream | FlagRAND uniform 的非默认 stream；七 API 的 hipRAND 默认/非默认 stream | 同 seed 重放结果一致，并同步对应 stream | 两组准确性测试 |
| Binary64 参数 | N=4096；mean=0.200000000123、stddev=0.700000000321，另含标准分布及 stddev=0；四种 offset | 仿射结果、CPU double exp、常量输出和 2N offset；仿射/exp 使用 rtol=atol=1e-13，常量完全相等 | `test_hygon.py` |

独立 CPU 参考的判定为：raw32 完全相等；FP32 满足
`abs(actual-reference) <= 2e-6 + 2e-5 * abs(reference)`；FP64 满足
`abs(actual-reference) <= 1e-13 + 1e-12 * abs(reference)`。

正式矩阵中的 CPU 参考比较只覆盖每个输出的前 min(N,128) 个值；
整段相等检查覆盖同一实现的两次重放。二者分别记录，不把抽样 CPU 比较写成
全部输出都与 CPU 逐元素核对。下面的最大绝对误差也只对应这一参考抽样范围：

| API | CPU 参考校验的最大绝对误差 |
|---|---:|
| raw32 | 0 |
| uniform_f32 | 0 |
| uniform_f64 | 0 |
| normal_f32 | 5.4122e-07 |
| normal_f64 | 4.44089e-16 |
| lognormal_f32 | 1.14322e-06 |
| lognormal_f64 | 8.88178e-16 |

## 3. 分布统计

正式矩阵取每个输出的前 min(N,2^20) 个元素，检查有限值；raw32 检查非全零，
uniform 检查 `(0,1]` 范围，lognormal 检查正值。
样本量达到 4096 后，对 uniform 检查均值与 0.5 的差小于 0.04、
方差与 1/12 的差小于 0.01；对标准 normal 检查均值绝对值小于 0.04、
标准差位于 (0.96,1.04)；对 lognormal 的对数执行相同的标准正态统计检查。
这些条件对 FlagRAND 和 hipRAND 分别执行。

专项统计测试使用 N=262144。uniform 使用更紧的均值误差 0.004、方差误差 0.002；
normal(mean=0.25,stddev=1.5) 的均值及标准差误差均小于 0.015；
lognormal(mean=0.2,stddev=0.5) 取对数后的均值及标准差误差均小于 0.01。
统计结论对应上述样本量和矩检查范围。

## 4. 本次执行结果及计数口径

- Preflight：7 组、35 个探针，全部通过。
- 正式矩阵：FlagRAND 300 passed；hipRAND 300 passed；共 600 条记录，0 failed。
- 矩阵准确性 pytest：按测试 ID 去重，111 passed、8 skipped、0 failed。
  8 个跳过项为 CoreX 专用构造测试，在海光上不适用。
- 前置海光/API/后端协议测试：27 passed；统一 runner 测试：14 passed。
- 七 API 的 accuracy/performance 共 14 个阶段：全部通过。

同一个 pytest 用例可能被多个 API 的 marker 选中，逐 API 数字不能直接相加当作
独立用例数。300 个矩阵 case、111 个去重 pytest 用例和 35 个 preflight 探针
是不同计数口径，分别列示。

## 5. 实现与证据入口

- [CPU 参考实现](../../../benchmark/hygon_matrix/reference.py)
- [矩阵校验与 preflight](../../../benchmark/hygon_matrix/cases.py)
- [七 API 通用准确性测试](../../../tests/accuracy/test_philox_cuda.py)
- [海光 CPU 参考和 hipRAND stream 测试](../../../tests/accuracy/test_philox_hygon.py)
- [海光 binary64 与 counter 边界测试](../../../tests/accuracy/test_hygon.py)
- [性能、正确性与实现报告](./Hygon_BW1000_FlagRAND_性能正确性与实现说明.md)

完整原始产物在项目工作区 `reports/flagrand-hygon-baseline-20260914/full/`。
其 `acceptance/<API>/accuracy_result.json`、`performance_raw/*/manifest.json`、
`records.jsonl` 和 `ci_logs/` 分别提供用例、探针、矩阵记录和执行日志。
