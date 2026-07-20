# FlagRAND PPU 性能与正确性总结

## 结论

本次在 PPU-ZW810E 上完成了 Philox 路径下四类端到端随机数 API 的迁移、性能验证和正确性核验：

- Raw
- Uniform
- Normal
- LogNormal

其中 Raw、Uniform 支持任意长度；Normal、LogNormal 基于 Box-Muller，一次生成两个值，因此输出长度要求为偶数。

按“先对同一精度/类型下多个 shape 的加速比求平均，再对同一算子的不同精度/类型求平均，最后对四类 API 求平均”的口径，排除 16M 级别极大 shape 后，本次整体端到端结果为：

```text
FlagRAND vs PPU acrand = 1.015x
```

## 计算口径

单个 shape 的加速比：

```text
speedup(shape) = PPU acrand total time / FlagRAND total time
```

第一层，按 shape 聚合：

```text
precision_avg = mean(speedup(shape_1), speedup(shape_2), ...)
```

第二层，按同一算子的不同精度/类型聚合：

```text
op_avg = mean(precision_avg_1, precision_avg_2, ...)
```

第三层，四类 API 总平均：

```text
overall = mean(Raw_avg, Uniform_avg, Normal_avg, LogNormal_avg)
```

本次正式聚合纳入的 shape：

| API 组 | 纳入 shape |
| --- | --- |
| Raw / Uniform | `1,2,3,4,5,7,8,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,1023,1024,1025,4095,4096,4097,65535,65536,65537,1048575,1048576,1048577` |
| Normal / LogNormal | `2,4,6,8,10,14,16,18,126,128,130,1022,1024,1026,4094,4096,4098,65534,65536,65538,1048574,1048576,1048578` |

## 对标 API

本次性能对标采用 PPU SDK 中的 native `libacrand.so`。PPU SDK 同时提供 `libcurand.so` 兼容层，实测 `ppu_curand` 与 `ppu_acrand` 结果基本一致；文档主表使用 `ppu_acrand` 作为正式 baseline。

正式性能测试使用默认 fast/cached 路径，不设置 `FLAGRAND_FORCE_PORTABLE=1`。该变量仅用于兼容性排障；开启后会禁用 cached binary launcher，并可能让部分 f32 normal/lognormal 使用 portable math，因此不作为正式性能口径。

如果历史结果是在 `FLAGRAND_FORCE_PORTABLE=1` 的环境中生成，应按运行指南重新采集正式性能数据，再更新本节表格和聚合结果。

| 算子 | FlagRAND 端到端 API | PPU baseline API |
| --- | --- | --- |
| Raw32 | `curand.generate` | `acrandGenerate` |
| Uniform FP32 | `curand.generate_uniform` | `acrandGenerateUniform` |
| Uniform FP64 | `curand.generate_uniform_double` | `acrandGenerateUniformDouble` |
| Normal FP32 | `curand.generate_normal` | `acrandGenerateNormal` |
| Normal FP64 | `curand.generate_normal_double` | `acrandGenerateNormalDouble` |
| LogNormal FP32 | `curand.generate_lognormal` | `acrandGenerateLogNormal` |
| LogNormal FP64 | `curand.generate_lognormal_double` | `acrandGenerateLogNormalDouble` |

## Shape 平均结果

| 类型 | shape 数 | shape 平均加速比 |
| --- | ---: | ---: |
| Raw32 | 34 | 0.770x |
| Uniform FP32 | 34 | 0.715x |
| Uniform FP64 | 34 | 0.870x |
| Normal FP32 | 23 | 0.865x |
| Normal FP64 | 23 | 1.595x |
| LogNormal FP32 | 23 | 0.922x |
| LogNormal FP64 | 23 | 1.614x |

## 算子平均结果

| 算子 | 类型平均方式 | 算子平均加速比 |
| --- | --- | ---: |
| Raw | Raw32 | 0.770x |
| Uniform | mean(Uniform FP32, Uniform FP64) | 0.793x |
| Normal | mean(Normal FP32, Normal FP64) | 1.230x |
| LogNormal | mean(LogNormal FP32, LogNormal FP64) | 1.268x |

最终整体加速比：

```text
(0.770 + 0.793 + 1.230 + 1.268) / 4 = 1.015x
```

## 代表性明细

下表只列代表性 shape，完整平均使用上文列出的全部 shape。

| 算子 | N | 类型 | FlagRAND total ms | PPU acrand total ms | 加速比 |
| --- | ---: | --- | ---: | ---: | ---: |
| Raw32 | 1 | 小 N | 0.0161 | 0.0089 | 0.553x |
| Raw32 | 65537 | 奇数长度 | 0.0228 | 0.0098 | 0.430x |
| Raw32 | 1048576 | 对齐大 shape | 0.0137 | 0.0459 | 3.358x |
| Uniform FP32 | 1 | 小 N | 0.0171 | 0.0090 | 0.528x |
| Uniform FP32 | 65537 | 奇数长度 | 0.0238 | 0.0097 | 0.408x |
| Uniform FP32 | 1048576 | 对齐大 shape | 0.0141 | 0.0423 | 3.008x |
| Uniform FP64 | 1 | 小 N | 0.0152 | 0.0091 | 0.595x |
| Uniform FP64 | 65537 | 奇数长度 | 0.0175 | 0.0099 | 0.565x |
| Uniform FP64 | 1048576 | 对齐大 shape | 0.0131 | 0.0440 | 3.370x |
| Normal FP32 | 65538 | 非 4 倍数偶数 | 0.0210 | 0.0105 | 0.500x |
| Normal FP32 | 1048576 | 对齐大 shape | 0.0161 | 0.0543 | 3.372x |
| Normal FP64 | 65538 | 非 4 倍数偶数 | 0.0217 | 0.0161 | 0.739x |
| Normal FP64 | 1048576 | 对齐大 shape | 0.0236 | 0.1994 | 8.461x |
| LogNormal FP32 | 65538 | 非 4 倍数偶数 | 0.0211 | 0.0104 | 0.494x |
| LogNormal FP32 | 1048576 | 对齐大 shape | 0.0163 | 0.0616 | 3.776x |
| LogNormal FP64 | 65538 | 非 4 倍数偶数 | 0.0210 | 0.0191 | 0.909x |
| LogNormal FP64 | 1048576 | 对齐大 shape | 0.0293 | 0.2442 | 8.329x |

## 正确性核验

本次正确性核验采用结果导向，不依赖单一 oracle。

| 验证项 | 方法 | 结果 |
| --- | --- | --- |
| 预检查 | `ppu_flagrand_matrix.py --preflight-only`，覆盖 golden vector 和 bulk/split | pass |
| 回归测试 | `tests/test_api.py`、`tests/test_philox_cuda.py` | pass |
| 序列语义 | 相同 seed/offset 可复现，不同 seed/offset 输出隔离 | pass |
| 分段一致性 | 一次 bulk 生成与多次 split 生成结果一致 | pass |
| offset 泛用性 | 验证非 4 倍数 offset；修复 raw generic lane 顺序问题后通过 | pass |
| 长度泛用性 | raw/uniform 覆盖奇数长度；normal/lognormal 覆盖非 4 倍数偶数长度 | pass |
| 边界条件 | 零长度不推进 offset，大 offset 可复现 | pass |
| 分布统计 | Uniform/Raw 检查范围、均值、方差、KS、chi-square；Normal/LogNormal 检查均值、标准差、偏度、峰度、KS | pass |
| 独立性 | autocorrelation、runs test、serial test | pass |

PPU SDK 的 `acrand/libcurand` 与当前 Triton Philox 输出未做到 bitwise 对齐，PPU SDK 仅作为端到端性能 baseline；正确性依据来自 golden vector、序列语义、边界行为和统计检验。
